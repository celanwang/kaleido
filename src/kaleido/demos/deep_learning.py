"""P2 CPU experiments on Digits, first-bit memory, attention and LineWorld."""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from kaleido.core import Tensor
from kaleido.core.attention import scaled_dot_product_attention
from kaleido.core.losses import cross_entropy
from kaleido.core.models import DigitCNN, GaussianVAE, MemoryClassifier, vae_loss
from kaleido.core.nn import MLP
from kaleido.core.optim import SGD, Adam
from kaleido.experiments import Run, array_digest
from kaleido.rl import evaluate, train_policy, train_q_learning

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Config:
    seeds: tuple = (42, 7, 2026)
    epochs: int = 12
    vae_epochs: int = 20
    batch_size: int = 64
    memory_updates: int = 150
    rl_updates: int = 60
    rl_episodes: int = 12

    def __post_init__(self):
        if (
            not isinstance(self.seeds, (list, tuple))
            or not self.seeds
            or len(set(self.seeds)) != len(self.seeds)
        ):
            raise ValueError("seeds must be unique and non-empty")
        if any(type(s) is not int or not 0 <= s < 2**32 - 10000 for s in self.seeds):
            raise ValueError("seeds must be non-negative 32-bit integers with stream offset space")
        for name in (
            "epochs",
            "vae_epochs",
            "batch_size",
            "memory_updates",
            "rl_updates",
            "rl_episodes",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")


def _classifier_metrics(model, x, y):
    logits = model(Tensor(x))
    return (
        float(cross_entropy(logits, y).data),
        float(np.mean(logits.data.argmax(axis=1) == y)),
        logits.data,
    )


def _train_classifier(model, x, y, splits, optimizer, *, epochs, batch_size, seed):
    rng = np.random.default_rng(seed + 100)
    tr, va, te = (splits[k] for k in ("train", "valid", "test"))
    initial = _classifier_metrics(model, x[va], y[va])[0]
    best, best_epoch, best_loss = model.state_dict(), 0, initial
    history = [{"epoch": 0, "valid_nll": initial}]
    for epoch in range(1, epochs + 1):
        order = rng.permutation(tr)
        for start in range(0, len(order), batch_size):
            ids = order[start : start + batch_size]
            optimizer.zero_grad()
            cross_entropy(model(Tensor(x[ids])), y[ids]).backward()
            optimizer.step()
        loss, accuracy, _ = _classifier_metrics(model, x[va], y[va])
        history.append({"epoch": epoch, "valid_nll": loss, "valid_accuracy": accuracy})
        if loss < best_loss:
            best, best_epoch, best_loss = model.state_dict(), epoch, loss
    model.load_state_dict(best)
    loss, accuracy, logits = _classifier_metrics(model, x[te], y[te])
    return (
        {
            "test_nll": loss,
            "test_accuracy": accuracy,
            "selected_epoch": best_epoch,
            "valid_nll": best_loss,
            "initial_valid_nll": initial,
        },
        history,
        logits,
    )


def _vae_metrics(model, x, noise):
    totals = []
    for epsilon in noise:
        logits, mean, logvar = model(Tensor(x), epsilon)
        total, reconstruction, kl = vae_loss(logits, Tensor(x), mean, logvar)
        totals.append([float(total.data), float(reconstruction.data), float(kl.data)])
    total, reconstruction, kl = np.mean(totals, axis=0)
    mean, _ = model.encode(Tensor(x))
    reconstruction_mean = model.decode(mean).sigmoid().data
    return {
        "negative_elbo": float(total),
        "reconstruction_nll": float(reconstruction),
        "kl": float(kl),
        "mean_latent_reconstruction_mse": float(np.mean((reconstruction_mean - x) ** 2)),
    }, reconstruction_mean


def _train_vae(model, x, splits, config, *, seed, beta):
    rng = np.random.default_rng(seed + 200)
    eval_rng = np.random.default_rng(seed + 300)
    tr, va, te = (splits[k] for k in ("train", "valid", "test"))
    noise_valid = eval_rng.normal(size=(3, len(va), 4))
    noise_test = eval_rng.normal(size=(3, len(te), 4))
    optimizer = Adam(model.parameters(), lr=0.005)
    initial, _ = _vae_metrics(model, x[va], noise_valid)
    best, best_epoch, best_loss = model.state_dict(), 0, initial["negative_elbo"]
    history = [{"epoch": 0, **initial}]
    for epoch in range(1, config.vae_epochs + 1):
        order = rng.permutation(tr)
        for start in range(0, len(order), config.batch_size):
            ids = order[start : start + config.batch_size]
            target = Tensor(x[ids])
            optimizer.zero_grad()
            logits, mean, logvar = model(target, rng.normal(size=(len(ids), 4)))
            loss, _, _ = vae_loss(logits, target, mean, logvar, beta=beta)
            loss.backward()
            optimizer.step()
        metrics, _ = _vae_metrics(model, x[va], noise_valid)
        history.append({"epoch": epoch, **metrics})
        if metrics["negative_elbo"] < best_loss:
            best, best_epoch, best_loss = model.state_dict(), epoch, metrics["negative_elbo"]
    final = model.state_dict()
    final_test, _ = _vae_metrics(model, x[te], noise_test)
    model.load_state_dict(best)
    metrics, prediction = _vae_metrics(model, x[te], noise_test)
    prior = eval_rng.normal(size=(16, 4))
    samples = model.decode(Tensor(prior)).sigmoid().data
    metrics.update(
        selected_epoch=best_epoch,
        valid_negative_elbo=best_loss,
        initial_valid_negative_elbo=initial["negative_elbo"],
        final_budget_kl=final_test["kl"],
        final_budget_reconstruction_nll=final_test["reconstruction_nll"],
    )
    return metrics, history, prediction, samples, prior, noise_test, final


def _memory(config, seed):
    rng = np.random.default_rng(seed + 400)

    def dataset(count, length):
        x = rng.integers(0, 2, size=(count, length, 1))
        return x * 2.0 - 1, x[:, 0, 0].copy()

    train, valid, test, long = dataset(128, 6), dataset(64, 6), dataset(128, 6), dataset(128, 12)
    model = MemoryClassifier(rng=np.random.default_rng(seed))
    optimizer = Adam(model.parameters(), lr=0.02)
    best, best_loss, best_epoch = model.state_dict(), _classifier_metrics(model, *valid)[0], 0
    initial = best_loss
    history = [{"update": 0, "valid_nll": best_loss}]
    for step in range(1, config.memory_updates + 1):
        optimizer.zero_grad()
        cross_entropy(model(Tensor(train[0])), train[1]).backward()
        optimizer.step()
        loss, accuracy, _ = _classifier_metrics(model, *valid)
        history.append({"update": step, "valid_nll": loss, "valid_accuracy": accuracy})
        if loss < best_loss:
            best, best_loss, best_epoch = model.state_dict(), loss, step
    model.load_state_dict(best)
    loss, accuracy, logits = _classifier_metrics(model, *test)
    long_loss, long_accuracy, long_logits = _classifier_metrics(model, *long)
    metrics = {
        "test_nll": loss,
        "test_accuracy": accuracy,
        "long_sequence_nll": long_loss,
        "long_sequence_accuracy": long_accuracy,
        "selected_update": best_epoch,
        "last_bit_baseline_accuracy": float(np.mean((test[0][:, -1, 0] > 0) == test[1])),
        "initial_valid_nll": initial,
        "valid_nll": best_loss,
    }
    arrays = {
        f"{name}_{part}": value
        for name, pair in (("train", train), ("valid", valid), ("test", test), ("long", long))
        for part, value in zip(("x", "y"), pair, strict=True)
    }
    arrays.update(test_logits=logits, long_logits=long_logits)
    return model, metrics, history, arrays


def _attention(seed):
    import torch

    rng = np.random.default_rng(seed + 500)
    arrays = [rng.normal(size=(6, 4)) for _ in range(3)]
    ours = [Tensor(v, True) for v in arrays]
    refs = [torch.tensor(v, requires_grad=True) for v in arrays]
    mask = np.tril(np.ones((6, 6), bool))
    output, prob = scaled_dot_product_attention(*ours, mask=mask)
    expected = torch.nn.functional.scaled_dot_product_attention(*refs, is_causal=True)
    (output**2).mean().backward()
    (expected**2).mean().backward()
    changed = arrays[2].copy()
    changed[3:] += 100
    altered, _ = scaled_dot_product_attention(
        Tensor(arrays[0]), Tensor(arrays[1]), Tensor(changed), mask=mask
    )
    zero = Tensor(np.zeros((6, 4)))
    uniform, _ = scaled_dot_product_attention(zero, zero, Tensor(arrays[2]), mask=mask)
    expected_mean = np.cumsum(arrays[2], axis=0) / np.arange(1, 7)[:, None]
    checks = {
        "forward_max_abs_error": float(np.max(np.abs(output.data - expected.detach().numpy()))),
        "gradient_max_abs_error": max(
            float(np.max(np.abs(a.grad - b.grad.numpy()))) for a, b in zip(ours, refs, strict=True)
        ),
        "future_perturbation_prefix_error": float(
            np.max(np.abs(output.data[:3] - altered.data[:3]))
        ),
        "uniform_prefix_mean_error": float(np.max(np.abs(uniform.data - expected_mean))),
    }
    if max(checks.values()) > 1e-10:
        raise RuntimeError("attention reference comparison failed")
    return checks, {
        "query": arrays[0],
        "key": arrays[1],
        "value": arrays[2],
        "probability": prob.data,
        "output": output.data,
        "reference_output": expected.detach().numpy(),
    }


def _rl(config, seed):
    metrics, history, arrays, models = {}, {}, {}, {}
    random = evaluate(lambda s: np.array([0.5, 0.5]), seed=seed + 1000)
    metrics["random"] = {k: random[k] for k in ("mean_return", "success_rate")}
    for name in ("returns", "successes", "steps"):
        arrays["random_" + name] = np.array(random[name])
    q, q_history = train_q_learning(seed=seed)
    history["q_learning"] = q_history
    arrays["q_table"] = q
    for method in ("q_learning", "reinforce", "ppo"):
        if method == "q_learning":
            probability = np.eye(2)[q.argmax(axis=1)]
        else:
            model, probability, curve = train_policy(
                seed=seed,
                method=method,
                updates=config.rl_updates,
                episodes_per_update=config.rl_episodes,
            )
            history[method] = curve
            models[method] = model.state_dict()
        stochastic = evaluate(lambda s, p=probability: p[s], seed=seed + 1000)
        greedy = evaluate(lambda s, p=probability: p[s], seed=seed + 2000, greedy=True)
        metrics[method] = {
            "mean_return": stochastic["mean_return"],
            "success_rate": stochastic["success_rate"],
            "greedy_mean_return": greedy["mean_return"],
            "greedy_success_rate": greedy["success_rate"],
        }
        arrays[method + "_probability"] = probability
        for key in ("returns", "successes", "steps"):
            arrays[method + "_" + key] = np.array(stochastic[key])
            arrays[method + "_greedy_" + key] = np.array(greedy[key])
    return metrics, history, arrays, models


def _save_npz(path, arrays):
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def execute(config: Config, output: Path, *, root=ROOT):
    try:
        import torch
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split
    except ImportError as error:
        raise RuntimeError("install P2 references: uv sync --locked --all-extras") from error

    torch.set_num_threads(1)
    with Run(output, asdict(config), root=root) as run:
        start = time.perf_counter()
        dataset = load_digits()
        x, y = dataset.data / 16.0, dataset.target
        metrics, histories, states, outputs, metadata, checks = (
            {},
            {},
            {},
            {},
            {},
            {},
        )
        data = {"digits_x": x, "digits_y": y}

        def store_state(prefix, state):
            for name, value in state.items():
                states[prefix + "__" + name] = value

        for seed in config.seeds:
            key = str(seed)
            metrics[key] = {}
            histories[key] = {}
            tr, held = train_test_split(
                np.arange(len(y)), test_size=0.4, random_state=seed, stratify=y
            )
            va, te = train_test_split(held, test_size=0.5, random_state=seed + 1, stratify=y[held])
            splits = dict(zip(("train", "valid", "test"), (tr, va, te), strict=True))
            metadata[key] = {"splits": {k: v.tolist() for k, v in splits.items()}}
            outputs[key + "__test_ids"] = te
            for name in ("mlp_sgd", "mlp_adam", "cnn"):
                rng = np.random.default_rng(seed)
                model = DigitCNN(rng=rng) if name == "cnn" else MLP([64, 32, 10], rng=rng)
                inputs = x.reshape(-1, 1, 8, 8) if name == "cnn" else x
                optimizer = (
                    SGD(model.parameters(), lr=0.1, momentum=0.9)
                    if name == "mlp_sgd"
                    else Adam(model.parameters(), lr=0.01)
                )
                result, curve, logits = _train_classifier(
                    model,
                    inputs,
                    y,
                    splits,
                    optimizer,
                    epochs=config.epochs,
                    batch_size=config.batch_size,
                    seed=seed,
                )
                metrics[key][name] = result
                histories[key][name] = curve
                store_state(key + "__" + name, model.state_dict())
                outputs[key + "__" + name + "_logits"] = logits
            mean = x[tr].mean(axis=0)
            metrics[key]["baselines"] = {
                "majority_accuracy": float(np.mean(y[te] == np.bincount(y[tr]).argmax())),
                "mean_pixel_mse": float(np.mean((x[te] - mean) ** 2)),
                "mean_pixel_nll": float(
                    -np.mean(
                        (
                            x[te] * np.log(np.clip(mean, 1e-8, 1 - 1e-8))
                            + (1 - x[te]) * np.log(np.clip(1 - mean, 1e-8, 1 - 1e-8))
                        ).sum(axis=1)
                    )
                ),
            }
            for beta in (1.0, 0.0):
                name = "vae" if beta == 1 else "vae_no_kl"
                model = GaussianVAE(64, 16, 4, rng=np.random.default_rng(seed))
                result, curve, prediction, samples, prior, noise, final = _train_vae(
                    model, x, splits, config, seed=seed, beta=beta
                )
                metrics[key][name] = result
                histories[key][name] = curve
                store_state(key + "__" + name, model.state_dict())
                store_state(key + "__" + name + "_final", final)
                outputs[key + "__" + name + "_reconstruction"] = prediction
                outputs[key + "__" + name + "_prior_samples"] = samples
                outputs[key + "__" + name + "_prior_z"] = prior
                outputs[key + "__" + name + "_test_epsilon"] = noise
            model, result, curve, arrays = _memory(config, seed)
            metrics[key]["memory"] = result
            histories[key]["memory"] = curve
            store_state(key + "__memory", model.state_dict())
            for name, value in arrays.items():
                destination = outputs if name.endswith("logits") else data
                destination[key + "__memory_" + name] = value
            result, curve, arrays, rl_states = _rl(config, seed)
            metrics[key]["rl"] = result
            histories[key]["rl"] = curve
            for name, value in arrays.items():
                outputs[key + "__rl_" + name] = value
            for name, state in rl_states.items():
                store_state(key + "__rl_" + name, state)
            checks[key], arrays = _attention(seed)
            for name, value in arrays.items():
                outputs[key + "__attention_" + name] = value
        _save_npz(run.path / "data.npz", data)
        _save_npz(run.path / "models.npz", states)
        _save_npz(run.path / "outputs.npz", outputs)
        run.record(
            "data.json",
            {
                "source": (
                    "sklearn bundled Digits, pixels /16; seeded synthetic memory and LineWorld"
                ),
                "digits_sha256": array_digest(x, y),
                "splits": metadata,
                "array_digests": {k: array_digest(v) for k, v in data.items()},
            },
        )
        run.record(
            "models.json",
            {
                "array_keys": list(states),
                "array_digests": {k: array_digest(v) for k, v in states.items()},
                "note": (
                    "model snapshots are inference checkpoints; "
                    "optimizer moments/RNG are not resume checkpoints"
                ),
            },
        )
        run.record(
            "outputs.json",
            {
                "array_keys": list(outputs),
                "array_digests": {k: array_digest(v) for k, v in outputs.items()},
            },
        )
        run.record("metrics.json", metrics)
        run.record("histories.json", histories)
        run.record("checks.json", checks)
        run.record(
            "protocol.json",
            {
                "digits": {
                    "normalization": "pixels /16, known range; no fitted dataset statistics",
                    "split": "60/20/20 stratified",
                    "mlp_sizes": [64, 32, 10],
                    "cnn": "4x3x3 valid + ReLU + 144->10",
                    "sgd": {"lr": 0.1, "momentum": 0.9},
                    "adam": {"lr": 0.01, "betas": [0.9, 0.999], "eps": 1e-8},
                    "matched_initialization_and_batch_order": "MLP SGD versus Adam",
                    "selection": "minimum validation NLL; test excluded",
                },
                "vae": {
                    "dimensions": [64, 16, 4],
                    "lr": 0.005,
                    "betas_compared": [1.0, 0.0],
                    "posterior_logvar_bounds": [-10, 10],
                    "evaluation_noise_samples": 3,
                    "selection": "common beta=1 validation negative ELBO for both objectives",
                    "ablation": "also compare final equal-budget reconstruction and KL",
                },
                "memory": {
                    "lengths": [6, 12],
                    "train_valid_test": [128, 64, 128],
                    "lr": 0.02,
                    "hidden_size": 8,
                },
                "rl": {
                    "q_episodes": 600,
                    "q_lr": 0.2,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "ppo_epochs": 4,
                    "ppo_clip": 0.2,
                    "policy_lr": 0.03,
                    "ppo_value_coef": 0.5,
                    "ppo_entropy_coef": 0.01,
                    "eval_episodes": 200,
                    "eval_seed_offsets": [1000, 2000],
                    "note": "different training budgets; no efficiency ranking",
                },
            },
        )
        run.record(
            "cost.json",
            {"wall_seconds": time.perf_counter() - start, "device": "cpu", "dtype": "float64"},
        )
    return run.path, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/p2_deep.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/p2-deep"))
    args = parser.parse_args()
    try:
        config = Config(**json.loads(args.config.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    path, metrics = execute(config, args.output)
    print(json.dumps({"run": str(path.resolve()), "metrics": metrics}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
