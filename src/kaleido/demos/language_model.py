"""P3: pinned text -> train-only BPE -> actual pretraining -> exact cached decoding."""

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from kaleido.experiments import Run, array_digest
from kaleido.lm.corpus import prepare_corpus
from kaleido.lm.inference import benchmark_decode, cache_equivalence, generate
from kaleido.lm.model import DecoderConfig, DecoderLM
from kaleido.lm.tokenizer import EOS, ByteBPE
from kaleido.lm.training import (
    TokenCorpus,
    count_baselines,
    evaluate,
    fit,
    load_weights,
    probe_resources,
    save_weights,
)

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Config:
    seeds: tuple = (42, 7, 2026)
    vocab_size: int = 512
    context_length: int = 128
    width: int = 128
    layers: int = 2
    heads: int = 4
    batch_size: int = 16
    steps: int = 1000
    learning_rate: float = 0.0008
    warmup_steps: int = 50
    eval_interval: int = 100
    split_seed: int = 17
    device: str = "auto"
    probe_repeats: int = 5
    benchmark_repeats: int = 5
    generation_tokens: int = 48

    def __post_init__(self):
        DecoderConfig(
            vocab_size=self.vocab_size,
            context_length=self.context_length,
            width=self.width,
            layers=self.layers,
            heads=self.heads,
        )
        if self.vocab_size < 257:
            raise ValueError("vocab_size must include all bytes and EOS")
        for name in (
            "batch_size",
            "steps",
            "eval_interval",
            "probe_repeats",
            "benchmark_repeats",
            "generation_tokens",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not isinstance(self.seeds, (tuple, list))
            or not self.seeds
            or len(set(self.seeds)) != len(self.seeds)
            or any(type(s) is not int or not 0 <= s < 2**32 - 100 for s in self.seeds)
            or type(self.split_seed) is not int
            or self.split_seed < 0
        ):
            raise ValueError("invalid unique seeds or split seed")
        if (
            type(self.warmup_steps) is not int
            or not 0 <= self.warmup_steps < self.steps
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
            or self.device not in ("auto", "cpu", "mps")
        ):
            raise ValueError("invalid learning rate, warmup or device")


def execute(config: Config, output: Path, *, root=ROOT, corpus_dir=ROOT / "data/p3"):
    original_threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        return _execute(config, output, root=root, corpus_dir=corpus_dir)
    finally:
        torch.set_num_threads(original_threads)
        torch.use_deterministic_algorithms(deterministic)


def _execute(config, output, *, root, corpus_dir):
    with Run(output, asdict(config), root=root) as run:
        start = time.perf_counter()
        documents, audit = prepare_corpus(corpus_dir, split_seed=config.split_seed)
        train_texts = [d["text"] for d in documents if d["split"] == "train"]
        tokenizer = ByteBPE.fit(train_texts, vocab_size=config.vocab_size)
        tokenizer.save(run.path / "tokenizer.json")
        corpus = TokenCorpus(documents, tokenizer)
        model_config = DecoderConfig(
            vocab_size=len(tokenizer),
            context_length=config.context_length,
            width=config.width,
            layers=config.layers,
            heads=config.heads,
        )
        probe = probe_resources(
            model_config, corpus, batch_size=config.batch_size, repeats=config.probe_repeats
        )
        selected = (
            probe["selected"]
            if config.device == "auto"
            else min(
                [
                    p
                    for p in probe["candidates"]
                    if p["status"] == "ok" and p["device"] == config.device
                ],
                key=lambda row: row["median_step_seconds"],
                default=None,
            )
        )
        if selected is None:
            raise RuntimeError("requested device did not complete the resource probe")
        device = selected["device"]
        torch.set_num_threads(selected["cpu_threads"])
        probe["used"] = {"device": device, "cpu_threads": selected["cpu_threads"]}
        run.record("resources.json", probe)
        run.record(
            "corpus.json",
            {
                "audit": audit,
                "documents": [
                    {k: v for k, v in doc.items() if k not in ("paragraphs", "text")}
                    for doc in documents
                ],
                "model_config": asdict(model_config),
                "tokenizer_fit_document_ids": [d["id"] for d in documents if d["split"] == "train"],
                "token_counts": {
                    s: sum(len(d["tokens"]) - 1 for d in group)
                    for s, group in corpus.splits.items()
                },
                "bytes_per_non_eos_token": {
                    s: sum(len(d["text"].encode()) for d in group)
                    / sum(len(d["tokens"]) - 2 for d in group)
                    for s, group in corpus.splits.items()
                },
            },
        )
        token_arrays = {d["id"]: d["tokens"] for d in corpus.documents}
        np.savez_compressed(run.path / "tokens.npz", **token_arrays)
        run.record("tokens.json", {"arrays": {k: array_digest(v) for k, v in token_arrays.items()}})
        baselines = {s: count_baselines(corpus, s) for s in ("valid", "test")}
        results, histories, generations, benchmarks, checks, details = {}, {}, {}, {}, {}, {}
        evaluation_arrays = {}
        prompts = (
            "Alice looked at the little door and said",
            "The Scarecrow and the Tin Woodman",
            "The queen turned to the little girl",
        )
        for seed in config.seeds:
            key = str(seed)
            torch.manual_seed(seed)
            model = DecoderLM(model_config).to(device)
            initial_sha = array_digest(
                *[v.detach().cpu().numpy() for v in model.state_dict().values()]
            )
            initial_test, initial_detail, initial_losses = evaluate(
                model, corpus, "test", batch_size=config.batch_size
            )
            training, history = fit(
                model,
                corpus,
                steps=config.steps,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                warmup_steps=config.warmup_steps,
                eval_interval=config.eval_interval,
                seed=seed,
            )
            final_test, detail, losses = evaluate(
                model, corpus, "test", batch_size=config.batch_size
            )
            selected_valid, valid_detail, valid_losses = evaluate(
                model, corpus, "valid", batch_size=config.batch_size
            )
            weights = save_weights(run.path / f"model-{seed}.npz", model)
            restored = load_weights(run.path / f"model-{seed}.npz", device=device)
            sample = (
                torch.from_numpy(
                    corpus.splits["test"][0]["tokens"][: min(32, config.context_length)].copy()
                )
                .unsqueeze(0)
                .to(device)
            )
            with torch.inference_mode():
                reload_error = float((model(sample)[0] - restored(sample)[0]).abs().max().cpu())
            if reload_error != 0:
                raise RuntimeError("checkpoint logits changed on same-device reload")
            cache_check = cache_equivalence(restored, sample, chunk_sizes=(3, 1, 4))
            if cache_check["max_abs_logit_error"] > 2e-5 or not cache_check["greedy_agreement"]:
                raise RuntimeError("cached logits failed numerical/greedy comparison")
            checks[key] = {"checkpoint_reload_max_abs_error": reload_error, **cache_check}
            results[key] = {
                "training": training,
                "initial_test": initial_test,
                "selected_test": final_test,
                "selected_valid": selected_valid,
                "parameters": sum(p.numel() for p in model.parameters()),
                "training_device": device,
                "cpu_threads": selected["cpu_threads"],
                "initial_weights_sha256": initial_sha,
                "selected_weights_arrays": weights["arrays"],
            }
            histories[key] = history
            details[key] = {
                "initial_test": initial_detail,
                "selected_test": detail,
                "selected_valid": valid_detail,
            }
            evaluation_arrays[key + "__initial_test_nll"] = initial_losses
            evaluation_arrays[key + "__test_nll"] = losses
            evaluation_arrays[key + "__valid_nll"] = valid_losses
            generations[key] = []
            for i, prompt in enumerate(prompts):
                prefix = [EOS] + tokenizer.encode(prompt)
                ids, metadata = generate(
                    restored,
                    prefix,
                    max_new_tokens=config.generation_tokens,
                    temperature=0.8,
                    top_k=40,
                    seed=seed + i,
                    use_cache=True,
                    eos_id=EOS,
                )
                generations[key].append(
                    {
                        "prompt": prompt,
                        "prompt_ids": prefix,
                        "token_ids": ids,
                        "continuation": tokenizer.decode(ids[len(prefix) :], errors="replace"),
                        "temperature": 0.8,
                        "top_k": 40,
                        "sampling_seed": seed + i,
                        **metadata,
                    }
                )
            # Time batch-one decoding on CPU, separately from accelerator training.
            restored = restored.to("cpu").eval()
            torch.set_num_threads(1)
            prefix = ([EOS] + tokenizer.encode(prompts[0]))[-max(1, config.context_length // 3) :]
            overflow = (tokenizer.encode("The little girl looked around. " * 50))[
                -max(1, config.context_length - 1) :
            ]
            benchmarks[key] = {
                "device": "cpu",
                "cpu_threads": 1,
                "within_window": benchmark_decode(
                    restored,
                    prefix,
                    new_tokens=min(32, config.generation_tokens),
                    repeats=config.benchmark_repeats,
                ),
                "window_rollover": benchmark_decode(
                    restored,
                    overflow,
                    new_tokens=min(16, config.generation_tokens),
                    repeats=config.benchmark_repeats,
                ),
            }
            torch.set_num_threads(selected["cpu_threads"])
        np.savez_compressed(run.path / "evaluation.npz", **evaluation_arrays)
        run.record(
            "evaluation.json",
            {
                "details": details,
                "arrays": {k: array_digest(v) for k, v in evaluation_arrays.items()},
                "protocol": (
                    "each within-chapter target counted once; nonoverlap context blocks; "
                    "EOS included; padding ignored"
                ),
            },
        )
        run.record("baselines.json", baselines)
        run.record("metrics.json", results)
        run.record("histories.json", histories)
        run.record("generations.json", generations)
        run.record("benchmarks.json", benchmarks)
        run.record("checks.json", checks)
        run.record(
            "cost.json",
            {
                "wall_seconds": time.perf_counter() - start,
                "training_device": device,
                "cpu_threads": selected["cpu_threads"],
                "model_config": asdict(model_config),
                "dtype": "float32",
            },
        )
        return run.path, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/p3_pretrain.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/p3-pretrain"))
    args = parser.parse_args()
    try:
        config = Config(**json.loads(args.config.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    path, results = execute(config, args.output)
    print(
        json.dumps(
            {
                "run": str(path.resolve()),
                "results": {
                    k: {
                        "selected_step": v["training"]["selected_step"],
                        "initial_nll": v["initial_test"]["nll"],
                        "test_nll": v["selected_test"]["nll"],
                        "test_perplexity": v["selected_test"]["perplexity"],
                        "training_device": v["training_device"],
                    }
                    for k, v in results.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
