"""P4 formal pinned-Qwen SFT/GRPO/rejection experiment; explicit fixture-only offline checks."""

import argparse
import gzip
import hashlib
import json
import shutil
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from kaleido.experiments import Run, write_json
from kaleido.posttrain.fixture import load_fixture
from kaleido.posttrain.model import (
    adapter_state,
    inject_lora,
    load_adapter,
    load_pinned,
    prompt_tokens,
    save_adapter,
)
from kaleido.posttrain.recipes import (
    evaluate_policy,
    probe,
    train_grpo,
    train_rejection_sft,
    train_sft,
)
from kaleido.posttrain.tasks import dataset_digest, make_tasks, score_response, summarize

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Config:
    seeds: tuple = (42, 7, 2026)
    device: str = "mps"
    rank: int = 8
    alpha: float = 16.0
    sft_steps: int = 64
    sft_batch_size: int = 4
    sft_lr: float = 0.0002
    sft_eval_interval: int = 16
    iterations: int = 24
    prompts_per_iteration: int = 2
    group_size: int = 4
    grpo_epochs: int = 2
    post_lr: float = 0.00005
    post_eval_interval: int = 4
    beta: float = 0.04
    clip: float = 0.2
    max_new_tokens: int = 32
    task_seed: int = 731
    train_per_family: int = 32
    heldout_per_family: int = 16
    operand_max: int = 30

    def __post_init__(self):
        if (
            not isinstance(self.seeds, (list, tuple))
            or not self.seeds
            or len(set(self.seeds)) != len(self.seeds)
            or any(type(s) is not int or not 0 <= s < 2**32 - 1000 for s in self.seeds)
        ):
            raise ValueError("seeds must be unique bounded non-negative integers")
        for name in (
            "rank",
            "sft_steps",
            "sft_batch_size",
            "sft_eval_interval",
            "iterations",
            "prompts_per_iteration",
            "group_size",
            "grpo_epochs",
            "post_eval_interval",
            "max_new_tokens",
            "train_per_family",
            "heldout_per_family",
            "operand_max",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"invalid {name}")
        for name in ("alpha", "sft_lr", "post_lr"):
            if (
                isinstance(getattr(self, name), bool)
                or not np.isfinite(getattr(self, name))
                or getattr(self, name) <= 0
            ):
                raise ValueError(f"invalid {name}")
        if (
            self.device not in ("cpu", "mps")
            or self.group_size < 2
            or not np.isfinite(self.beta)
            or self.beta < 0
            or not 0 < self.clip < 1
            or type(self.task_seed) is not int
            or self.task_seed < 0
            or self.prompts_per_iteration > 4 * self.train_per_family
        ):
            raise ValueError("invalid device, GRPO coefficient or task schedule")


def _without_time(value):
    if isinstance(value, dict):
        return {k: _without_time(v) for k, v in value.items() if k != "wall_seconds"}
    if isinstance(value, list):
        return [_without_time(v) for v in value]
    return value


def _save_native_tokenizer(run, tokenizer):
    directory = run.path / "tokenizer"
    directory.mkdir()
    tokenizer.save_pretrained(directory)
    files = {}
    for path in list(directory.iterdir()):
        if not path.is_file():
            continue
        raw = path.read_bytes()
        files[path.name] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        with gzip.GzipFile(filename=str(path) + ".gz", mode="wb", mtime=0) as stream:
            stream.write(raw)
        path.unlink()
    write_json(directory / "files.json", files)


def execute(
    config: Config,
    output: Path,
    *,
    root=ROOT,
    spec_path=ROOT / "configs/p4_model.json",
    fixture=False,
    emit=None,
):
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1 if fixture else 4)
    try:
        return _execute(config, output, root=root, spec_path=spec_path, fixture=fixture, emit=emit)
    finally:
        torch.set_num_threads(original_threads)


def _execute(config, output, *, root, spec_path, fixture, emit):
    kind = "fixture" if fixture else "formal_qwen"
    with Run(output, {**asdict(config), "experiment_kind": kind}, root=root) as run:
        start = time.perf_counter()
        tasks = make_tasks(
            seed=config.task_seed,
            train_per_family=config.train_per_family,
            heldout_per_family=config.heldout_per_family,
            operand_max=config.operand_max,
        )
        splits = {
            split: [r for r in tasks if r["split"] == split] for split in ("train", "valid", "test")
        }
        run.record(
            "tasks.json",
            {
                "dataset_sha256": dataset_digest(tasks),
                "rows": tasks,
                "answer_counts": {
                    s: dict(Counter(r["answer"] for r in rows)) for s, rows in splits.items()
                },
            },
        )
        baselines = {}
        majority = Counter(r["answer"] for r in splits["train"]).most_common(1)[0][0]
        for split in ("valid", "test"):
            baselines[split] = summarize(
                [
                    {
                        "id": r["id"],
                        "family": r["family"],
                        **score_response(r, f"<answer>{majority}</answer>"),
                    }
                    for r in splits[split]
                ]
            )
        all_metrics, all_histories, all_rollouts, validation_samples, test_samples, checks = (
            {},
            {},
            {},
            {},
            {},
            {},
        )
        common_base = None
        provenance = None
        for seed in config.seeds:
            key = str(seed)
            if emit:
                emit({"stage": "load", "seed": seed, "experiment_kind": kind})
            model, tokenizer, identity = (
                load_fixture() if fixture else load_pinned(spec_path, device=config.device)
            )
            targets = inject_lora(model, rank=config.rank, alpha=config.alpha, seed=seed)
            if provenance is None:
                provenance = identity
                run.record("model_source.json", identity)
                if not fixture:
                    _save_native_tokenizer(run, tokenizer)
                    from huggingface_hub import hf_hub_download

                    license_path = hf_hub_download(
                        identity["repo_id"],
                        "LICENSE",
                        revision=identity["revision"],
                        local_files_only=True,
                    )
                    shutil.copyfile(license_path, run.path / "QWEN-LICENSE.txt")
                run.record("prompts.json", {r["id"]: prompt_tokens(tokenizer, r) for r in tasks})
                resources = probe(
                    model,
                    tokenizer,
                    splits["train"],
                    batch_size=config.sft_batch_size,
                    lr=config.sft_lr,
                )
                run.record("resources.json", resources)
                if emit:
                    emit({"stage": "probe", "resources": resources})
                common_base = {}
                for split in ("valid", "test"):
                    metrics, records = evaluate_policy(
                        model, tokenizer, splits[split], max_new_tokens=config.max_new_tokens
                    )
                    common_base[split] = {"metrics": metrics, "samples": records}
                run.record("base_evaluation.json", common_base)

            def progress(event, seed=seed):
                if emit:
                    emit({"seed": seed, **event})

            adapter_meta = {
                "model_source": identity,
                "rank": config.rank,
                "alpha": config.alpha,
                "seed": seed,
                "targets": targets,
                "experiment_kind": kind,
            }
            all_metrics[key] = {}
            all_histories[key] = {}
            all_rollouts[key] = {}
            validation_samples[key] = {}
            test_samples[key] = {}
            sft, curve, validation = train_sft(
                model,
                tokenizer,
                splits["train"],
                splits["valid"],
                steps=config.sft_steps,
                batch_size=config.sft_batch_size,
                lr=config.sft_lr,
                eval_interval=config.sft_eval_interval,
                max_new_tokens=config.max_new_tokens,
                seed=seed,
                initial_validation=common_base["valid"]["metrics"],
                emit=progress,
            )
            save_adapter(
                run.path / f"sft-{seed}.npz",
                model,
                metadata={**adapter_meta, "method": "sft", "selection": sft["selected_step"]},
            )
            reference = adapter_state(model)
            ref_digest = {
                k: hashlib.sha256(v.numpy().tobytes()).hexdigest() for k, v in reference.items()
            }
            test, records = evaluate_policy(
                model, tokenizer, splits["test"], max_new_tokens=config.max_new_tokens
            )
            all_metrics[key]["sft"] = {"training": sft, "test": test}
            all_histories[key]["sft"] = curve
            validation_samples[key]["sft"] = validation
            test_samples[key]["sft"] = records
            if emit:
                emit({"seed": seed, "stage": "sft_complete", "selected_step": sft["selected_step"]})
            grpo, curve, rollouts, validation = train_grpo(
                model,
                tokenizer,
                splits["train"],
                splits["valid"],
                reference,
                iterations=config.iterations,
                prompts_per_iteration=config.prompts_per_iteration,
                group_size=config.group_size,
                epochs=config.grpo_epochs,
                lr=config.post_lr,
                eval_interval=config.post_eval_interval,
                max_new_tokens=config.max_new_tokens,
                beta=config.beta,
                clip=config.clip,
                seed=seed,
                initial_validation=sft["selected_validation"],
                emit=progress,
            )
            save_adapter(
                run.path / f"grpo-{seed}.npz",
                model,
                metadata={
                    **adapter_meta,
                    "method": "grpo",
                    "reference_sft_arrays": ref_digest,
                    "selection": grpo["selected_iteration"],
                },
            )
            test, records = evaluate_policy(
                model, tokenizer, splits["test"], max_new_tokens=config.max_new_tokens
            )
            all_metrics[key]["grpo"] = {"training": grpo, "test": test}
            all_histories[key]["grpo"] = curve
            all_rollouts[key]["grpo"] = rollouts
            validation_samples[key]["grpo"] = validation
            test_samples[key]["grpo"] = records
            load_adapter(model, reference)
            rft, curve, rollouts, validation = train_rejection_sft(
                model,
                tokenizer,
                splits["train"],
                splits["valid"],
                iterations=config.iterations,
                prompts_per_iteration=config.prompts_per_iteration,
                group_size=config.group_size,
                updates=config.iterations * config.grpo_epochs,
                batch_size=config.prompts_per_iteration * config.group_size,
                lr=config.post_lr,
                eval_interval=config.post_eval_interval * config.grpo_epochs,
                max_new_tokens=config.max_new_tokens,
                seed=seed,
                initial_validation=sft["selected_validation"],
                emit=progress,
            )
            save_adapter(
                run.path / f"rejection-{seed}.npz",
                model,
                metadata={
                    **adapter_meta,
                    "method": "rejection_sft",
                    "reference_sft_arrays": ref_digest,
                    "selection": rft["selected_step"],
                },
            )
            test, records = evaluate_policy(
                model, tokenizer, splits["test"], max_new_tokens=config.max_new_tokens
            )
            all_metrics[key]["rejection_sft"] = {"training": rft, "test": test}
            all_histories[key]["rejection_sft"] = curve
            all_rollouts[key]["rejection_sft"] = rollouts
            validation_samples[key]["rejection_sft"] = validation
            test_samples[key]["rejection_sft"] = records
            grpo_schedule = [r["question_ids"] for r in all_rollouts[key]["grpo"]]
            rejection_schedule = [r["question_ids"] for r in all_rollouts[key]["rejection_sft"]]
            checks[key] = {
                "base_weights_frozen": all(
                    p.grad is None for p in model.parameters() if not p.requires_grad
                ),
                "shared_prompt_schedule": grpo_schedule == rejection_schedule,
                "reference_sft_arrays": ref_digest,
            }
            if not checks[key]["base_weights_frozen"] or not checks[key]["shared_prompt_schedule"]:
                raise AssertionError("frozen-base or matched-schedule invariant failed")
            # Persist each completed seed so failures do not erase earlier evidence.
            for name, value in (
                ("metrics", all_metrics),
                ("histories", all_histories),
                ("rollouts", all_rollouts),
                ("validation_samples", validation_samples),
                ("test_samples", test_samples),
                ("checks", checks),
            ):
                run.record(name + ".json", value)
            if emit:
                emit({"seed": seed, "stage": "seed_complete"})
            del model
            if config.device == "mps" and not fixture:
                torch.mps.empty_cache()
        run.record(
            "baselines.json", {"constant_training_majority": majority, "evaluation": baselines}
        )
        run.record(
            "protocol.json",
            {
                "kind": kind,
                "reward": "strict full response + AST label, nontruncated",
                "evaluation": "independent regex parser + expanded polynomial oracle",
                "selection": "validation strict accuracy, tie by completion NLL; test report only",
                "grpo": {
                    "loss_type": "per-completion mean GRPO",
                    "std": "population +1e-4",
                    "temperature": 1.0,
                    "top_k": None,
                    "top_p": None,
                    "repetition_penalty": 1.0,
                    "reference": "selected SFT checkpoint",
                    "clip": config.clip,
                    "beta": config.beta,
                    "old_policy": "fixed within each group update",
                },
                "rejection": (
                    "same scheduled prompts/groups; fixed SFT sampling; "
                    "unique accepted generated IDs; fresh optimizer"
                ),
                "budget_limits": (
                    "same generation ceilings and planned updates; "
                    "actual generated/update tokens and time differ"
                ),
                "optimizer": {
                    "name": "AdamW",
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 0.0,
                    "gradient_clip_norm": 1.0,
                },
                "base_weights_in_repository": False,
            },
        )
        run.record(
            "cost.json",
            {
                "wall_seconds": time.perf_counter() - start,
                "device": "cpu" if fixture else config.device,
            },
        )
        return run.path, _without_time(all_metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/p4_posttrain.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/p4-posttrain"))
    parser.add_argument(
        "--fixture", action="store_true", help="random tiny offline fixture, not formal evidence"
    )
    args = parser.parse_args()
    try:
        config = Config(**json.loads(args.config.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    path, _metrics = execute(
        config,
        args.output,
        fixture=args.fixture,
        emit=lambda value: print(json.dumps(value), flush=True),
    )
    print(
        json.dumps({"run": str(path.resolve()), "status": "completed", "fixture": args.fixture}),
        flush=True,
    )


if __name__ == "__main__":
    main()
