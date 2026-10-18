"""P5: frozen benchmarks, P3/P4 adapters, paired inference and calibrated local Judge."""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from kaleido.evaluation.adapters import ArchiveAdapter, DecoderLikelihoodAdapter, QwenAdapter
from kaleido.evaluation.benchmarks import build_suite
from kaleido.evaluation.comparison import compare
from kaleido.evaluation.judge import PROTOCOL as JUDGE_PROTOCOL
from kaleido.evaluation.judge import calibrate, judge_cases, summarize_judgments
from kaleido.evaluation.runner import evaluate
from kaleido.evaluation.schema import Case, GenerationProtocol, digest
from kaleido.evaluation.statistics import document_nll_bootstrap
from kaleido.experiments import Run
from kaleido.lm.corpus import prepare_corpus
from kaleido.posttrain.fixture import load_fixture
from kaleido.posttrain.model import (
    adapter_state,
    file_sha256,
    inject_lora,
    load_pinned,
    read_adapter,
)
from kaleido.posttrain.tasks import SYSTEM, independent_oracle

ROOT = Path(__file__).resolve().parents[3]
METHODS = ("sft", "grpo", "rejection_sft")


@dataclass(frozen=True)
class Config:
    seeds: tuple = (42, 7, 2026)
    device: str = "mps"
    math_per_family: int = 16
    gsm_count: int = 32
    qa_count: int = 12
    max_new_tokens: int = 64
    batch_size: int = 8
    judge_pairs: int = 16
    resamples: int = 4000

    def __post_init__(self):
        if (
            not isinstance(self.seeds, (tuple, list))
            or not self.seeds
            or len(set(self.seeds)) != len(self.seeds)
            or any(type(s) is not int or s not in (42, 7, 2026) for s in self.seeds)
        ):
            raise ValueError("choose unique published P3/P4 seeds")
        if self.device not in ("cpu", "mps"):
            raise ValueError("device must be cpu or mps")
        for name in (
            "math_per_family",
            "gsm_count",
            "qa_count",
            "max_new_tokens",
            "batch_size",
            "judge_pairs",
            "resamples",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"invalid {name}")
        if self.math_per_family % 2 or self.judge_pairs % 4 or self.resamples < 100:
            raise ValueError("invalid family/pair/resample count")


def archived_panel(root, seeds):
    directory = root / "results/p4/reference"
    tasks = json.loads((directory / "tasks.json").read_text())["rows"]
    if any(independent_oracle(r) != r["answer"] for r in tasks):
        raise AssertionError("P4 gold no longer matches independent arithmetic oracle")
    cases = [
        Case(r["id"], "p4_original_test", r["family"], r["prompt"], str(r["answer"]), system=SYSTEM)
        for r in tasks
        if r["split"] == "test"
    ]
    samples = json.loads((directory / "test_samples.json").read_text())
    protocol = GenerationProtocol(max_new_tokens=32, seed=9000, batch_size=8)
    results = {}
    for seed in seeds:
        for method in METHODS:
            name = f"{method}-{seed}"
            raw = samples[str(seed)][method]
            adapter = ArchiveAdapter(
                name,
                raw,
                protocol=protocol,
                source={
                    "test_samples_sha256": file_sha256(directory / "test_samples.json"),
                    "manifest_sha256": file_sha256(directory / "manifest.json"),
                },
            )
            value = evaluate(adapter, cases, protocol)
            original = {r["id"]: r for r in raw}
            if any(r["correct"] != original[r["id"]]["strict_correct"] for r in value["records"]):
                raise AssertionError("unified scoring disagrees with P4")
            results[name] = value
    return {"cases": [c.to_dict() for c in cases], "results": results}


def judge_panel(adapter, cases, results, seeds):
    pairs = []
    for seed in seeds:
        for method in ("grpo", "rejection_sft"):
            left = {r["id"]: r for r in results[f"sft-{seed}"]["records"]}
            right = {r["id"]: r for r in results[f"{method}-{seed}"]["records"]}
            for c in cases:
                if c.scorer != "context_color":
                    continue
                a, b = left[c.id], right[c.id]
                pairs.append(
                    {
                        "id": f"panel-{seed}-{method}-{c.id}",
                        "split": "test",
                        "category": "model_color_qa",
                        "question": c.prompt,
                        "reference": c.answer,
                        "left": a["response"],
                        "right": b["response"],
                        "gold": "tie"
                        if a["correct"] == b["correct"]
                        else "left"
                        if a["correct"]
                        else "right",
                    }
                )
    return {
        "pairs": pairs,
        **summarize_judgments(pairs, adapter.generate(judge_cases(pairs), JUDGE_PROTOCOL)),
    }


def execute(config, output, *, root=ROOT, fixture=False, emit=None):
    previous = torch.get_num_threads()
    torch.set_num_threads(1 if fixture else 4)
    try:
        return _execute(config, output, root=Path(root), fixture=fixture, emit=emit)
    finally:
        torch.set_num_threads(previous)


def _execute(config, output, *, root, fixture, emit):
    start = time.perf_counter()
    with Run(output, {**asdict(config), "fixture": fixture}, root=root) as run:
        cases, manifest = build_suite(
            root,
            math_per_family=config.math_per_family,
            gsm_count=config.gsm_count,
            qa_count=config.qa_count,
        )
        protocol = GenerationProtocol(
            max_new_tokens=config.max_new_tokens, batch_size=config.batch_size
        )
        run.record("benchmarks.json", {**manifest, "cases": [c.to_dict() for c in cases]})
        run.record(
            "protocol.json",
            {
                "generation": protocol.to_dict(),
                "judge": JUDGE_PROTOCOL.to_dict(),
                "checkpoint_selection": "unchanged published P4 validation-selected checkpoints",
                "training_updates": 0,
                "suite_macro_average_is_not_a_quality_ranking": True,
                "comparisons": (
                    "SFT vs GRPO and rejection_sft in each suite and fixed seed; Holm per panel"
                ),
                "bootstrap": (
                    "question within family; all fixed seeds retained, not independent observations"
                ),
            },
        )
        archive = archived_panel(root, config.seeds)
        run.record("p4_archive.json", archive)
        run.record(
            "p4_comparisons.json",
            compare(archive["results"], config.seeds, resamples=config.resamples),
        )
        # Independently re-evaluate actual P3 checkpoints on their original chapter split.
        documents, audit = prepare_corpus(root / "data/p3")
        p3, losses = {}, {}
        for seed in config.seeds:
            adapter = DecoderLikelihoodAdapter(
                f"decoder-{seed}",
                root / f"results/p3/reference/model-{seed}.npz",
                root / "results/p3/reference/tokenizer.json",
            )
            value = adapter.evaluate(documents)
            losses[str(seed)] = value.pop("token_losses")
            value["uncertainty"] = document_nll_bootstrap(
                value["documents"], resamples=config.resamples
            )
            p3[str(seed)] = value
            if emit:
                emit({"stage": "p3", "seed": seed, "nll": value["metrics"]["nll"]})
        np.savez_compressed(run.path / "p3_token_losses.npz", **losses)
        run.record(
            "p3_likelihood.json",
            {"models": p3, "corpus_audit": audit, "document_sha256": digest(documents)},
        )
        model, tokenizer, identity = (
            load_fixture()
            if fixture
            else load_pinned(root / "configs/p4_model.json", device=config.device)
        )
        if not fixture:
            expected = json.loads((root / "results/p4/reference/model_source.json").read_text())
            keys = ("repo_id", "revision", "weights_sha256", "chat_template_sha256")
            if any(identity[k] != expected[k] for k in keys):
                raise ValueError("native base/template differs from frozen P4 source")
        inject_lora(model, rank=8, alpha=16, seed=42)
        initial = adapter_state(model)
        base = QwenAdapter("base", model, tokenizer, initial, source=identity)
        run.record("model_source.json", identity)
        results = {"base": evaluate(base, cases, protocol)}
        if emit:
            emit(
                {"stage": "model", "model": "base", "summary": results["base"]["summary"]["suite"]}
            )
        for seed in config.seeds:
            for method in METHODS:
                name = f"{method}-{seed}"
                file_method = "rejection" if method == "rejection_sft" else method
                path = root / f"results/p4/reference/{file_method}-{seed}.npz"
                state, info = (initial, {"fixture": True}) if fixture else read_adapter(path)
                if not fixture and (
                    info["rank"] != 8
                    or info["alpha"] != 16
                    or info["seed"] != seed
                    or info["method"] != method
                    or any(info["model_source"][k] != identity[k] for k in keys)
                ):
                    raise ValueError("published checkpoint identity differs from requested model")
                source = (
                    {"fixture": True}
                    if fixture
                    else {
                        "npz_sha256": file_sha256(path),
                        "metadata_sha256": file_sha256(path.with_suffix(".json")),
                        "checkpoint": str(path.relative_to(root)),
                        "metadata": info,
                        "base": identity,
                    }
                )
                adapter = QwenAdapter(name, model, tokenizer, state, source=source)
                results[name] = evaluate(adapter, cases, protocol)
                run.record("evaluations.json", results)
                if emit:
                    emit(
                        {
                            "stage": "model",
                            "model": name,
                            "summary": results[name]["summary"]["suite"],
                        }
                    )
        run.record("comparisons.json", compare(results, config.seeds, resamples=config.resamples))
        judge = calibrate(base, count=config.judge_pairs, emit=emit)
        run.record("judge_calibration.json", judge)
        panel = (
            judge_panel(base, cases, results, config.seeds)
            if judge["enabled"]
            else {"status": "disabled", "reason": "predeclared calibration/audit gate failed"}
        )
        run.record("judge_panel.json", panel)
        run.record(
            "checks.json",
            {
                "fixture": fixture,
                "no_training": True,
                "p4_scoring_reproduced": True,
                "p3_original_split": audit["splits"]["test"]["documents"],
                "judge_enabled": judge["enabled"],
                "checkpoint_reloaded_before_every_generation": True,
            },
        )
        run.record(
            "cost.json",
            {
                "wall_seconds": time.perf_counter() - start,
                "device": "cpu" if fixture else config.device,
                "live_generated_tokens": sum(
                    r.get("generated_tokens", 0) for v in results.values() for r in v["records"]
                ),
                "live_prompt_tokens": sum(
                    r.get("prompt_tokens", 0) for v in results.values() for r in v["records"]
                ),
                "judge_generated_tokens": sum(
                    r.get("generated_tokens", 0)
                    for split in ("calibration", "audit")
                    for r in judge[split]["records"]
                )
                + sum(r.get("generated_tokens", 0) for r in panel.get("records", [])),
            },
        )
        return run.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/p5_evaluation.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/p5-evaluation"))
    parser.add_argument(
        "--fixture", action="store_true", help="random model wiring only; not evidence"
    )
    args = parser.parse_args()
    try:
        config = Config(**json.loads(args.config.read_text()))
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    path = execute(
        config,
        args.output,
        fixture=args.fixture,
        emit=lambda value: print(json.dumps(value), flush=True),
    )
    print(json.dumps({"status": "completed", "run": str(path.resolve()), "fixture": args.fixture}))


if __name__ == "__main__":
    main()
