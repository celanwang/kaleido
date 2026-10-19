"""Retain replay receipts and explicit metric/array comparisons against reviewed P0-P5 evidence."""

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np

from kaleido.evaluation.schema import digest
from kaleido.experiments import array_digest, write_json
from kaleido.repro.integrity import file_info

ROOT = Path(__file__).resolve().parents[1]


def differences(a, b, path="", *, atol=1e-5, rtol=1e-5):
    if isinstance(a, dict) and isinstance(b, dict):
        keys = (set(a) | set(b)) - {"wall_seconds"}
        return [
            item
            for k in sorted(keys)
            for item in (
                differences(a[k], b[k], f"{path}/{k}", atol=atol, rtol=rtol)
                if k in a and k in b
                else [f"{path}/{k}: missing key"]
            )
        ]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length differs"]
        return [
            item
            for i, (x, y) in enumerate(zip(a, b, strict=True))
            for item in differences(x, y, f"{path}/{i}", atol=atol, rtol=rtol)
        ]
    if type(a) is bool or type(b) is bool:
        return [] if type(a) is type(b) and a == b else [f"{path}: boolean type/value differs"]
    if type(a) is float or type(b) is float:
        if type(a) in (int, float) and type(b) in (int, float):
            return [] if math.isclose(a, b, abs_tol=atol, rel_tol=rtol) else [f"{path}: {a} != {b}"]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def summarize(suite, output):
    suite, output = Path(suite), Path(output)
    manifest = json.loads((suite / "manifest.json").read_text())
    steps = json.loads((suite / "steps.json").read_text())
    if manifest["status"] != "completed" or manifest["config"]["profile"] != "formal":
        raise ValueError("only completed formal reproduction is a release evidence candidate")
    stages = {}
    for receipt in steps["completed"]:
        stage = receipt["stage"]
        run = Path(receipt["run"])
        reference = ROOT / f"results/{stage}/reference"
        own_manifest = json.loads((run / "manifest.json").read_text())
        if own_manifest["status"] != "completed" or own_manifest["environment"]["git_dirty"]:
            raise ValueError(f"{stage} is incomplete or was run from a dirty checkout")
        comparisons = {}
        arrays = {}
        retained = {}
        for name in (
            "metrics.json",
            "summary.json",
            "checks.json",
            "baselines.json",
            "comparisons.json",
        ):
            if (run / name).exists() and (reference / name).exists():
                new, old = [json.loads((d / name).read_text()) for d in (run, reference)]
                comparisons[name] = {"differences": differences(old, new), "actual": new}
        for path in sorted(run.glob("*.npz")):
            if not (reference / path.name).exists():
                continue
            with (
                np.load(path, allow_pickle=False) as new,
                np.load(reference / path.name, allow_pickle=False) as old,
            ):
                hashes = {k: array_digest(new[k]) for k in new}
                old_hashes = {k: array_digest(old[k]) for k in old}
                arrays[path.name] = {
                    "arrays": len(hashes),
                    "identical": hashes == old_hashes,
                    "new_digests": hashes,
                    "reference_digests": old_hashes,
                }
                if hashes != old_hashes:
                    destination = output.parent / "changed" / stage
                    destination.mkdir(parents=True, exist_ok=True)
                    for source in (path, path.with_suffix(".json")):
                        if source.exists():
                            shutil.copyfile(source, destination / source.name)
                    retained[path.name] = str((destination / path.name).relative_to(output.parent))
        traces = {}
        for name in ("test_samples.json", "evaluations.json"):
            if (run / name).exists() and (reference / name).exists():
                new, old = [json.loads((d / name).read_text()) for d in (run, reference)]
                if name == "evaluations.json":
                    new = {k: v["records"] for k, v in new.items()}
                    old = {k: v["records"] for k, v in old.items()}
                traces[name] = {
                    "identical": new == old,
                    "actual_sha256": digest(new),
                    "reference_sha256": digest(old),
                }
                if new != old:
                    destination = output.parent / "changed" / stage
                    destination.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(run / name, destination / name)
                    retained[name] = str((destination / name).relative_to(output.parent))
        stages[stage] = {
            "receipt": receipt,
            "manifest": own_manifest,
            "cost": json.loads((run / "cost.json").read_text()),
            "json_comparisons": comparisons,
            "array_comparisons": arrays,
            "trace_comparisons": traces,
            "retained_changes": retained,
            "files": {
                str(p.relative_to(run)): file_info(p) for p in sorted(run.rglob("*")) if p.is_file()
            },
        }
    return {
        "schema_version": 1,
        "status": "completed",
        "suite_manifest": manifest,
        "stages": stages,
        "all_compared_metrics_match": all(
            not v["differences"] for s in stages.values() for v in s["json_comparisons"].values()
        ),
        "all_compared_arrays_identical": all(
            v["identical"] for s in stages.values() for v in s["array_comparisons"].values()
        ),
        "all_compared_traces_identical": all(
            v["identical"] for s in stages.values() for v in s["trace_comparisons"].values()
        ),
        "metric_tolerance": {"atol": 1e-5, "rtol": 1e-5, "ignored_fields": ["wall_seconds"]},
        "scope": (
            "one formal replay on this recorded host; new P5 uses published checkpoints; "
            "not cross-device determinism"
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = summarize(args.suite, args.output)
    write_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k.startswith("all_") or k == "status"}))
