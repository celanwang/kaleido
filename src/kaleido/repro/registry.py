"""Explicit recipe dispatch; separate smoke wiring from formal experimental evidence."""

import importlib
import json
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from subprocess import run as run_process

from kaleido.experiments import Run
from kaleido.repro.integrity import file_info


@dataclass(frozen=True)
class Recipe:
    module: str
    formal: str
    smoke: str
    requires: tuple
    external_model: bool = False


RECIPES = {
    "p0": Recipe("linear_regression", "p0_linear.json", "p0_linear.json", ()),
    "p1": Recipe("classical_ml", "p1_classical.json", "p1_classical.json", ("scikit-learn",)),
    "p2": Recipe("deep_learning", "p2_deep.json", "p2_deep.json", ("scikit-learn", "torch")),
    "p3": Recipe("language_model", "p3_pretrain.json", "p3_smoke.json", ("torch",)),
    "p4": Recipe(
        "post_training",
        "p4_posttrain.json",
        "p4_smoke.json",
        ("torch", "transformers", "huggingface-hub"),
        True,
    ),
    "p5": Recipe(
        "model_evaluation",
        "p5_evaluation.json",
        "p5_smoke.json",
        ("torch", "transformers", "huggingface-hub"),
        True,
    ),
}


def repository(root):
    root = Path(root).resolve()
    if not all(
        (root / p).is_file() for p in ("pyproject.toml", "uv.lock", "configs/p0_linear.json")
    ):
        raise ValueError("repository root required; pass --root PATH to the source checkout")
    return root


def doctor(root):
    root = repository(root)
    packages = {}
    import tomllib

    project = tomllib.loads((root / "pyproject.toml").read_text())
    pins = {s.split("==")[0]: s.split("==")[1] for s in project["project"]["dependencies"]}
    for dependencies in project["project"]["optional-dependencies"].values():
        pins.update({s.split("==")[0]: s.split("==")[1] for s in dependencies})
    for package, expected in pins.items():
        try:
            actual = version(package)
        except PackageNotFoundError:
            actual = None
        packages[package] = {
            "expected": expected,
            "installed": actual,
            "matches": actual == expected,
        }
    python_ok = sys.version_info[:2] == (3, 12)
    return {
        "ok": python_ok and all(v["matches"] for v in packages.values()),
        "python": sys.version,
        "python_matches": python_ok,
        "packages": packages,
        "scope": "full-reproduction dependency pins; no model download or accelerator probe",
    }


def run_recipe(stage, root, output, *, profile="formal", config_path=None, emit=None):
    root = repository(root)
    if stage not in RECIPES or profile not in ("formal", "smoke"):
        raise ValueError("unknown recipe/profile")
    recipe = RECIPES[stage]
    missing = []
    for package in recipe.requires:
        try:
            version(package)
        except PackageNotFoundError:
            missing.append(package)
    if missing:
        raise ValueError(
            f"missing dependencies: {', '.join(missing)}; uv sync --locked --all-extras"
        )
    selected = (
        Path(config_path)
        if config_path is not None
        else root / "configs" / getattr(recipe, profile)
    )
    selected = selected.resolve()
    module = importlib.import_module("kaleido.demos." + recipe.module)
    config = module.Config(**json.loads(selected.read_text()))
    kwargs = {"root": root}
    if stage == "p3":
        kwargs["corpus_dir"] = root / "data/p3"
    if stage in ("p4", "p5"):
        kwargs.update(fixture=profile == "smoke", emit=emit)
    if stage == "p4":
        kwargs["spec_path"] = root / "configs/p4_model.json"
    result = module.execute(config, Path(output), **kwargs)
    path = result[0] if isinstance(result, tuple) else result
    manifest = json.loads((path / "manifest.json").read_text())
    return {
        "stage": stage,
        "profile": profile,
        "run": str(path.resolve()),
        "status": manifest["status"],
        "effective_config": asdict(config),
        "config_file": str(selected),
        "config_identity": file_info(selected),
        "fixture": profile == "smoke" and stage in ("p4", "p5"),
        "note": "P5 re-evaluates published P3/P4 artifacts, not newly produced sibling runs"
        if stage == "p5"
        else None,
    }


def reproduce(root, output, *, stages=tuple(RECIPES), profile="smoke", emit=None):
    root = repository(root)
    stages = list(stages)
    if not stages or len(set(stages)) != len(stages) or any(s not in RECIPES for s in stages):
        raise ValueError("choose unique non-empty known stages")
    if profile not in ("smoke", "formal"):
        raise ValueError("unknown reproduction profile")
    with Run(Path(output), {"stages": stages, "profile": profile}, root=root) as suite:
        records = []
        for stage in stages:
            if emit:
                emit({"stage": stage, "status": "started", "profile": profile})
            # Separate processes isolate Torch thread/RNG/global state between old recipes.
            command = [
                sys.executable,
                "-m",
                "kaleido",
                "--root",
                str(root),
                "run",
                stage,
                "--profile",
                profile,
                "--output",
                str(suite.path.resolve() / stage),
            ]
            process = run_process(command, cwd=root, text=True, capture_output=True)
            (suite.path / f"{stage}.stdout.log").write_text(process.stdout)
            (suite.path / f"{stage}.stderr.log").write_text(process.stderr)
            if process.returncode != 0:
                suite.record(
                    "steps.json",
                    {"completed": records, "failed_stage": stage, "exit_code": process.returncode},
                )
                raise RuntimeError(f"{stage} failed; evidence retained at {suite.path}")
            try:
                value = json.loads(process.stdout.splitlines()[-1])
                if value.get("status") != "completed" or not Path(value["run"]).is_dir():
                    raise ValueError("child did not report a completed run")
            except (ValueError, KeyError, IndexError, TypeError) as error:
                suite.record(
                    "steps.json",
                    {
                        "completed": records,
                        "failed_stage": stage,
                        "exit_code": 0,
                        "error": "invalid_child_result",
                    },
                )
                raise RuntimeError(f"{stage} returned invalid evidence at {suite.path}") from error
            records.append(value)
            suite.record("steps.json", {"completed": records, "failed_stage": None})
            if emit:
                emit({"stage": stage, "status": "completed", "run": value["run"]})
        return suite.path
