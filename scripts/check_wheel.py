"""Install the built wheel and locked CPU runtime into an independent, offline environment."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from kaleido.experiments import write_json
from kaleido.repro.integrity import file_info

ROOT = Path(__file__).resolve().parents[1]


def check(wheel, bundle):
    wheel, bundle = Path(wheel).resolve(), Path(bundle).resolve()
    examples = json.loads((bundle / "examples.json").read_text())["cases"]
    with tempfile.TemporaryDirectory(prefix="kaleido-wheel-") as directory:
        directory = Path(directory)
        environment = directory / "venv"
        consumer = directory / "consumer"
        consumer.mkdir()
        requirements = directory / "runtime.txt"
        commands = [
            [
                "uv",
                "export",
                "--locked",
                "--no-dev",
                "--extra",
                "torch-reference",
                "--no-emit-project",
                "--no-header",
                "--no-annotate",
                "--output-file",
                str(requirements),
            ],
            ["uv", "venv", "--python", sys.executable, str(environment)],
        ]
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        # Follow locked artifact URLs directly; requirements resolution additionally needs
        # index metadata that can be absent from an otherwise warm package cache.
        sync = [
            "uv",
            "sync",
            "--locked",
            "--offline",
            "--no-dev",
            "--extra",
            "torch-reference",
            "--no-install-project",
        ]
        commands += [
            sync,
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                "--no-index",
                "--no-deps",
                str(wheel),
            ],
        ]
        for command in commands:
            result = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                capture_output=True,
                env={
                    **{k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"},
                    "UV_PROJECT_ENVIRONMENT": str(environment),
                },
            )
            if result.returncode:
                raise RuntimeError(f"{command[:3]} failed:\n{result.stdout}\n{result.stderr}")
        # Isolated mode ignores PYTHONPATH/CWD; assert that this is an ordinary wheel installation.
        probe = """
import importlib.metadata, importlib.util, json, sys
from pathlib import Path
import kaleido
root=Path(sys.argv[1]).resolve()
module=Path(kaleido.__file__).resolve()
assert module.is_relative_to(Path(sys.argv[2]).resolve())
assert not any(Path(p).resolve().is_relative_to(root) for p in sys.path if p)
assert importlib.util.find_spec("transformers") is None
assert importlib.util.find_spec("huggingface_hub") is None
print(json.dumps({"hf_packages_required":False,"module":str(module),"version":importlib.metadata.version("kaleido"),"source_on_sys_path":False}))
"""
        result = subprocess.run(
            [str(python), "-I", "-c", probe, str(ROOT), str(environment)],
            cwd=consumer,
            text=True,
            capture_output=True,
            check=True,
        )
        installation = json.loads(result.stdout)
        import tomllib

        expected_version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
            "version"
        ]
        if installation["version"] != expected_version:
            raise ValueError("wheel metadata differs from the checked-out release version")
        runtime = """
import sys
from kaleido.cli import main
status=main(["predict","--bundle",sys.argv[1]])
assert "kaleido.lm.model" not in sys.modules
assert "kaleido.lm.training" not in sys.modules
assert "kaleido.posttrain.model" not in sys.modules
raise SystemExit(status)
"""
        requests = "".join(json.dumps(r["request"]) + "\n" for r in examples)
        result = subprocess.run(
            [str(python), "-I", "-c", runtime, str(bundle)],
            cwd=consumer,
            input=requests,
            text=True,
            capture_output=True,
            check=True,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        )
        rows = [json.loads(line) for line in result.stdout.splitlines()]
        if len(rows) != len(examples) or any(
            r["completion_ids"] != e["completion_ids"] or r["completion"] != e["completion"]
            for r, e in zip(rows, examples, strict=True)
        ):
            raise AssertionError("installed wheel changed validated model outputs")
        return {
            "status": "completed",
            "wheel": file_info(wheel),
            "installation": installation,
            "runtime_requirements": file_info(requirements),
            "runtime_lockfile": file_info(ROOT / "uv.lock"),
            "installation_method": (
                "uv sync --locked --offline --no-install-project into explicit fresh venv"
            ),
            "offline_install": True,
            "no_training_imports": True,
            "outside_checkout": True,
            "isolated_python": True,
            "requests": rows,
            "exact_token_replay": True,
            "checker_script": file_info(Path(__file__)),
            "scope": (
                "fresh venv with locked cached runtime, non-editable wheel; "
                "no checkout or HF needed for inference"
            ),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = check(args.wheel, args.bundle)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        write_json(
            args.report, {"status": "failed", "error": type(error).__name__, "message": str(error)}
        )
        raise
    write_json(args.report, result)
    print(json.dumps({"status": "completed", "report": str(args.report.resolve())}))


if __name__ == "__main__":
    main()
