"""Standalone delivery installer: standard library only, no source checkout required."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath


def identity(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return {"bytes": path.stat().st_size, "sha256": h.hexdigest()}


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        not isinstance(manifest, dict)
        or manifest.get("kind") != "kaleido_cpu_delivery"
        or manifest.get("schema_version") != 1
    ):
        raise ValueError("unsupported delivery manifest")
    if not isinstance(manifest.get("wheel"), str) or not isinstance(manifest.get("files"), dict):
        raise ValueError("delivery requires a wheel name and file mapping")
    required = {
        "install.py",
        "README.md",
        "pylock.cpu.toml",
        manifest["wheel"],
        "decoder-42/manifest.json",
        "decoder-42/model.pt2",
        "decoder-42/tokenizer.json",
        "decoder-42/examples.json",
        "decoder-42/validation.json",
    }
    if set(manifest["files"]) != required:
        raise ValueError("delivery file set differs from the supported format")
    for name, expected in manifest["files"].items():
        pure = PurePosixPath(name)
        path = root / name
        if (
            pure.is_absolute()
            or ".." in pure.parts
            or "\\" in name
            or str(pure) != name
            or path.is_symlink()
            or not path.resolve().is_relative_to(root)
            or not path.is_file()
            or identity(path) != expected
        ):
            raise ValueError(f"delivery integrity failed: {name}")
    return manifest


def install(root, environment, *, offline=False, cache_dir=None):
    root, environment = Path(root).resolve(), Path(environment).resolve()
    manifest = verify(root)
    if sys.version_info[:2] != (3, 12):
        raise ValueError("Python 3.12 is required")
    if environment.exists() or environment.is_relative_to(root / "decoder-42"):
        raise ValueError("choose a new environment outside the decoder directory")
    uv = subprocess.run(["uv", "--version"], text=True, capture_output=True, check=True)
    if uv.stdout.split()[1] != manifest["uv_version"]:
        raise ValueError(f"uv {manifest['uv_version']} is required")
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("VIRTUAL_ENV", "PYTHONPATH", "UV_PROJECT_ENVIRONMENT", "UV_PROJECT")
    }
    if cache_dir is not None:
        env["UV_CACHE_DIR"] = str(Path(cache_dir).resolve())
    environment.mkdir(parents=True, exist_ok=False)
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    commands = [
        [
            "uv",
            "--no-config",
            "venv",
            "--allow-existing",
            "--python",
            sys.executable,
            str(environment),
        ],
        [
            "uv",
            "--no-config",
            "pip",
            "sync",
            "--python",
            str(python),
            str(root / "pylock.cpu.toml"),
            *(["--offline"] if offline else []),
        ],
        [
            "uv",
            "--no-config",
            "pip",
            "install",
            "--python",
            str(python),
            "--no-index",
            "--no-deps",
            str(root / manifest["wheel"]),
        ],
    ]
    for command in commands:
        result = subprocess.run(command, cwd=environment, env=env, text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(f"installation failed: {result.stderr}")
    probe = """
import importlib.metadata, importlib.util, json, sys
from pathlib import Path
import kaleido
assert Path(kaleido.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve())
assert importlib.util.find_spec("transformers") is None
assert importlib.util.find_spec("huggingface_hub") is None
print(json.dumps({"version":importlib.metadata.version("kaleido"),"module":kaleido.__file__}))
"""
    result = subprocess.run(
        [str(python), "-I", "-c", probe, str(environment)],
        cwd=environment,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    installed = json.loads(result.stdout)
    if installed["version"] != manifest["version"]:
        raise ValueError("installed wheel version differs from delivery")
    examples = json.loads((root / "decoder-42/examples.json").read_text())["cases"]
    if not examples:
        raise ValueError("delivery must contain non-empty replay examples")
    runtime = """
import sys
from kaleido.cli import main
status=main(["predict","--bundle",sys.argv[1]])
assert "kaleido.lm.training" not in sys.modules
assert "kaleido.lm.model" not in sys.modules
raise SystemExit(status)
"""
    result = subprocess.run(
        [str(python), "-I", "-c", runtime, str(root / "decoder-42")],
        cwd=environment,
        env=env,
        text=True,
        capture_output=True,
        check=True,
        input="".join(json.dumps(r["request"]) + "\n" for r in examples),
    )
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    if len(rows) != len(examples) or any(
        r["completion_ids"] != e["completion_ids"] for r, e in zip(rows, examples, strict=True)
    ):
        raise ValueError("installed delivery changed validated token outputs")
    return {
        "status": "completed",
        "installation": installed,
        "offline": offline,
        "exact_token_replay": True,
        "requests": len(rows),
        "no_hf_packages": True,
        "no_training_imports": True,
        "manifest": identity(root / "manifest.json"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--cache-dir", type=Path, help="explicit prepared uv cache for offline installation"
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    try:
        if args.verify_only:
            verify(root)
            result = {"status": "completed", "verified": True}
        else:
            if args.env is None:
                raise ValueError("--env must name a new virtual environment")
            result = install(root, args.env, offline=args.offline, cache_dir=args.cache_dir)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        RuntimeError,
        subprocess.CalledProcessError,
    ) as error:
        print(json.dumps({"status": "failed", "message": str(error)}))
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
