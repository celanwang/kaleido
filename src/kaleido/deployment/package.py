"""Build a portable wheel/model/locked-runtime delivery without bundling dependency wheels."""

import json
import shutil
import subprocess
import tempfile
import tomllib
import zipfile
from email.parser import Parser
from pathlib import Path

from kaleido.deployment.runtime import Predictor
from kaleido.experiments import provenance, write_json
from kaleido.repro.integrity import file_info

README = """# Kaleido CPU delivery

Requires Python 3.12 and uv 0.12.23. Validated on macOS arm64 and Linux x86_64.
No source checkout, Qwen weights or training data are needed.

    python3.12 install.py --verify-only
    python3.12 install.py --env /absolute/path/to/new-venv

The first install downloads the exact runtime artifacts in pylock.cpu.toml.
With a prepared uv wheel cache, add --offline --cache-dir /absolute/path/to/cache.
Dependency wheels are not included.
The installer refuses an existing environment and preserves failed environments for diagnosis.

    /absolute/path/to/new-venv/bin/kaleido predict --bundle /absolute/path/to/decoder-42

Input one JSON line, e.g. {"prompt":"Alice was","max_new_tokens":12}.
CPU, batch 1, greedy cropped-window decoding, PyTorch 2.14.1; no HTTP server or KV cache export.
Checksums ensure consistency with this manifest, not an independent signature.
"""


def package_delivery(wheel, bundle, output, *, root):
    root, wheel, bundle, output = map(lambda p: Path(p).resolve(), (root, wheel, bundle, output))
    if output.exists():
        raise FileExistsError("delivery destination exists")
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    with zipfile.ZipFile(wheel) as archive:
        names = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        if len(names) != 1:
            raise ValueError("wheel must contain exactly one distribution metadata file")
        metadata = Parser().parsestr(archive.read(names[0]).decode())
        if metadata["Name"] != "kaleido" or metadata["Version"] != version:
            raise ValueError("wheel name/version differs from this release")
    predictor = Predictor(bundle)
    examples = json.loads((bundle / "examples.json").read_text())["cases"]
    for row in examples:
        if predictor.predict(row["request"])["completion_ids"] != row["completion_ids"]:
            raise ValueError("bundle examples do not replay")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".delivery-", dir=output.parent) as temporary:
        staging = Path(temporary) / "delivery"
        staging.mkdir()
        shutil.copyfile(wheel, staging / wheel.name)
        shutil.copytree(bundle, staging / "decoder-42")
        shutil.copyfile(Path(__file__).with_name("install.py"), staging / "install.py")
        (staging / "README.md").write_text(README)
        command = [
            "uv",
            "export",
            "--locked",
            "--no-dev",
            "--extra",
            "torch-reference",
            "--no-emit-project",
            "--format",
            "pylock.toml",
            "--no-header",
            "--output-file",
            str(staging / "pylock.cpu.toml"),
        ]
        result = subprocess.run(command, cwd=root, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"locked runtime export failed: {result.stderr}")
        write_json(
            staging / "manifest.json",
            {
                "schema_version": 1,
                "kind": "kaleido_cpu_delivery",
                "version": version,
                "uv_version": "0.12.23",
                "wheel": wheel.name,
                "producer": provenance(root),
                "files": {
                    str(p.relative_to(staging)): file_info(p)
                    for p in sorted(staging.rglob("*"))
                    if p.is_file()
                },
            },
        )
        from kaleido.deployment.install import verify

        verify(staging)
        staging.rename(output)
    return output
