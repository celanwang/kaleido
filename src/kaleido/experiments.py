"""Local experiment records: provenance, finite JSON, and failure-aware finalization."""

import hashlib
import json
import platform
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


def write_json(path: Path, value: object) -> None:
    """Replace a JSON artifact atomically; reject NaN/Inf instead of hiding divergence."""
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload + "\n", encoding="utf-8")
    temporary.replace(path)


def array_digest(*arrays: np.ndarray) -> str:
    """Include dtype and shape so differently interpreted bytes cannot alias."""
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject:
            raise TypeError("object arrays do not have a reproducible byte representation")
        digest.update(json.dumps([array.dtype.str, array.shape]).encode() + b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def provenance(root: Path) -> dict:
    def git(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *args], stderr=subprocess.DEVNULL, text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    # Hash executable sources, configs, and dependency pins even before a commit exists.
    files = sorted(
        [*root.glob("src/**/*.py"), *root.glob("configs/**/*.json")]
        + [root / name for name in ("pyproject.toml", "uv.lock", ".python-version")]
    )
    digest = hashlib.sha256()
    for path in files:
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode() + b"\0")
            digest.update(path.read_bytes() + b"\0")
    status = git("status", "--porcelain")
    return {
        "git_revision": git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "git_status": status,
        "source_sha256": digest.hexdigest(),
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "command": sys.argv,
    }


class Run:
    """A unique directory per attempt; use as a context manager to record exceptions."""

    def __init__(self, output: Path, config: dict, *, root: Path):
        self.path = output / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8])
        self.path.mkdir(parents=True, exist_ok=False)
        self.metadata = {
            "schema_version": 1,
            "status": "running",
            "started_at": datetime.now(UTC).isoformat(),
            "config": config,
            "environment": provenance(root),
        }
        write_json(self.path / "manifest.json", self.metadata)

    def __enter__(self):
        return self

    def record(self, name: str, value: object) -> None:
        if Path(name).name != name or not name.endswith(".json") or name == "manifest.json":
            raise ValueError("artifact must be a local .json filename other than manifest.json")
        write_json(self.path / name, value)

    def __exit__(self, exc_type, exc_value, traceback):
        self.metadata.update(
            status="completed" if exc_type is None else "failed",
            finished_at=datetime.now(UTC).isoformat(),
        )
        if exc_type is not None:
            self.metadata["error"] = {"type": exc_type.__name__, "message": str(exc_value)}
        write_json(self.path / "manifest.json", self.metadata)
        return False
