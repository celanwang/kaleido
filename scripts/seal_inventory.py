"""Maintainer-only inventory creation; verification never regenerates expected hashes."""

from pathlib import Path

from kaleido.experiments import write_json
from kaleido.repro.integrity import file_info

ROOT = Path(__file__).resolve().parents[1]
files = [ROOT / p for p in (".python-version", "pyproject.toml", "uv.lock")]
for directory in ("configs", "data", *(f"results/p{i}" for i in range(6))):
    files.extend(p for p in (ROOT / directory).rglob("*") if p.is_file())
write_json(
    ROOT / "artifacts/inventory-v1.json",
    {
        "schema_version": 1,
        "scope": (
            "P0-P5 reviewed evidence, source data/configs and dependency pins; "
            "excludes future P6 evidence"
        ),
        "files": {str(p.relative_to(ROOT)): file_info(p) for p in sorted(set(files))},
    },
)
