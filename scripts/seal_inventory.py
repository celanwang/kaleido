"""Maintainer-only inventory creation; verification never regenerates expected hashes."""

from pathlib import Path

from kaleido.experiments import write_json
from kaleido.repro.integrity import file_info, reviewed_files

ROOT = Path(__file__).resolve().parents[1]
files = [ROOT / name for name in reviewed_files(ROOT)]
write_json(
    ROOT / "artifacts/inventory-v1.json",
    {
        "schema_version": 1,
        "scope": (
            "P0-P6 reviewed evidence, source data/configs and dependency pins; "
            "includes final delivery evidence"
        ),
        "files": {str(p.relative_to(ROOT)): file_info(p) for p in sorted(set(files))},
    },
)
