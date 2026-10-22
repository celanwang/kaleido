"""Portable byte/array inventories with explicit roots and no implicit trust bootstrap."""

import hashlib
import re
import zipfile
from pathlib import Path, PurePosixPath

from kaleido.experiments import array_digest
from kaleido.serialization import load_object


def file_info(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            hasher.update(block)
    return {"sha256": hasher.hexdigest(), "bytes": Path(path).stat().st_size}


def local_path(root, name):
    if not isinstance(name, str):
        raise ValueError("inventory paths must be text")
    pure = PurePosixPath(name)
    if not name or pure.is_absolute() or ".." in pure.parts or "\\" in name or str(pure) != name:
        raise ValueError("inventory paths must be canonical relative POSIX paths")
    root = Path(root).resolve()
    path = root / name
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("inventory path escapes root or is a symlink")
    return path


def verify_index(root, index, *, deep=False):
    if (
        not isinstance(index, dict)
        or type(index.get("schema_version")) is not int
        or index["schema_version"] != 1
        or not isinstance(index.get("files"), dict)
        or not index["files"]
    ):
        raise ValueError("unsupported or empty integrity inventory")
    errors = []
    checked = 0
    arrays = 0
    for name, expected in sorted(index["files"].items()):
        path = local_path(root, name)
        if (
            not isinstance(expected, dict)
            or set(expected) != {"sha256", "bytes"}
            or type(expected["bytes"]) is not int
            or expected["bytes"] < 0
            or not isinstance(expected["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", expected["sha256"]) is None
        ):
            raise ValueError("invalid inventory file identity")
        if not path.is_file():
            errors.append({"path": name, "error": "missing"})
            continue
        actual = file_info(path)
        checked += 1
        if actual != expected:
            errors.append({"path": name, "error": "checksum_mismatch", "actual": actual})
            continue
        if deep and path.suffix == ".npz" and path.with_suffix(".json").is_file():
            import numpy as np

            try:
                info = load_object(path.with_suffix(".json"))
                digests = info.get("arrays", info.get("array_digests"))
                if digests is not None:
                    if not isinstance(digests, dict):
                        raise ValueError("array digests must be a mapping")
                    with np.load(path, allow_pickle=False) as state:
                        if set(state) != set(digests) or any(
                            array_digest(state[k]) != digests[k] for k in state
                        ):
                            errors.append({"path": name, "error": "array_digest_mismatch"})
                        arrays += len(state)
            except (OSError, ValueError, TypeError, zipfile.BadZipFile) as error:
                errors.append(
                    {"path": name, "error": "invalid_array_artifact", "detail": str(error)}
                )
    return {
        "ok": not errors,
        "expected_files": len(index["files"]),
        "checked_files": checked,
        "checked_arrays": arrays,
        "errors": errors,
        "scope": (
            "integrity relative to the reviewed inventory; "
            "not external authenticity or scientific correctness"
        ),
    }


REVIEWED_DIRECTORIES = ("configs", "data", *(f"results/p{i}" for i in range(7)), "results/final")
REVIEWED_PINS = (".python-version", "pyproject.toml", "uv.lock")


def reviewed_files(root):
    root = Path(root)
    files = {name for name in REVIEWED_PINS if (root / name).is_file()}
    files.update(
        str(p.relative_to(root))
        for d in REVIEWED_DIRECTORIES
        for p in (root / d).rglob("*")
        if p.is_file()
    )
    return sorted(files)


def verify_repository(root, *, deep=False):
    root = Path(root)
    index = load_object(root / "artifacts/inventory-v1.json")
    result = verify_index(root, index, deep=deep)
    missing_entries = sorted(set(reviewed_files(root)) - set(index["files"]))
    result["errors"].extend(
        {"path": p, "error": "unindexed_reviewed_file"} for p in missing_entries
    )
    result["ok"] = not result["errors"]
    result["coverage"] = "configs, data, P0-P6/final delivery evidence and dependency pins"
    return result
