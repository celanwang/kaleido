"""Portable byte/array inventories with explicit roots and no implicit trust bootstrap."""

import hashlib
import json
from pathlib import Path, PurePosixPath

from kaleido.experiments import array_digest


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
    if (
        not isinstance(name, str)
        or not name
        or pure.is_absolute()
        or ".." in pure.parts
        or "\\" in name
        or str(pure) != name
    ):
        raise ValueError("inventory paths must be canonical relative POSIX paths")
    root = Path(root).resolve()
    path = root / name
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("inventory path escapes root or is a symlink")
    return path


def verify_index(root, index, *, deep=False):
    if (
        index.get("schema_version") != 1
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
            set(expected) != {"sha256", "bytes"}
            or type(expected["bytes"]) is not int
            or expected["bytes"] < 0
            or not isinstance(expected["sha256"], str)
            or len(expected["sha256"]) != 64
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

            info = json.loads(path.with_suffix(".json").read_text())
            digests = info.get("arrays", info.get("array_digests"))
            if digests is not None:
                with np.load(path, allow_pickle=False) as state:
                    if set(state) != set(digests) or any(
                        array_digest(state[k]) != digests[k] for k in state
                    ):
                        errors.append({"path": name, "error": "array_digest_mismatch"})
                    arrays += len(state)
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
