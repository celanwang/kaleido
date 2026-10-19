import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kaleido.experiments import array_digest, write_json
from kaleido.repro.integrity import file_info, local_path, verify_index
from kaleido.repro.registry import reproduce, run_recipe

ROOT = Path(__file__).resolve().parents[1]


def test_reviewed_inventory_covers_all_inputs_and_archives():
    index = json.loads((ROOT / "artifacts/inventory-v1.json").read_text())
    actual = verify_index(ROOT, index, deep=True)
    assert actual["ok"] and actual["checked_arrays"] > 500
    required = {
        str(p.relative_to(ROOT))
        for d in ["configs", "data", *(f"results/p{i}" for i in range(6))]
        for p in (ROOT / d).rglob("*")
        if p.is_file()
    }
    assert required <= set(index["files"])


def test_corruption_missing_escape_and_npz_metadata_fail_closed(tmp_path):
    (tmp_path / "a.txt").write_text("reviewed")
    index = {"schema_version": 1, "files": {"a.txt": file_info(tmp_path / "a.txt")}}
    assert verify_index(tmp_path, index)["ok"]
    (tmp_path / "a.txt").write_text("modified")
    assert verify_index(tmp_path, index)["errors"][0]["error"] == "checksum_mismatch"
    (tmp_path / "a.txt").unlink()
    assert verify_index(tmp_path, index)["errors"][0]["error"] == "missing"
    for name in ("../outside", "/absolute", "x/../y", "a\\b", "a//b", ""):
        with pytest.raises(ValueError):
            local_path(tmp_path, name)
    (tmp_path / "link").symlink_to(tmp_path.parent)
    with pytest.raises(ValueError):
        local_path(tmp_path, "link/outside")
    np.savez(tmp_path / "state.npz", weight=np.ones(3))
    write_json(tmp_path / "state.json", {"arrays": {"weight": array_digest(np.zeros(3))}})
    index = {"schema_version": 1, "files": {"state.npz": file_info(tmp_path / "state.npz")}}
    assert verify_index(tmp_path, index, deep=True)["errors"][0]["error"] == "array_digest_mismatch"


def test_cli_module_and_console_recipe_from_outside_root(tmp_path):
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "kaleido",
            "--root",
            str(ROOT),
            "run",
            "p0",
            "--output",
            str(tmp_path / "runs"),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    result = json.loads(process.stdout.splitlines()[-1])
    assert result["status"] == "completed" and result["profile"] == "formal"
    assert result["config_identity"] == file_info(ROOT / "configs/p0_linear.json")
    assert Path(result["run"]).is_dir()
    with pytest.raises(ValueError):
        run_recipe("p0", tmp_path, tmp_path / "bad")


def test_reproduction_failure_retains_evidence_and_does_not_continue(tmp_path, monkeypatch):
    calls = []

    def fail(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=7, stdout="partial output", stderr="failure")

    monkeypatch.setattr("kaleido.repro.registry.run_process", fail)
    with pytest.raises(RuntimeError):
        reproduce(ROOT, tmp_path, stages=["p0", "p1"])
    assert len(calls) == 1
    run = next(tmp_path.iterdir())
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert json.loads((run / "steps.json").read_text())["failed_stage"] == "p0"
    assert (run / "p0.stderr.log").read_text() == "failure"
