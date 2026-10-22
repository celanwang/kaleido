import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from kaleido.experiments import Run, write_json
from kaleido.repro.integrity import file_info, verify_repository
from kaleido.repro.registry import reproduce, run_recipe
from kaleido.serialization import json_object

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("raw", ["[]", "null", '{"a":1,"a":2}', '{"x":NaN}', '{"x":1e9999}'])
def test_metadata_parser_rejects_ambiguous_or_non_object_json(raw):
    with pytest.raises(ValueError):
        json_object(raw)


def test_p6_is_checked_and_missing_inventory_entries_cannot_be_ignored(tmp_path):
    target = tmp_path / "results/p6/reference/bundle/model.pt2"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"reviewed graph")
    (tmp_path / "artifacts").mkdir()
    write_json(
        tmp_path / "artifacts/inventory-v1.json",
        {"schema_version": 1, "files": {str(target.relative_to(tmp_path)): file_info(target)}},
    )
    assert verify_repository(tmp_path)["ok"]
    target.write_bytes(b"corrupt graph")
    assert verify_repository(tmp_path)["errors"][0]["error"] == "checksum_mismatch"
    extra = target.parent / "unindexed.pt2"
    extra.write_bytes(b"unexpected graph")
    assert any(
        r["error"] == "unindexed_reviewed_file" for r in verify_repository(tmp_path)["errors"]
    )


def test_config_receipt_records_the_bytes_actually_used_not_later_file_edits(tmp_path, monkeypatch):
    from kaleido.demos.linear_regression import Config
    from kaleido.repro import registry

    source = tmp_path / "custom.json"
    original = b'{"seed":42,"epochs":1}\n'
    source.write_bytes(original)

    def execute(config, output, *, root):
        with Run(output, asdict(config), root=root) as run:
            source.write_text('{"seed":7,"epochs":3}')
            return run.path, {}

    original_import = registry.importlib.import_module
    monkeypatch.setattr(
        registry.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(Config=Config, execute=execute)
            if name == "kaleido.demos.linear_regression"
            else original_import(name)
        ),
    )
    receipt = run_recipe("p0", ROOT, tmp_path / "runs", config_path=source)
    run = Path(receipt["run"])
    assert (run / "configuration.json").read_bytes() == original
    assert receipt["effective_config"]["seed"] == 42
    assert receipt["config_identity"] == file_info(run / "configuration.json")
    assert receipt["config_changed_during_run"] and not receipt["canonical_config"]
    assert json.loads((run / "receipt.json").read_text())["stage"] == "p0"


def test_zero_exit_cannot_launder_an_unrelated_run_into_success(tmp_path, monkeypatch):
    from kaleido.repro import registry

    other = tmp_path / "unrelated"
    other.mkdir()
    write_json(other / "manifest.json", {"status": "completed"})

    def fake_success(command, **kwargs):
        kwargs["stdout"].write(json.dumps({"status": "completed", "run": str(other)}) + "\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(registry, "run_process", fake_success)
    output = tmp_path / "suite"
    with pytest.raises(RuntimeError):
        reproduce(ROOT, output, stages=["p0"])
    run = next(output.iterdir())
    assert json.loads((run / "manifest.json").read_text())["status"] == "failed"
    assert json.loads((run / "steps.json").read_text())["error"] == "invalid_child_result"


def test_snapshot_pins_tokenizer_and_configs_as_well_as_weights(tmp_path):
    pytest.importorskip("torch")
    from kaleido.posttrain.model import validate_snapshot

    names = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "LICENSE",
        "merges.txt",
        "vocab.json",
    ]
    for name in ["model.safetensors", *names]:
        (tmp_path / name).write_bytes(b"fixed official content")
    spec = {
        "weights_file": "model.safetensors",
        "weights_sha256": file_info(tmp_path / "model.safetensors")["sha256"],
        "files_sha256": {name: file_info(tmp_path / name)["sha256"] for name in names},
    }
    assert len(validate_snapshot(tmp_path, spec)) == 8
    (tmp_path / "tokenizer.json").write_text("changed tokenizer with unchanged base weights")
    with pytest.raises(ValueError, match="tokenizer.json"):
        validate_snapshot(tmp_path, spec)
