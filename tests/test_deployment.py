import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from kaleido.deployment.export import export_decoder  # noqa: E402
from kaleido.deployment.runtime import MAX_LINE_BYTES, Predictor, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    path = tmp_path_factory.mktemp("export") / "bundle"
    export_decoder(
        ROOT / "results/p3/reference/model-42.npz",
        ROOT / "results/p3/reference/tokenizer.json",
        path,
        root=ROOT,
    )
    return path


def test_actual_trained_export_matches_eager_boundaries_and_cropped_generation(bundle):
    predictor = Predictor(bundle)
    validation = json.loads((bundle / "validation.json").read_text())
    assert validation["passed"] and {r["length"] for r in validation["logits"]} == {
        1,
        2,
        7,
        127,
        128,
    }
    examples = json.loads((bundle / "examples.json").read_text())["cases"]
    for row in examples:
        result = predictor.predict(row["request"])
        assert result["completion_ids"] == row["completion_ids"]
    assert predictor.predict(examples[-1]["request"])["initial_cropped_tokens"] > 0
    for bad in ([], [512], [True], list(range(129))):
        with pytest.raises(ValueError):
            predictor.logits(bad)
    with pytest.raises(ValueError):
        predictor.predict({"prompt": "a", "max_new_tokens": 129})
    with pytest.raises(ValueError):
        predictor.predict({"prompt": "a", "max_new_tokens": True})
    assert predictor.predict({"prompt": "a", "max_new_tokens": 0})["completion_ids"] == []


def test_loading_rejects_corruption_and_export_does_not_overwrite(bundle, tmp_path):
    copy = tmp_path / "bad"
    shutil.copytree(bundle, copy)
    with (copy / "model.pt2").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        Predictor(copy)
    with pytest.raises(FileExistsError):
        export_decoder(
            ROOT / "results/p3/reference/model-42.npz",
            ROOT / "results/p3/reference/tokenizer.json",
            bundle,
            root=ROOT,
        )


def test_export_failure_leaves_no_published_bundle(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("simulated save failure")

    monkeypatch.setattr(torch.export, "save", fail)
    with pytest.raises(RuntimeError):
        export_decoder(
            ROOT / "results/p3/reference/model-42.npz",
            ROOT / "results/p3/reference/tokenizer.json",
            tmp_path / "bundle",
            root=ROOT,
        )
    assert not (tmp_path / "bundle").exists()
    assert not list(tmp_path.glob(".kaleido-export-*"))


def test_json_lines_recovers_per_request_and_does_not_split_oversized_line(bundle):
    predictor = Predictor(bundle)
    valid = b'{"prompt":"Alice was","max_new_tokens":2}\n'
    data = b"invalid\n" + b"x" * (MAX_LINE_BYTES + 20) + b"\n" + valid
    output = io.StringIO()
    assert serve(predictor, io.BytesIO(data), output) == 1
    rows = [json.loads(line) for line in output.getvalue().splitlines()]
    assert len(rows) == 3 and [r["request_id"] for r in rows] == [1, 2, 3]
    assert [r["status"] for r in rows] == ["failed", "failed", "completed"]


def test_standalone_runtime_from_outside_checkout_does_not_import_training(bundle, tmp_path):
    script = """
import json, sys
from kaleido.deployment.runtime import Predictor
p=Predictor(sys.argv[1])
print(json.dumps(p.predict({"prompt":"Alice was","max_new_tokens":3})))
assert "kaleido.lm.model" not in sys.modules
assert "kaleido.lm.training" not in sys.modules
assert "kaleido.posttrain.model" not in sys.modules
"""
    process = subprocess.run(
        [sys.executable, "-I", "-c", script, str(bundle)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    assert json.loads(process.stdout)["status"] == "completed"


@pytest.mark.parametrize("field,value", [("runtime", []), ("model_config", []), ("files", [])])
def test_bundle_schema_errors_are_reported_as_value_errors(bundle, tmp_path, field, value):
    copy = tmp_path / "schema"
    shutil.copytree(bundle, copy)
    manifest = json.loads((copy / "manifest.json").read_text())
    manifest[field] = value
    (copy / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        Predictor(copy)


def test_manifest_cannot_claim_a_context_larger_than_the_actual_graph(bundle, tmp_path):
    copy = tmp_path / "dimensions"
    shutil.copytree(bundle, copy)
    manifest = json.loads((copy / "manifest.json").read_text())
    manifest["model_config"]["context_length"] = 256
    (copy / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="embedding"):
        Predictor(copy)


def test_cli_returns_a_structured_error_for_a_non_object_manifest(tmp_path):
    (tmp_path / "manifest.json").write_text("[]")
    process = subprocess.run(
        [sys.executable, "-m", "kaleido", "predict", "--bundle", str(tmp_path)],
        text=True,
        capture_output=True,
    )
    assert process.returncode == 1
    assert json.loads(process.stdout)["status"] == "failed"
    assert "Traceback" not in process.stderr
