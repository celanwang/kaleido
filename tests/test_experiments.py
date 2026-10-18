import json

import numpy as np
import pytest

from kaleido.experiments import Run, array_digest


def test_records_success_and_never_overwrites(tmp_path):
    with Run(tmp_path, {"seed": 7}, root=tmp_path) as first:
        first.record("metrics.json", {"mse": 0.1})
    with Run(tmp_path, {"seed": 7}, root=tmp_path) as second:
        pass
    manifest = json.loads((first.path / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["config"] == {"seed": 7}
    assert manifest["environment"]["git_revision"] is None
    assert first.path != second.path
    assert json.loads((first.path / "metrics.json").read_text()) == {"mse": 0.1}


def test_failure_and_nonfinite_metric_are_visible(tmp_path):
    with pytest.raises(ValueError), Run(tmp_path, {}, root=tmp_path) as run:
        run.record("metrics.json", {"mse": float("nan")})
    manifest = json.loads((run.path / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error"]["type"] == "ValueError"
    assert not (run.path / "metrics.json").exists()


@pytest.mark.parametrize("name", ["../escape.json", "manifest.json", "metrics.txt", "/x.json"])
def test_artifact_names_are_local(tmp_path, name):
    with Run(tmp_path, {}, root=tmp_path) as run, pytest.raises(ValueError):
        run.record(name, {})


def test_data_digest_tracks_content_shape_and_dtype():
    a = np.arange(6, dtype=np.float64)
    assert array_digest(a) == array_digest(a.copy())
    assert array_digest(a) != array_digest(a.reshape(2, 3))
    assert array_digest(a) != array_digest(a.astype(np.float32))
    assert array_digest(a) != array_digest(a + 1)
    with pytest.raises(TypeError):
        array_digest(np.array([object()]))
