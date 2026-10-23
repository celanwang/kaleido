"""Verify the actual frozen pretraining artifact, without retraining."""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kaleido.experiments import array_digest  # noqa:E402
from kaleido.lm.corpus import prepare_corpus  # noqa:E402
from kaleido.lm.tokenizer import ByteBPE  # noqa:E402
from kaleido.lm.training import TokenCorpus, count_baselines, evaluate, load_weights  # noqa:E402

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "results/p3/reference"


@pytest.fixture(scope="module")
def archive():
    jsons = {p.stem: json.loads(p.read_text()) for p in FOLDER.glob("*.json")}
    docs, audit = prepare_corpus(
        ROOT / "data/p3", split_seed=jsons["manifest"]["config"]["split_seed"]
    )
    tokenizer = ByteBPE.load(FOLDER / "tokenizer.json")
    return jsons, TokenCorpus(docs, tokenizer), docs, audit, tokenizer


def test_frozen_corpus_tokenizer_and_baselines(archive):
    jsons, corpus, docs, audit, tokenizer = archive
    assert (
        jsons["manifest"]["status"] == "completed"
        and jsons["manifest"]["environment"]["git_dirty"] is False
    )
    assert audit == jsons["corpus"]["audit"]
    trained = ByteBPE.fit(
        [d["text"] for d in docs if d["split"] == "train"],
        vocab_size=jsons["manifest"]["config"]["vocab_size"],
    )
    assert trained.to_dict() == tokenizer.to_dict()
    arrays = dict(np.load(FOLDER / "tokens.npz", allow_pickle=False))
    for doc in corpus.documents:
        np.testing.assert_array_equal(arrays[doc["id"]], doc["tokens"])
        assert array_digest(arrays[doc["id"]]) == jsons["tokens"]["arrays"][doc["id"]]
    for split in ("valid", "test"):
        observed = count_baselines(corpus, split)
        for method, metrics in observed.items():
            for name, value in metrics.items():
                assert value == pytest.approx(jsons["baselines"][split][method][name], abs=1e-10)


def test_frozen_per_token_scores_and_validation_selection(archive):
    jsons, _, _, _, _ = archive
    arrays = dict(np.load(FOLDER / "evaluation.npz", allow_pickle=False))
    for name, array in arrays.items():
        assert array_digest(array) == jsons["evaluation"]["arrays"][name]
    for seed, metrics in jsons["metrics"].items():
        history = jsons["histories"][seed]
        assert (
            min(history, key=lambda row: row["valid_nll"])["step"]
            == metrics["training"]["selected_step"]
        )
        assert metrics["training"]["steps"] == 1000
        assert metrics["training"]["actual_training_target_tokens"] > 2_000_000
        losses = arrays[seed + "__test_nll"]
        assert len(losses) == metrics["selected_test"]["target_tokens"]
        assert losses.mean() == pytest.approx(metrics["selected_test"]["nll"], abs=1e-12)
        assert metrics["selected_test"]["nll"] < jsons["baselines"]["test"]["bigram"]["nll"]
        windows = jsons["evaluation"]["details"][seed]["selected_test"]["windows"]
        assert sum(w["target_tokens"] for w in windows) == len(losses)
        assert sum(w["nll_sum"] for w in windows) == pytest.approx(float(losses.sum()))


@pytest.mark.parametrize("seed", [42, 7, 2026])
def test_frozen_weights_recompute_mps_scores_on_cpu(archive, seed):
    jsons, corpus, _, _, _ = archive
    torch.set_num_threads(1)
    model = load_weights(FOLDER / f"model-{seed}.npz", device="cpu")
    metrics, _, losses = evaluate(model, corpus, "test", batch_size=16)
    values = dict(np.load(FOLDER / "evaluation.npz", allow_pickle=False))[str(seed) + "__test_nll"]
    # MPS and CPU GEMM/softmax reduction orders differ at float32 precision.
    np.testing.assert_allclose(losses, values, atol=5e-5, rtol=1e-5)
    assert metrics["nll"] == pytest.approx(
        jsons["metrics"][str(seed)]["selected_test"]["nll"], abs=2e-6
    )
