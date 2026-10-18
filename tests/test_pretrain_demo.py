import hashlib
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kaleido.demos.language_model import Config, execute  # noqa:E402
from kaleido.experiments import array_digest  # noqa:E402
from kaleido.lm.corpus import prepare_corpus  # noqa:E402
from kaleido.lm.tokenizer import ByteBPE  # noqa:E402
from kaleido.lm.training import TokenCorpus, evaluate, load_weights  # noqa:E402


def source(tmp_path):
    directory = tmp_path / "corpus"
    (directory / "raw").mkdir(parents=True)
    text = (
        "*** START OF THE TEST EBOOK ***\n"
        + "".join(
            f"CHAPTER {numeral}\nTitle {i}\n\n"
            + f"Alice saw rabbit number {i}. "
            + "The girl said hello. " * 8
            + "\n\n"
            for i, numeral in enumerate(("I", "II", "III"), 1)
        )
        + "*** END OF THE TEST EBOOK ***\n"
    )
    raw = text.encode()
    (directory / "raw/tiny.txt").write_bytes(raw)
    (directory / "sources.json").write_text(
        json.dumps(
            {"books": [{"id": "tiny", "chapters": 3, "sha256": hashlib.sha256(raw).hexdigest()}]}
        )
    )
    return directory


def small_config():
    return Config(
        seeds=(3,),
        vocab_size=272,
        context_length=8,
        width=8,
        layers=1,
        heads=2,
        batch_size=2,
        steps=5,
        warmup_steps=1,
        eval_interval=2,
        device="cpu",
        probe_repeats=1,
        benchmark_repeats=1,
        generation_tokens=4,
    )


def test_full_pretrain_artifacts_checkpoint_and_token_metrics(tmp_path):
    directory = source(tmp_path)
    path, results = execute(small_config(), tmp_path / "runs", root=tmp_path, corpus_dir=directory)
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert results["3"]["training"]["steps"] == 5
    history = json.loads((path / "histories.json").read_text())["3"]
    assert (
        results["3"]["training"]["selected_step"]
        == min(history, key=lambda row: row["valid_nll"])["step"]
    )
    tokenizer = ByteBPE.load(path / "tokenizer.json")
    docs, _ = prepare_corpus(directory)
    corpus = TokenCorpus(docs, tokenizer)
    model = load_weights(path / "model-3.npz")
    metrics, _, losses = evaluate(model, corpus, "test", batch_size=2)
    assert metrics == results["3"]["selected_test"]
    archive = dict(np.load(path / "evaluation.npz", allow_pickle=False))
    np.testing.assert_array_equal(archive["3__test_nll"], losses)
    evaluation = json.loads((path / "evaluation.json").read_text())
    for name, value in archive.items():
        assert array_digest(value) == evaluation["arrays"][name]
    benchmarks = json.loads((path / "benchmarks.json").read_text())["3"]
    for name in ("within_window", "window_rollover"):
        assert benchmarks[name]["naive"]["token_ids"] == benchmarks[name]["cached"]["token_ids"]
    assert benchmarks["window_rollover"]["cached"]["metadata"]["cache_rebuilds"] > 0


def test_cpu_seeded_training_repeats_and_original_global_mode_returns(tmp_path):
    directory = source(tmp_path)
    config = small_config()
    before = torch.are_deterministic_algorithms_enabled()
    threads = torch.get_num_threads()
    a, first = execute(config, tmp_path / "runs", root=tmp_path, corpus_dir=directory)
    b, second = execute(config, tmp_path / "runs", root=tmp_path, corpus_dir=directory)
    assert (
        torch.are_deterministic_algorithms_enabled() == before
        and torch.get_num_threads() == threads
    )
    assert first["3"]["selected_test"] == second["3"]["selected_test"]
    assert json.loads((a / "model-3.json").read_text()) == json.loads(
        (b / "model-3.json").read_text()
    )


@pytest.mark.parametrize(
    "options",
    [{"seeds": []}, {"steps": 0}, {"warmup_steps": 1000}, {"device": "cuda"}, {"vocab_size": 200}],
)
def test_pretrain_config_boundaries(options):
    with pytest.raises(ValueError):
        Config(**options)
