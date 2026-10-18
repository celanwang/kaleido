import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kaleido.lm.model import DecoderConfig, DecoderLM  # noqa:E402
from kaleido.lm.tokenizer import EOS, ByteBPE  # noqa:E402
from kaleido.lm.training import (  # noqa:E402
    TokenCorpus,
    count_baselines,
    evaluate,
    load_weights,
    metrics_from_losses,
    save_weights,
)


def corpus():
    docs = [
        {"id": s, "split": s, "text": text}
        for s, text in zip(("train", "valid", "test"), ("abababab", "ac", "ababa"), strict=True)
    ]
    return TokenCorpus(docs, ByteBPE())


def test_document_windows_shift_once_include_eos_and_ignore_padding():
    data = corpus()
    rows = list(data.windows("valid", 8))
    assert len(rows) == 1
    _, offset, x, y = rows[0]
    assert offset == 0
    np.testing.assert_array_equal(x[:3], [EOS, 97, 99])
    np.testing.assert_array_equal(y[:3], [97, 99, EOS])
    assert np.all(y[3:] == -100)
    x, y = data.sample(3, 20, np.random.default_rng(3))
    assert np.all(x[:, 0] == EOS) and np.all(y[:, 8] == EOS)
    assert np.all(y[:, 9:] == -100)


def test_evaluation_weights_every_token_and_restores_training_mode():
    torch.manual_seed(4)
    model = DecoderLM(
        DecoderConfig(vocab_size=257, context_length=3, width=8, layers=1, heads=2)
    ).train()
    result, detail, losses = evaluate(model, corpus(), "test", batch_size=2)
    assert model.training
    assert len(losses) == 6 and sum(w["target_tokens"] for w in detail["windows"]) == 6
    assert result["nll"] == pytest.approx(losses.mean())
    assert result["perplexity"] == pytest.approx(np.exp(losses.mean()))
    assert result["nll_sum"] == pytest.approx(sum(w["nll_sum"] for w in detail["windows"]))
    independent = metrics_from_losses(np.array([1.0, 1.0, 4.0]), 5)
    assert independent["nll"] == 2


def test_count_baselines_train_only_and_checkpoint_validation(tmp_path):
    data = corpus()
    baseline = count_baselines(data, "valid")
    assert baseline["unigram"]["target_tokens"] == 3
    torch.manual_seed(5)
    model = DecoderLM(
        DecoderConfig(vocab_size=257, context_length=8, width=8, layers=1, heads=2)
    ).eval()
    path = tmp_path / "model.npz"
    save_weights(path, model)
    restored = load_weights(path)
    x = torch.tensor([[EOS, 97, 98]])
    torch.testing.assert_close(model(x)[0], restored(x)[0], atol=0, rtol=0)
    arrays = dict(np.load(path, allow_pickle=False))
    name = next(iter(arrays))
    arrays[name] = arrays[name] + 1
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="digest"):
        load_weights(path)
