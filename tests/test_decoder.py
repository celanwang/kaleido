import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kaleido.lm.inference import cache_equivalence, generate  # noqa:E402
from kaleido.lm.model import CausalAttention, DecoderConfig, DecoderLM  # noqa:E402


def tiny():
    torch.manual_seed(3)
    torch.set_num_threads(1)
    return DecoderLM(
        DecoderConfig(vocab_size=32, context_length=12, width=16, layers=2, heads=4)
    ).double()


def test_attention_forward_and_gradients_match_sdpa():
    torch.manual_seed(4)
    config = DecoderConfig(vocab_size=32, width=16, layers=1, heads=4)
    module = CausalAttention(config).double()
    x = torch.randn(2, 5, 16, dtype=torch.float64, requires_grad=True)
    output, _ = module(x)
    q, k, v = module.qkv(x).chunk(3, dim=-1)
    q, k, v = [a.reshape(2, 5, 4, 4).transpose(1, 2) for a in (q, k, v)]
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    expected = module.output(expected.transpose(1, 2).reshape(2, 5, 16))
    np.testing.assert_allclose(output.detach().numpy(), expected.detach().numpy(), atol=1e-12)
    actual_grad = torch.autograd.grad(
        output.square().sum(), (x, *module.parameters()), retain_graph=True
    )
    ref_grad = torch.autograd.grad(expected.square().sum(), (x, *module.parameters()))
    for a, b in zip(actual_grad, ref_grad, strict=True):
        torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-11)


def test_decoder_causality_and_multi_token_offset_cache():
    model = tiny().eval()
    ids = torch.randint(0, 32, (2, 10))
    result = cache_equivalence(model, ids, chunk_sizes=(4, 2, 1))
    assert result["max_abs_logit_error"] < 1e-12 and result["greedy_agreement"]
    changed = ids.clone()
    changed[:, 6:] = (changed[:, 6:] + 1) % 32
    torch.testing.assert_close(model(ids)[0][:, :6], model(changed)[0][:, :6], atol=0, rtol=0)


def test_generation_cache_matches_across_context_resets_and_seeded_sampling():
    model = tiny().eval()
    prompt = [1, 2, 3, 4, 5]
    naive, _ = generate(model, prompt, max_new_tokens=20, temperature=0, use_cache=False)
    cached, metadata = generate(model, prompt, max_new_tokens=20, temperature=0, use_cache=True)
    assert naive == cached and metadata["cache_rebuilds"] > 0
    a, _ = generate(
        model, prompt, max_new_tokens=6, temperature=0.8, top_k=5, seed=7, use_cache=False
    )
    b, _ = generate(
        model, prompt, max_new_tokens=6, temperature=0.8, top_k=5, seed=7, use_cache=True
    )
    assert a == b


def test_context_cache_and_parameter_boundaries():
    model = tiny()
    x = torch.ones((1, 2), dtype=torch.long)
    with pytest.raises(ValueError):
        model(x, use_cache=True)
    model.eval()
    _, cache = model(x, use_cache=True)
    with pytest.raises(ValueError):
        model(torch.ones((1, 11), dtype=torch.long), past_key_values=cache, use_cache=True)
    with pytest.raises(ValueError):
        model(x, past_key_values=cache)
    with pytest.raises(ValueError):
        model(torch.tensor([[99]]))
    with pytest.raises(ValueError):
        DecoderConfig(width=17, heads=4)
    with pytest.raises(ValueError):
        generate(model, [], max_new_tokens=1)


def test_tiny_corpus_overfit_and_weight_roundtrip(tmp_path):
    model = tiny().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0)
    x = torch.tensor([[1, 2, 3, 4]] * 4)
    y = torch.tensor([[2, 3, 4, 1]] * 4)
    initial = float(
        torch.nn.functional.cross_entropy(model(x)[0].reshape(-1, 32), y.reshape(-1)).detach()
    )
    for _ in range(45):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x)[0].reshape(-1, 32), y.reshape(-1))
        loss.backward()
        optimizer.step()
    assert float(loss.detach()) < initial * 0.02
    state = {name: value.detach().numpy() for name, value in model.state_dict().items()}
    path = tmp_path / "weights.npz"
    np.savez_compressed(path, **state)
    restored = tiny()
    restored.load_state_dict(
        {k: torch.from_numpy(v) for k, v in dict(np.load(path, allow_pickle=False)).items()}
    )
    torch.testing.assert_close(model(x)[0], restored(x)[0], atol=0, rtol=0)
