import numpy as np
import pytest

from kaleido.core import Tensor
from kaleido.core.attention import SelfAttention, scaled_dot_product_attention
from kaleido.core.gradcheck import gradcheck


def test_attention_all_input_gradients_finite_difference():
    rng = np.random.default_rng(10)
    inputs = [rng.normal(size=shape) for shape in ((3, 2), (4, 2), (4, 3))]
    assert (
        gradcheck(lambda q, k, v: (scaled_dot_product_attention(q, k, v)[0] ** 2).mean(), inputs)
        < 1e-7
    )


def test_causal_attention_matches_torch_forward_and_all_gradients():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(14)
    arrays = [rng.normal(size=(4, 3)) for _ in range(3)]
    ours = [Tensor(x, True) for x in arrays]
    refs = [torch.tensor(x, requires_grad=True) for x in arrays]
    mask = np.tril(np.ones((4, 4), bool))
    output, prob = scaled_dot_product_attention(*ours, mask=mask)
    expected = torch.nn.functional.scaled_dot_product_attention(*refs, is_causal=True)
    (output**2).sum().backward()
    (expected**2).sum().backward()
    np.testing.assert_allclose(output.data, expected.detach().numpy(), atol=1e-12)
    for own, ref in zip(ours, refs, strict=True):
        np.testing.assert_allclose(own.grad, ref.grad.numpy(), atol=1e-12)
    assert np.all(prob.data[~mask] == 0)


def test_future_perturbations_cannot_change_prefix_outputs_or_gradients():
    model = SelfAttention(3, rng=np.random.default_rng(15))
    x = np.random.default_rng(16).normal(size=(5, 3))
    initial = model(Tensor(x)).data.copy()
    changed = x.copy()
    changed[3:] += 100
    np.testing.assert_allclose(model(Tensor(changed)).data[:3], initial[:3], atol=1e-12)
    tx = Tensor(x, True)
    model(tx)[:3].sum().backward()
    np.testing.assert_array_equal(tx.grad[3:], np.zeros((2, 3)))


def test_attention_shape_and_fully_masked_boundaries():
    x = Tensor(np.ones((2, 3)))
    with pytest.raises(ValueError):
        scaled_dot_product_attention(x, Tensor(np.ones((2, 4))), x)
    with pytest.raises(ValueError):
        scaled_dot_product_attention(x, x, x, mask=np.zeros((2, 2), bool))
