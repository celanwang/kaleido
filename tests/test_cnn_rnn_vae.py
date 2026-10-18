import numpy as np
import pytest

from kaleido.core import Tensor
from kaleido.core.gradcheck import gradcheck
from kaleido.core.models import GaussianVAE, SimpleRNN, vae_loss
from kaleido.core.ops import conv2d, stack


def test_convolution_gradcheck_and_torch():
    rng = np.random.default_rng(20)
    x, w = rng.normal(size=(2, 2, 4, 4)), rng.normal(size=(3, 2, 2, 2))
    assert gradcheck(lambda a, b: (conv2d(a, b) ** 2).mean(), [x, w]) < 1e-7
    torch = pytest.importorskip("torch")
    a, b = Tensor(x, True), Tensor(w, True)
    ta, tb = torch.tensor(x, requires_grad=True), torch.tensor(w, requires_grad=True)
    out = conv2d(a, b)
    ref = torch.nn.functional.conv2d(ta, tb)
    (out**2).mean().backward()
    (ref**2).mean().backward()
    np.testing.assert_allclose(out.data, ref.detach().numpy(), atol=1e-12)
    np.testing.assert_allclose(a.grad, ta.grad.numpy(), atol=1e-12)
    np.testing.assert_allclose(b.grad, tb.grad.numpy(), atol=1e-12)


def test_rnn_bptt_matches_torch_and_depends_on_early_inputs():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(21)
    model = SimpleRNN(2, 3, rng=rng)
    ref = torch.nn.RNN(2, 3, batch_first=True, bias=True).double()
    with torch.no_grad():
        ref.weight_ih_l0.copy_(torch.tensor(model.input.weight.data.T))
        ref.weight_hh_l0.copy_(torch.tensor(model.recurrent.weight.data.T))
        ref.bias_ih_l0.copy_(torch.tensor(model.input.bias.data))
        ref.bias_hh_l0.zero_()
    value = rng.normal(size=(2, 4, 2))
    x = Tensor(value, True)
    tx = torch.tensor(value, requires_grad=True)
    out = model(x)
    expected, _ = ref(tx)
    out[:, -1, :].sum().backward()
    expected[:, -1, :].sum().backward()
    np.testing.assert_allclose(out.data, expected.detach().numpy(), atol=1e-12)
    np.testing.assert_allclose(x.grad, tx.grad.numpy(), atol=1e-12)
    np.testing.assert_allclose(
        model.recurrent.weight.grad, ref.weight_hh_l0.grad.numpy().T, atol=1e-12
    )
    assert np.linalg.norm(x.grad[:, 0, :]) > 0


def test_vae_reparameterization_and_kl_match_torch():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(22)
    x, mean, lv, epsilon = (
        rng.uniform(size=(4, 3)),
        rng.normal(size=(4, 2)),
        rng.normal(size=(4, 2)),
        rng.normal(size=(4, 2)),
    )
    mu, logvar = Tensor(mean, True), Tensor(lv, True)
    tm, tl = torch.tensor(mean, requires_grad=True), torch.tensor(lv, requires_grad=True)
    decoder = rng.normal(size=(2, 3))
    z = mu + (logvar * 0.5).exp() * Tensor(epsilon)
    logits = z @ Tensor(decoder)
    tz = tm + torch.exp(tl * 0.5) * torch.tensor(epsilon)
    ref_logits = tz @ torch.tensor(decoder)
    loss, reconstruction, kl = vae_loss(logits, Tensor(x), mu, logvar)
    ref_rec = (
        torch.nn.functional.binary_cross_entropy_with_logits(
            ref_logits, torch.tensor(x), reduction="sum"
        )
        / 4
    )
    ref_kl = 0.5 * (tm**2 + tl.exp() - tl - 1).sum(dim=1).mean()
    loss.backward()
    (ref_rec + ref_kl).backward()
    np.testing.assert_allclose(
        [loss.data, reconstruction.data, kl.data],
        [(ref_rec + ref_kl).detach().numpy(), ref_rec.detach().numpy(), ref_kl.detach().numpy()],
        atol=1e-12,
    )
    np.testing.assert_allclose(mu.grad, tm.grad.numpy(), atol=1e-12)
    np.testing.assert_allclose(logvar.grad, tl.grad.numpy(), atol=1e-12)


def test_stack_duplicate_gradient_and_shape_boundaries():
    x = Tensor([1.0, 2.0], True)
    stack([x, x], axis=1).sum().backward()
    np.testing.assert_array_equal(x.grad, [2, 2])
    with pytest.raises(ValueError):
        stack([])
    with pytest.raises(ValueError):
        conv2d(Tensor(np.ones((1, 1, 2, 2))), Tensor(np.ones((1, 1, 3, 3))))
    with pytest.raises(ValueError):
        SimpleRNN(1, 2, rng=np.random.default_rng(1))(Tensor(np.ones((2, 0, 1))))
    with pytest.raises(ValueError):
        GaussianVAE(3, 4, 2, rng=np.random.default_rng(1))(Tensor(np.ones((2, 3))), np.ones((2, 1)))
