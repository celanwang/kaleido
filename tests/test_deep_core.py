import numpy as np
import pytest

from kaleido.core import Tensor
from kaleido.core.gradcheck import gradcheck
from kaleido.core.losses import cross_entropy
from kaleido.core.nn import MLP, Linear, Parameter
from kaleido.core.optim import SGD, Adam


@pytest.mark.parametrize(
    "function",
    [
        lambda x: x.tanh().sum(),
        lambda x: x.logsumexp(1).sum(),
        lambda x: (x.softmax(1) ** 2).sum(),
        lambda x: (x[[0, 0, 1], [1, 1, 0]] ** 2).sum(),
    ],
)
def test_new_vjps(function):
    assert gradcheck(function, [np.array([[0.2, -0.4], [1.2, 0.5]])]) < 1e-7


def test_stable_ce_masks_and_index_snapshot():
    x = Tensor([[10000.0, -10000], [9999, 10000]], True)
    loss = cross_entropy(x, [1, 0])
    loss.backward()
    assert np.isfinite(loss.data)
    np.testing.assert_allclose(x.grad.sum(axis=1), 0, atol=1e-12)
    values = Tensor([[2.0, 3.0], [4.0, 5.0]], True)
    mask = np.tril(np.ones((2, 2), bool))
    out = values.softmax(1, mask=mask)
    np.testing.assert_array_equal(out.data[0], [1, 0])
    with pytest.raises(ValueError):
        values.softmax(1, mask=np.zeros((2, 2), bool))
    index = np.array([0, 0])
    selected = values[index]
    index[:] = 1
    selected.sum().backward()
    np.testing.assert_array_equal(values.grad, [[2, 2], [0, 0]])
    for y in ([0.0, 1.0], [-1, 0], [0]):
        with pytest.raises(ValueError):
            cross_entropy(x, y)


def test_immutable_parameter_graph_and_atomic_state_load():
    layer = Linear(2, 1, rng=np.random.default_rng(2))
    x = Tensor([[1.0, 2.0]], True)
    y = layer(x).sum()
    old = layer.weight.data.copy()
    layer.weight.replace(np.ones((2, 1)) * 100)
    y.backward()
    np.testing.assert_allclose(x.grad, old.T)
    state = layer.state_dict()
    state["bias"] = np.zeros(3)
    before = layer.state_dict()
    with pytest.raises(ValueError):
        layer.load_state_dict(state)
    for key, value in before.items():
        np.testing.assert_array_equal(layer.state_dict()[key], value)


@pytest.mark.parametrize("optimizer", [SGD, Adam])
def test_optimizer_transaction_skip_and_old_snapshot(optimizer):
    a, b = Parameter([1.0]), Parameter([2.0])
    opt = optimizer([a, b], lr=0.1)
    a.tensor.grad = np.array([1.0])
    b.tensor.grad = np.array([np.nan])
    with pytest.raises(ValueError):
        opt.step()
    np.testing.assert_array_equal(a.data, [1.0])
    assert opt.state == {}
    b.tensor.grad = None
    old = a.tensor
    opt.step()
    np.testing.assert_array_equal(old.data, [1.0])
    assert b.grad is None
    assert a.grad is not None
    opt.zero_grad()
    assert a.grad is None


def test_mlp_forward_gradient_and_update_vs_torch():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    rng = np.random.default_rng(4)
    model = MLP([3, 5, 2], rng=rng)
    reference = torch.nn.Sequential(
        torch.nn.Linear(3, 5), torch.nn.ReLU(), torch.nn.Linear(5, 2)
    ).double()
    with torch.no_grad():
        for ours, ref in zip(model.layers, [reference[0], reference[2]], strict=True):
            ref.weight.copy_(torch.tensor(ours.weight.data.T))
            ref.bias.copy_(torch.tensor(ours.bias.data))
    x, y = rng.normal(size=(7, 3)), rng.integers(0, 2, 7)
    out = model(Tensor(x))
    expected = reference(torch.tensor(x))
    np.testing.assert_allclose(out.data, expected.detach().numpy(), atol=1e-12)
    loss = cross_entropy(out, y)
    loss.backward()
    ref_loss = torch.nn.functional.cross_entropy(expected, torch.tensor(y))
    ref_loss.backward()
    np.testing.assert_allclose(loss.data, ref_loss.detach().numpy(), atol=1e-12)
    for ours, ref in zip(model.layers, [reference[0], reference[2]], strict=True):
        np.testing.assert_allclose(ours.weight.grad, ref.weight.grad.numpy().T, atol=1e-12)
    Adam(model.parameters(), lr=0.02).step()
    torch.optim.Adam(reference.parameters(), lr=0.02).step()
    for ours, ref in zip(model.layers, [reference[0], reference[2]], strict=True):
        np.testing.assert_allclose(ours.weight.data, ref.weight.detach().numpy().T, atol=1e-12)


@pytest.mark.parametrize("kind", [SGD, Adam])
def test_multi_step_optimizers_match_torch_with_decay_and_missing_gradients(kind):
    torch = pytest.importorskip("torch")
    ours = [Parameter([1.0, -2.0]), Parameter([0.3])]
    refs = [torch.nn.Parameter(torch.tensor(p.data)) for p in ours]
    kwargs = {"lr": 0.03, "weight_decay": 0.1}
    if kind is SGD:
        kwargs["momentum"] = 0.8
    opt = kind(ours, **kwargs)
    ref = getattr(torch.optim, kind.__name__)(refs, **kwargs)
    for step in range(4):
        for i, (a, b) in enumerate(zip(ours, refs, strict=True)):
            gradient = None if i == 1 and step % 2 == 0 else np.ones_like(a.data) * (step + 0.4)
            a.tensor.grad = gradient
            b.grad = None if gradient is None else torch.tensor(gradient)
        opt.step()
        ref.step()
        for a, b in zip(ours, refs, strict=True):
            np.testing.assert_allclose(a.data, b.detach().numpy(), atol=1e-12)


@pytest.mark.parametrize("kwargs", [{"lr": 0}, {"momentum": 1}, {"weight_decay": -1}])
def test_optimizer_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        SGD([Parameter([1.0])], **kwargs)
