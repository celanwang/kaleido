import numpy as np
import pytest

from kaleido.core import Tensor
from kaleido.core.gradcheck import gradcheck


@pytest.mark.parametrize(
    ("function", "reference", "values"),
    [
        (lambda x: (x + 2).sum(), lambda x: (x + 2).sum(), [[-0.7, 1.2]]),
        (lambda x: (3 - x).sum(), lambda x: (3 - x).sum(), [[-0.7, 1.2]]),
        (lambda x: (x * x + x).sum(), lambda x: (x * x + x).sum(), [[-0.7, 1.2]]),
        (lambda x: (2 / x).sum(), lambda x: (2 / x).sum(), [[0.7, 1.2]]),
        (lambda x: (x**3).sum(), lambda x: (x**3).sum(), [[-0.7, 1.2]]),
        (lambda x: (x**0.5).sum(), lambda x: np.sqrt(x).sum(), [[0.7, 1.2]]),
        (lambda x: x.exp().sum(), lambda x: np.exp(x).sum(), [[-0.7, 1.2]]),
        (lambda x: x.log().sum(), lambda x: np.log(x).sum(), [[0.7, 1.2]]),
        (lambda x: x.relu().sum(), lambda x: np.maximum(x, 0).sum(), [[-0.7, 1.2]]),
        (lambda x: x.reshape(2, 1).T.sum(), lambda x: x.reshape(2, 1).T.sum(), [[0.7, 1.2]]),
    ],
)
def test_forward_numpy_and_coordinate_gradients(function, reference, values):
    values = np.array(values)
    np.testing.assert_allclose(function(Tensor(values)).data, reference(values))
    assert gradcheck(function, [values]) < 1e-7


@pytest.mark.parametrize("operation", ["add", "multiply", "divide"])
@pytest.mark.parametrize("shapes", [((2, 1, 3), (1, 4, 1)), ((2, 3), (3,)), ((2, 3), ())])
def test_broadcast_gradients(operation, shapes):
    rng = np.random.default_rng(4)
    a, b = [rng.uniform(0.5, 1.5, size=shape) for shape in shapes]

    def function(x, y):
        result = {"add": lambda: x + y, "multiply": lambda: x * y, "divide": lambda: x / y}[
            operation
        ]()
        return (result**2).mean()

    assert gradcheck(function, [a, b]) < 1e-7


@pytest.mark.parametrize("axis", [None, 0, 1, -1, (0, 2), (-1, 0), ()])
@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_reductions(axis, keepdims, reduction):
    value = np.arange(24).reshape(2, 3, 4) / 20 + 0.2

    def function(x):
        return (getattr(x, reduction)(axis=axis, keepdims=keepdims) ** 2).sum()

    expected = (getattr(value, reduction)(axis=axis, keepdims=keepdims) ** 2).sum()
    np.testing.assert_allclose(function(Tensor(value)).data, expected)
    assert gradcheck(function, [value]) < 1e-6


def test_matmul_forward_and_both_gradients():
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=(3, 2)), rng.normal(size=(2, 4))
    np.testing.assert_allclose((Tensor(a) @ Tensor(b)).data, a @ b)
    assert gradcheck(lambda x, y: ((x @ y) ** 2).mean(), [a, b]) < 1e-7


def test_linear_regression_gradient_matches_analytic_matrix_calculus():
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=(7, 3)), rng.normal(size=(7, 1))
    w, b = Tensor(rng.normal(size=(3, 1)), True), Tensor([0.3], True)
    residual = x @ w.data + b.data - y
    loss = ((Tensor(x) @ w + b - y) ** 2).mean()
    loss.backward()
    np.testing.assert_allclose(loss.data, np.mean(residual**2))
    np.testing.assert_allclose(w.grad, 2 * x.T @ residual / len(x))
    np.testing.assert_allclose(b.grad, [2 * residual.mean()])


def test_two_layer_network_chain_rule_against_numpy():
    rng = np.random.default_rng(11)
    x, target = rng.normal(size=(5, 2)), rng.normal(size=(5, 1))
    a, b = rng.normal(size=(2, 3)), rng.normal(size=(3, 1))
    wa, wb = Tensor(a, True), Tensor(b, True)
    ((Tensor(x) @ wa).relu() @ wb - target).mean().backward()
    hidden = np.maximum(x @ a, 0)
    np.testing.assert_allclose(wb.grad, hidden.T @ (np.ones((5, 1)) / 5))
    np.testing.assert_allclose(wa.grad, x.T @ ((np.ones((5, 1)) @ b.T / 5) * (x @ a > 0)))


def test_shared_graph_repeated_backward_and_reset():
    x = Tensor(2.0, True)
    shared = x * x
    y = shared * shared + shared
    y.backward()
    assert x.grad == 36
    y.backward()
    assert x.grad == 72
    x.zero_grad()
    y.backward()
    assert x.grad == 36


def test_multiple_outputs_accumulate_only_new_adjoints():
    x = Tensor(2.0, True)
    common = x * x
    (common * 3).backward()
    (common * 5).backward()
    assert x.grad == 32


def test_vector_jacobian_seed_and_constant_pruning():
    x, constant = Tensor([2.0, 3.0], True), Tensor([4.0, 5.0])
    y = x * constant
    with pytest.raises(ValueError, match="explicit"):
        y.backward()
    y.backward([2, -1])
    np.testing.assert_array_equal(x.grad, [8, -5])
    assert constant.grad is None
    assert not x.detach().requires_grad
    assert not (constant + 1).requires_grad


def test_values_are_copied_and_read_only():
    source = np.array([1.0, 2.0])
    tensor = Tensor(source, True)
    source[0] = 100
    np.testing.assert_array_equal(tensor.data, [1, 2])
    with pytest.raises(ValueError):
        tensor.data[0] = 8
    with pytest.raises(ValueError):
        tensor.data.flags.writeable = True


def test_deep_graph_does_not_use_recursive_traversal():
    x = Tensor(1.0, True)
    y = x
    for _ in range(3000):
        y = y + 1
    y.backward()
    assert x.grad == 1


def test_numpy_left_hand_operators():
    x = Tensor([[1.0], [2.0]], True)
    result = (np.array([[3.0, 4.0]]) @ x + np.array([[2.0]])).sum()
    result.backward()
    assert result.data == 13
    np.testing.assert_array_equal(x.grad, [[3], [4]])


def test_zero_power_and_relu_subgradient():
    x = Tensor(0.0, True)
    (x**0).backward()
    assert x.grad == 0
    x.zero_grad()
    x.relu().backward()
    assert x.grad == 0


@pytest.mark.parametrize("value", [float("inf"), float("nan"), [1, float("nan")]])
def test_rejects_nonfinite_values(value):
    with pytest.raises(ValueError, match="finite"):
        Tensor(value)


def test_unsupported_inputs_and_invalid_domains():
    with pytest.raises(TypeError):
        Tensor(1 + 2j)
    with pytest.raises(ValueError, match="rank-2"):
        Tensor([1, 2]) @ Tensor([3, 4])
    with pytest.raises(ValueError, match="positive"):
        Tensor(0).log()
    with pytest.raises(FloatingPointError):
        Tensor(0) ** -1
    with pytest.raises(TypeError):
        Tensor(2) ** Tensor(3)
    with pytest.raises(ValueError, match="empty"):
        Tensor([]).mean()
    with pytest.raises(ValueError):
        Tensor(2).backward()
    for seed in ([1], float("inf")):
        with pytest.raises(ValueError, match="gradient"):
            Tensor(2, True).backward(seed)
    with pytest.raises(ValueError):
        Tensor(np.ones((2, 3))).sum(axis=(0, 0))


def test_gradcheck_detects_disconnected_gradient():
    with pytest.raises(AssertionError):
        gradcheck(lambda x: Tensor(x.data**2, True), [2.0])
    with pytest.raises(ValueError, match="scalar"):
        gradcheck(lambda x: x, [[1, 2]])
    with pytest.raises(ValueError):
        gradcheck(lambda x: x, [2.0], eps=0)
