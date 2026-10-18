import numpy as np
import pytest

from kaleido.classical import LinearRegression, LogisticRegression, StandardScaler
from kaleido.classical.linear import logistic_objective
from kaleido.core import Tensor
from kaleido.core.gradcheck import gradcheck
from kaleido.core.losses import binary_cross_entropy_with_logits


def test_stable_binary_loss_and_extreme_logit_gradients():
    logits = Tensor([-1000.0, 1000], True)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        loss = binary_cross_entropy_with_logits(logits, [1, 0])
        loss.backward()
    assert loss.data == 1000
    np.testing.assert_array_equal(logits.grad, [-0.5, 0.5])
    assert binary_cross_entropy_with_logits(logits, [0, 1]).data == 0
    assert gradcheck(lambda x: x.softplus().sum(), [[-2.0, 0.0, 3.0]]) < 1e-8
    assert gradcheck(lambda x: x.sigmoid().sum(), [[-2.0, 0.0, 3.0]]) < 1e-8
    with pytest.raises(ValueError):
        binary_cross_entropy_with_logits(logits, [1])
    with pytest.raises(ValueError):
        binary_cross_entropy_with_logits(logits, [1, 2])


def test_logistic_gradient_hessian_and_autodiff_agree():
    rng = np.random.default_rng(13)
    x, y, theta = rng.normal(size=(12, 3)), rng.integers(0, 2, 12), rng.normal(size=4)
    design = np.column_stack([x, np.ones(12)])
    loss, gradient, hessian = logistic_objective(theta, design, y, 0.2)
    w, b = Tensor(theta[:-1].reshape(-1, 1), True), Tensor(theta[-1], True)
    tensor_loss = binary_cross_entropy_with_logits(Tensor(x) @ w + b, y[:, None])
    tensor_loss = tensor_loss + 0.1 * (w * w).sum()
    tensor_loss.backward()
    np.testing.assert_allclose(tensor_loss.data, loss)
    np.testing.assert_allclose(np.r_[w.grad.ravel(), b.grad], gradient)
    numeric_hessian = np.zeros_like(hessian)
    for index in range(len(theta)):
        offset = np.eye(len(theta))[index] * 1e-6
        plus = logistic_objective(theta + offset, design, y, 0.2)[1]
        minus = logistic_objective(theta - offset, design, y, 0.2)[1]
        numeric_hessian[:, index] = (plus - minus) / 2e-6
    np.testing.assert_allclose(hessian, numeric_hessian, atol=1e-9)


def test_scaler_training_only_constants_and_inverse():
    x = np.array([[1.0, 7], [3, 7]])
    scaler = StandardScaler().fit(x)
    np.testing.assert_array_equal(scaler.transform([[101, 9]]), [[99, 2]])
    np.testing.assert_allclose(scaler.inverse_transform(scaler.transform(x)), x)
    np.testing.assert_array_equal(scaler.scale_, [1, 1])


def test_linear_rank_deficiency_and_refit():
    x = np.array([[0, 0], [1, 1], [2, 2]])
    estimator = LinearRegression().fit(x, [1, 5, 9])
    assert estimator.rank_ == 1
    np.testing.assert_allclose(estimator.coef_, [2, 2])
    np.testing.assert_allclose(estimator.predict(x), [1, 5, 9])
    assert estimator.fit([[0], [1]], [1, 3]) is estimator
    assert estimator.predict([[2]])[0] == pytest.approx(5)


def test_linear_and_scaler_against_sklearn():
    linear = pytest.importorskip("sklearn.linear_model")
    preprocessing = pytest.importorskip("sklearn.preprocessing")
    rng = np.random.default_rng(8)
    x, y = rng.normal(size=(100, 5)), rng.normal(size=100)
    ours, reference = LinearRegression().fit(x, y), linear.LinearRegression().fit(x, y)
    np.testing.assert_allclose(ours.predict(x), reference.predict(x), atol=1e-12)
    np.testing.assert_allclose(
        StandardScaler().fit_transform(x),
        preprocessing.StandardScaler().fit_transform(x),
        atol=1e-12,
    )


def test_logistic_sklearn_objective_and_string_labels():
    linear = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(19)
    x = rng.normal(size=(150, 3))
    y = np.where(x[:, 0] + rng.normal(size=150) > 0, "positive", "negative")
    ours = LogisticRegression(l2=0.03).fit(x, y)
    reference = linear.LogisticRegression(C=1 / (len(x) * 0.03), tol=1e-11, max_iter=1000).fit(x, y)
    assert ours.converged_
    assert np.all(np.diff(ours.loss_history_) <= 1e-12)
    np.testing.assert_allclose(ours.coef_, reference.coef_[0], atol=1e-6)
    np.testing.assert_allclose(ours.predict_proba(x), reference.predict_proba(x), atol=1e-6)
    np.testing.assert_array_equal(ours.predict(x), reference.predict(x))


def test_nonconvergence_is_explicit_and_one_class_rejected():
    x = np.arange(10.0)[:, None]
    with pytest.warns(RuntimeWarning, match="tolerance"):
        model = LogisticRegression(max_iter=1, tol=1e-15).fit(x, np.arange(10) % 2)
    assert not model.converged_
    with pytest.raises(ValueError, match="two classes"):
        LogisticRegression().fit(x, np.zeros(10))


@pytest.mark.parametrize(
    "model,method",
    [
        (LinearRegression, "predict"),
        (LogisticRegression, "predict"),
        (StandardScaler, "transform"),
    ],
)
def test_unfitted_and_feature_count(model, method):
    estimator = model()
    with pytest.raises(RuntimeError, match="fitted"):
        getattr(estimator, method)([[1]])
    if model is StandardScaler:
        estimator.fit([[0], [1]])
    else:
        estimator.fit([[0], [1]], [0, 1])
    with pytest.raises(ValueError, match="count"):
        getattr(estimator, method)([[1, 2]])


@pytest.mark.parametrize("x", [[], [[float("nan")]], [[1 + 2j]], [["a"]], [1, 2]])
def test_feature_validation(x):
    with pytest.raises(ValueError):
        StandardScaler().fit(x)


@pytest.mark.parametrize("options", [{"l2": -1}, {"tol": 0}, {"max_iter": True}])
def test_logistic_parameter_validation(options):
    with pytest.raises(ValueError):
        LogisticRegression(**options)
