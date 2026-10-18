import numpy as np
import pytest

from kaleido.classical import DecisionTreeClassifier, LogisticRegression


def test_xor_zero_gain_root_and_linear_boundary_failure():
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    tree = DecisionTreeClassifier(max_depth=2).fit(x, y)
    np.testing.assert_array_equal(tree.predict(x), y)
    assert tree.depth_ == 2
    assert tree.n_leaves_ == 4
    linear = LogisticRegression().fit(x, y)
    assert np.mean(linear.predict(x) == y) == 0.5
    constrained = DecisionTreeClassifier(max_depth=2, min_impurity_decrease=0.01).fit(x, y)
    assert constrained.n_leaves_ == 1


def test_unique_split_probabilities_and_sklearn():
    reference = pytest.importorskip("sklearn.tree")
    x = np.arange(12, dtype=float)[:, None]
    y = np.array(["a"] * 4 + ["b"] * 4 + ["c"] * 4)
    ours = DecisionTreeClassifier(max_depth=3).fit(x, y)
    sklearn = reference.DecisionTreeClassifier(max_depth=3, random_state=0).fit(x, y)
    query = np.linspace(-1, 13, 101)[:, None]
    np.testing.assert_array_equal(ours.predict(query), sklearn.predict(query))
    np.testing.assert_allclose(ours.predict_proba(query), sklearn.predict_proba(query))
    np.testing.assert_array_equal(ours.feature_importances_, [1])


def test_constant_features_pure_class_and_tie_order():
    model = DecisionTreeClassifier().fit(np.ones((4, 2)), ["b", "a", "b", "a"])
    assert model.depth_ == 0
    assert model.predict([[8, 9]])[0] == "a"
    np.testing.assert_allclose(model.predict_proba([[8, 9]]), [[0.5, 0.5]])
    np.testing.assert_array_equal(model.feature_importances_, [0, 0])
    pure = DecisionTreeClassifier().fit([[1], [2]], [4, 4])
    assert pure.predict([[3]])[0] == 4


def test_min_samples_leaf_and_weighted_gain_gate():
    x = np.arange(10)[:, None]
    y = np.array([0] * 4 + [1] * 5 + [0])
    model = DecisionTreeClassifier(min_samples_leaf=3).fit(x, y)

    def inspect(node):
        if node.feature < 0:
            assert node.samples >= 3
        else:
            inspect(node.left)
            inspect(node.right)

    inspect(model.root_)
    reference = pytest.importorskip("sklearn.tree")
    ours = DecisionTreeClassifier(min_impurity_decrease=0.05).fit(x, y)
    sklearn = reference.DecisionTreeClassifier(min_impurity_decrease=0.05).fit(x, y)
    np.testing.assert_array_equal(ours.predict(x), sklearn.predict(x))


def test_adjacent_float_threshold_routes_as_trained():
    low = 1.0
    high = np.nextafter(low, np.inf)
    model = DecisionTreeClassifier().fit([[low], [high]], [0, 1])
    np.testing.assert_array_equal(model.predict([[low], [high]]), [0, 1])


def test_refit_and_fitted_state():
    model = DecisionTreeClassifier()
    with pytest.raises(RuntimeError):
        model.predict([[0]])
    assert model.fit([[0], [1]], [0, 1]) is model
    model.fit([[0, 0], [1, 1]], ["x", "y"])
    with pytest.raises(ValueError, match="count"):
        model.predict([[1]])
    with pytest.raises(ValueError):
        model.fit([[0], [1]], [0])


@pytest.mark.parametrize(
    "kwargs",
    [{"max_depth": 0}, {"max_depth": 65}, {"min_samples_leaf": 0}, {"min_impurity_decrease": -1}],
)
def test_parameter_boundaries(kwargs):
    with pytest.raises(ValueError):
        DecisionTreeClassifier(**kwargs)
