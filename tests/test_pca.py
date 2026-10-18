import numpy as np
import pytest

from kaleido.classical import PCA


def test_pca_projection_and_variance_against_sklearn():
    decomposition = pytest.importorskip("sklearn.decomposition")
    rng = np.random.default_rng(14)
    x = rng.normal(size=(60, 5)) * [1, 2, 3, 4, 5] + 100
    ours, reference = PCA(3).fit(x), decomposition.PCA(3, svd_solver="full").fit(x)
    np.testing.assert_allclose(ours.explained_variance_, reference.explained_variance_, atol=1e-12)
    np.testing.assert_allclose(
        ours.components_.T @ ours.components_,
        reference.components_.T @ reference.components_,
        atol=1e-12,
    )
    reconstruction = ours.inverse_transform(ours.transform(x))
    expected = reference.inverse_transform(reference.transform(x))
    np.testing.assert_allclose(reconstruction, expected, atol=1e-12)
    np.testing.assert_allclose(ours.components_ @ ours.components_.T, np.eye(3), atol=1e-12)


def test_full_reconstruction_and_training_mean_only():
    x = np.array([[1.0, 0], [2, 1], [3, 0]])
    model = PCA(2).fit(x)
    query = np.array([[100.0, 50]])
    np.testing.assert_allclose(model.inverse_transform(model.transform(query)), query)
    np.testing.assert_allclose(model.mean_, x.mean(axis=0))
    np.testing.assert_allclose(model.explained_variance_ratio_.sum(), 1)


def test_constant_data_zero_variance_is_explicit():
    model = PCA(1).fit(np.ones((5, 3)))
    np.testing.assert_array_equal(model.explained_variance_, [0])
    np.testing.assert_array_equal(model.explained_variance_ratio_, [0])
    np.testing.assert_array_equal(model.transform([[1, 1, 1]]), [[0]])


def test_pca_boundaries_and_refit():
    with pytest.raises(ValueError):
        PCA(0)
    model = PCA(2)
    with pytest.raises(RuntimeError):
        model.inverse_transform([[1, 2]])
    with pytest.raises(ValueError):
        model.fit([[1, 2]])
    with pytest.raises(ValueError):
        PCA(3).fit(np.ones((2, 5)))
    model.fit(np.eye(3))
    with pytest.raises(ValueError):
        model.transform([[1, 2]])
    with pytest.raises(ValueError):
        model.inverse_transform([[1]])
    assert model.fit(np.eye(4)) is model
    assert model.transform([[1, 2, 3, 4]]).shape == (1, 2)
