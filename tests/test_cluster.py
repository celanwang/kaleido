import numpy as np
import pytest

from kaleido.classical import GaussianMixture, KMeans


def blobs():
    rng = np.random.default_rng(21)
    return np.r_[
        rng.normal([-4, 0], [0.5, 1], size=(80, 2)), rng.normal([4, 1], [0.7, 0.4], size=(100, 2))
    ]


def test_kmeans_identical_initial_centers_match_sklearn_and_objective():
    cluster = pytest.importorskip("sklearn.cluster")
    x = blobs()
    init = np.array([[-4, 0], [4, 1]])
    ours = KMeans(2, init=init, n_init=1).fit(x)
    reference = cluster.KMeans(2, init=init, n_init=1, tol=1e-10).fit(x)
    np.testing.assert_allclose(ours.cluster_centers_, reference.cluster_centers_, atol=1e-12)
    np.testing.assert_array_equal(ours.predict(x), ours.labels_)
    np.testing.assert_array_equal(ours.labels_, reference.labels_)
    assert ours.inertia_ == pytest.approx(reference.inertia_)
    assert np.all(np.diff(ours.inertia_history_) <= 1e-10)
    for label in range(2):
        np.testing.assert_allclose(
            ours.cluster_centers_[label], x[ours.labels_ == label].mean(axis=0)
        )


def test_kmeans_empty_clusters_are_repaired_and_seed_is_repeatable():
    x = np.array([[0.0], [1], [2], [10], [11], [12]])
    model = KMeans(3, n_init=1, init=[[0], [0], [0]]).fit(x)
    assert len(np.unique(model.labels_)) == 3
    assert np.isfinite(model.inertia_)
    a, b = KMeans(2, seed=8).fit(x), KMeans(2, seed=8).fit(x)
    np.testing.assert_array_equal(a.cluster_centers_, b.cluster_centers_)
    assert a.inertia_ == b.inertia_
    np.testing.assert_allclose(
        a.transform(x).min(axis=1) ** 2, ((x - a.cluster_centers_[a.labels_]) ** 2).sum(axis=1)
    )


def test_mixture_responsibilities_likelihood_and_sklearn_from_same_initialization():
    mixture = pytest.importorskip("sklearn.mixture")
    x = blobs()
    ours = GaussianMixture(2, n_init=1, means_init=[[-4, 0], [4, 1]], tol=1e-9, reg_covar=1e-6).fit(
        x
    )
    initial = ours.initialization_
    reference = mixture.GaussianMixture(
        2,
        covariance_type="diag",
        n_init=1,
        tol=1e-9,
        reg_covar=1e-6,
        weights_init=initial["weights"],
        means_init=initial["means"],
        precisions_init=1 / initial["variances"],
        max_iter=200,
    ).fit(x)
    assert ours.converged_
    assert np.all(np.diff(ours.lower_bounds_) >= -1e-8)
    np.testing.assert_allclose(ours.weights_.sum(), 1)
    np.testing.assert_allclose(ours.predict_proba(x).sum(axis=1), 1, atol=1e-12)
    assert np.all(ours.covariances_ >= ours.reg_covar)
    np.testing.assert_allclose(ours.means_, reference.means_, atol=1e-6)
    np.testing.assert_allclose(ours.score_samples(x), reference.score_samples(x), atol=1e-6)
    np.testing.assert_allclose(ours.predict_proba(x), reference.predict_proba(x), atol=1e-6)


def test_mixture_covariance_floor_and_fixed_seed():
    x = np.array([[0.0, 1], [0, 1], [5, 1], [5, 1]])
    a, b = GaussianMixture(2, seed=4).fit(x), GaussianMixture(2, seed=4).fit(x)
    np.testing.assert_array_equal(a.means_, b.means_)
    np.testing.assert_allclose(a.covariances_[:, 1], a.reg_covar)
    assert np.isfinite(a.score_samples([[100, 100]])).all()


def test_all_collapsed_restarts_are_rejected():
    with pytest.raises(RuntimeError, match="collapsed"):
        GaussianMixture(2, n_init=1, means_init=[[0], [1e8]]).fit([[0], [1], [2]])


@pytest.mark.parametrize("model", [KMeans, GaussianMixture])
def test_unfitted_impossible_clusters_shapes_and_iterations(model):
    with pytest.raises(RuntimeError):
        model(2).predict([[1]])
    with pytest.raises(ValueError, match="distinct"):
        model(3).fit([[1], [1], [2]])
    with pytest.warns(RuntimeWarning):
        estimator = model(2, max_iter=1, tol=1e-15).fit(blobs())
    assert not estimator.converged_
    with pytest.raises(ValueError, match="count"):
        estimator.predict([[1]])


@pytest.mark.parametrize("kwargs", [{"n_init": 0}, {"seed": -1}, {"tol": -1}])
@pytest.mark.parametrize("model", [KMeans, GaussianMixture])
def test_cluster_parameter_validation(model, kwargs):
    with pytest.raises(ValueError):
        model(2, **kwargs)
