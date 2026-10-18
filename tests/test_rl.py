import numpy as np
import pytest

from kaleido.core import Tensor
from kaleido.rl import (
    LineWorld,
    evaluate,
    generalized_advantage,
    ppo_loss,
    train_policy,
    train_q_learning,
)


def test_termination_truncation_and_gae_boundaries():
    env = LineWorld(max_steps=1)
    assert env.step(1) == (1, -0.02, False, True)
    with pytest.raises(ValueError):
        env.step(1)
    # Terminal ignores next value; truncation bootstraps but stops the following episode's GAE.
    advantage, returns = generalized_advantage(
        np.array([1.0, 2.0, 100.0]),
        np.array([0.5, 0.6, 0.7]),
        np.array([999.0, 3.0, 999.0]),
        np.array([True, False, True]),
        np.array([True, True, True]),
        gamma=0.9,
        lam=0.8,
    )
    np.testing.assert_allclose(advantage, [0.5, 4.1, 99.3])
    np.testing.assert_allclose(returns, [1, 4.7, 100])
    a, _ = generalized_advantage(
        np.array([1.0, 2.0]),
        np.zeros(2),
        np.zeros(2),
        np.array([False, True]),
        np.array([False, True]),
        gamma=0.9,
        lam=1.0,
    )
    np.testing.assert_allclose(a, [2.8, 2.0])


def test_ppo_clipping_and_detached_rollout_matches_torch():
    torch = pytest.importorskip("torch")
    data = np.array([[0.4, -0.1], [0.2, 0.8], [-0.3, 0.4], [0.9, 0.1]])
    actions = np.array([0, 1, 0, 1])
    adv = np.array([1.0, 1.0, -1.0, -1.0])
    returns = np.array([0.3, 0.2, 0.8, -0.4])
    expected_ratio = np.array([1.5, 0.5, 1.5, 0.5])
    logp = data - np.logaddexp.reduce(data, axis=1)[:, None]
    old = logp[np.arange(4), actions] - np.log(expected_ratio)
    logits, values = Tensor(data, True), Tensor(np.zeros(4), True)
    loss, diag = ppo_loss(logits, values, actions, old, adv, returns)
    loss.backward()
    tx, tv = (
        torch.tensor(data, requires_grad=True),
        torch.zeros(4, dtype=torch.float64, requires_grad=True),
    )
    tlp = tx.log_softmax(1)
    ratio = (tlp[torch.arange(4), torch.tensor(actions)] - torch.tensor(old)).exp()
    policy = -torch.minimum(
        ratio * torch.tensor(adv), ratio.clamp(0.8, 1.2) * torch.tensor(adv)
    ).mean()
    entropy = -(tx.softmax(1) * tlp).sum(1).mean()
    expected = policy + 0.5 * 0.5 * ((tv - torch.tensor(returns)) ** 2).mean() - 0.01 * entropy
    expected.backward()
    np.testing.assert_allclose(loss.data, expected.detach().numpy(), atol=1e-12)
    np.testing.assert_allclose(logits.grad, tx.grad.numpy(), atol=1e-12)
    np.testing.assert_allclose(values.grad, tv.grad.numpy(), atol=1e-12)
    assert diag["clip_fraction"] == 1


def test_q_learning_beats_random_and_bootstraps_toward_goal():
    q, _ = train_q_learning(seed=4, episodes=400)
    result = evaluate(lambda s: np.eye(2)[q[s].argmax()], seed=100, episodes=30)
    assert result["success_rate"] == 1
    assert result["mean_return"] == pytest.approx(0.94)
    assert np.all(q[:4, 1] > q[:4, 0])


@pytest.mark.parametrize("method", ["reinforce", "ppo"])
def test_on_policy_training_and_reproducibility(method):
    _, p, history = train_policy(seed=4, method=method, updates=35, episodes_per_update=8)
    result = evaluate(lambda s: p[s], seed=100, episodes=100, greedy=True)
    assert result["success_rate"] == 1
    assert np.isfinite([h["mean_rollout_return"] for h in history]).all()


def test_invalid_rollout_and_policy_are_rejected():
    with pytest.raises(ValueError):
        generalized_advantage(
            np.ones(2), np.zeros(2), np.zeros(2), np.array([True, False]), np.array([False, True])
        )
    with pytest.raises(ValueError):
        evaluate(lambda s: [0.2, 0.2], seed=1, episodes=1)
