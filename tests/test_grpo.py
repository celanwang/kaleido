import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kaleido.posttrain.objective import group_advantages, grpo_loss  # noqa:E402


def test_group_normalization_and_zero_variance_signals():
    adv, stats = group_advantages([0, 1, 0, 1, 1, 1, 1, 1], 4)
    torch.testing.assert_close(adv[:4], torch.tensor([-1, 1, -1, 1.0]) / (1 + 0.0002))
    assert torch.all(adv[4:] == 0) and stats["zero_variance_group_fraction"] == 0.5


def test_grpo_independent_formula_gradient_and_equal_sequence_weighting():
    values = np.array([[-1.0, -2.0, 1000.0], [-0.7, -1.1, -0.2]])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    old = torch.tensor([[-1.3, -1.7, 0.0], [-0.5, -1.4, -0.2]], dtype=torch.float64)
    ref = old - 0.15
    adv = torch.tensor([1.0, -1.0], dtype=torch.float64)
    lp = torch.tensor(values, requires_grad=True)
    loss, stats = grpo_loss(lp, old, ref, mask, adv, beta=0.04)
    explicit = []
    for row in range(2):
        terms = []
        for token in range(3):
            if not mask[row, token]:
                continue
            ratio = torch.exp(lp[row, token] - old[row, token])
            surrogate = torch.minimum(ratio * adv[row], torch.clamp(ratio, 0.8, 1.2) * adv[row])
            delta = ref[row, token] - lp[row, token]
            terms.append(-surrogate + 0.04 * (torch.exp(delta) - delta - 1))
        explicit.append(torch.stack(terms).mean())
    expected = torch.stack(explicit).mean()
    actual_grad = torch.autograd.grad(loss, lp, retain_graph=True)[0]
    reference_grad = torch.autograd.grad(expected, lp)[0]
    torch.testing.assert_close(loss, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(actual_grad, reference_grad, atol=1e-12, rtol=1e-12)
    assert actual_grad[0, 2] == 0 and stats["sampled_kl"] >= 0


def test_kl_penalty_zero_signal_and_detach_boundaries():
    lp = torch.tensor([[-1.0, -2.0]], requires_grad=True)
    mask = torch.ones_like(lp, dtype=torch.bool)
    loss, _ = grpo_loss(lp, lp.detach(), lp.detach(), mask, torch.zeros(1))
    loss.backward()
    torch.testing.assert_close(lp.grad, torch.zeros_like(lp))
    with pytest.raises(ValueError):
        grpo_loss(lp, lp, lp.detach(), mask, torch.ones(1))
    with pytest.raises(ValueError):
        group_advantages([1, 0, 1], 2)
