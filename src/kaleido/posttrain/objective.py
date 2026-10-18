"""Explicit GRPO variant: per-completion mean, population group scaling, sampled KL."""

import torch


def group_advantages(rewards, group_size, *, epsilon=1e-4):
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    if (
        rewards.ndim != 1
        or type(group_size) is not int
        or group_size < 2
        or rewards.numel() % group_size
        or not rewards.numel()
        or not torch.isfinite(rewards).all()
        or epsilon <= 0
    ):
        raise ValueError("rewards must form complete finite groups of size >=2")
    grouped = rewards.reshape(-1, group_size)
    std = grouped.std(dim=1, unbiased=False)
    advantages = (grouped - grouped.mean(dim=1, keepdim=True)) / (std[:, None] + epsilon)
    return advantages.flatten().detach(), {
        "zero_variance_group_fraction": float((std == 0).float().mean()),
        "reward_mean": float(rewards.mean()),
        "reward_std_population": float(rewards.std(unbiased=False)),
    }


def grpo_loss(
    log_probs, old_log_probs, reference_log_probs, mask, advantages, *, clip=0.2, beta=0.04
):
    if (
        log_probs.ndim != 2
        or old_log_probs.shape != log_probs.shape
        or reference_log_probs.shape != log_probs.shape
        or mask.shape != log_probs.shape
        or mask.dtype != torch.bool
        or torch.any(mask.sum(1) == 0)
        or advantages.shape != (log_probs.shape[0],)
        or not 0 < clip < 1
        or beta < 0
    ):
        raise ValueError("invalid GRPO shapes, completion mask or coefficients")
    if any(value.requires_grad for value in (old_log_probs, reference_log_probs, advantages)):
        raise ValueError("old policy, reference and advantages must be detached")
    if not all(
        torch.isfinite(value).all()
        for value in (log_probs, old_log_probs, reference_log_probs, advantages)
    ):
        raise ValueError("GRPO probabilities and advantages must be finite")
    ratio = (log_probs - old_log_probs).masked_fill(~mask, 0).exp()
    advantage = advantages[:, None]
    policy = torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
    delta = (reference_log_probs - log_probs).masked_fill(~mask, 0)
    kl = delta.exp() - delta - 1
    per_sequence = ((-policy + beta * kl) * mask).sum(1) / mask.sum(1)
    loss = per_sequence.mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("GRPO objective diverged")
    return loss, {
        "sampled_kl": float((kl.detach() * mask).sum() / mask.sum()),
        "clip_fraction": float(((ratio.detach() - 1).abs() > clip)[mask].float().mean()),
        "policy_loss": float(((-policy.detach() * mask).sum(1) / mask.sum(1)).mean()),
    }
