"""Finite LineWorld: tabular Q-learning, REINFORCE, and on-policy clipped PPO."""

import numpy as np

from kaleido.core import Tensor
from kaleido.core.nn import Linear, Module
from kaleido.core.optim import Adam
from kaleido.numerics import nonnegative_float


class LineWorld:
    n_states = 5

    def __init__(self, max_steps=12):
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.state, self.steps, self.ended = 0, 0, False
        return self.state

    def step(self, action):
        if self.ended or action not in (0, 1):
            raise ValueError("reset ended episodes and use action 0 or 1")
        self.state = max(0, min(4, self.state + (1 if action else -1)))
        self.steps += 1
        terminated = self.state == 4
        truncated = self.steps >= self.max_steps and not terminated
        self.ended = terminated or truncated
        return self.state, 1.0 if terminated else -0.02, terminated, truncated


def generalized_advantage(
    rewards, values, next_values, terminated, episode_end, *, gamma=0.99, lam=0.95
):
    arrays = [np.asarray(v) for v in (rewards, values, next_values, terminated, episode_end)]
    rewards, values, next_values, terminated, episode_end = arrays
    if (
        not rewards.size
        or rewards.ndim != 1
        or any(v.shape != rewards.shape for v in arrays)
        or not all(np.isfinite(v).all() for v in arrays)
        or terminated.dtype != bool
        or episode_end.dtype != bool
        or np.any(terminated & ~episode_end)
        or not 0 <= gamma <= 1
        or not 0 <= lam <= 1
    ):
        raise ValueError("GAE requires matching finite vectors and valid episode boundaries")
    advantage = np.zeros_like(rewards, dtype=float)
    following = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        delta = rewards[t] + gamma * (not terminated[t]) * next_values[t] - values[t]
        following = delta + gamma * lam * (not episode_end[t]) * following
        advantage[t] = following
    return advantage, advantage + values


def ppo_loss(
    logits,
    values,
    actions,
    old_logp,
    advantages,
    returns,
    *,
    clip=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
):
    nonnegative_float(clip, "clip", positive=True)
    if clip >= 1:
        raise ValueError("PPO clip must be < 1")
    nonnegative_float(value_coef, "value_coef")
    nonnegative_float(entropy_coef, "entropy_coef")
    actions = np.asarray(actions)
    n = logits.shape[0] if len(logits.shape) == 2 else 0
    arrays = [np.asarray(x) for x in (old_logp, advantages, returns)]
    if (
        n == 0
        or actions.shape != (n,)
        or actions.dtype.kind not in "iu"
        or np.any(actions < 0)
        or np.any(actions >= logits.shape[1])
        or values.shape != (n,)
        or any(a.shape != (n,) or not np.isfinite(a).all() for a in arrays)
    ):
        raise ValueError("PPO requires matching finite rollout vectors and integer actions")
    log_probability = logits - logits.logsumexp(axis=1, keepdims=True)
    selected = log_probability[np.arange(n), actions]
    ratio = (selected - Tensor(old_logp)).exp()
    surrogate = (ratio * Tensor(advantages)).minimum(
        ratio.clip(1 - clip, 1 + clip) * Tensor(advantages)
    )
    policy = -surrogate.mean()
    value = ((values - Tensor(returns)) ** 2).mean() * 0.5
    entropy = -(logits.softmax(axis=1) * log_probability).sum(axis=1).mean()
    return policy + value_coef * value - entropy_coef * entropy, {
        "policy": policy,
        "value": value,
        "entropy": entropy,
        "clip_fraction": float(np.mean(np.abs(ratio.data - 1) > clip)),
        "approx_kl": float(np.mean(ratio.data - 1 - (selected.data - old_logp))),
    }


class ActorCritic(Module):
    def __init__(self, *, rng):
        self.policy = Linear(5, 2, rng=rng)
        self.value = Linear(5, 1, rng=rng)

    def forward(self, x):
        return self.policy(x), self.value(x).reshape(-1)


def evaluate(probabilities, *, seed, episodes=200, greedy=False):
    env, rng = LineWorld(), np.random.default_rng(seed)
    rewards, successes, steps = [], [], []
    for _ in range(episodes):
        state, total = env.reset(), 0.0
        while True:
            p = np.asarray(probabilities(state))
            if (
                p.shape != (2,)
                or np.any(p < 0)
                or not np.isfinite(p).all()
                or not np.isclose(p.sum(), 1)
            ):
                raise ValueError("policy must return two normalized probabilities")
            action = int(p.argmax()) if greedy else int(rng.choice(2, p=p))
            state, reward, terminal, truncated = env.step(action)
            total += reward
            if terminal or truncated:
                rewards.append(total)
                successes.append(terminal)
                steps.append(env.steps)
                break
    return {
        "mean_return": float(np.mean(rewards)),
        "success_rate": float(np.mean(successes)),
        "returns": rewards,
        "successes": successes,
        "steps": steps,
    }


def train_q_learning(*, seed, episodes=600, gamma=0.99):
    env, rng = LineWorld(), np.random.default_rng(seed)
    q = np.zeros((5, 2))
    history = []
    for episode in range(episodes):
        state, total = env.reset(), 0.0
        while True:
            epsilon = max(0.05, 0.5 * (1 - episode / episodes))
            action = int(rng.integers(2)) if rng.random() < epsilon else int(q[state].argmax())
            following, reward, terminal, truncated = env.step(action)
            target = reward + gamma * (not terminal) * q[following].max()
            q[state, action] += 0.2 * (target - q[state, action])
            total += reward
            state = following
            if terminal or truncated:
                history.append(total)
                break
    return q, history


def train_policy(*, seed, method="ppo", updates=80, episodes_per_update=12, gamma=0.99, lam=0.95):
    if (
        method not in ("reinforce", "ppo")
        or type(updates) is not int
        or updates < 1
        or type(episodes_per_update) is not int
        or episodes_per_update < 1
    ):
        raise ValueError("choose reinforce/ppo and positive update/episode counts")
    env, rng = LineWorld(), np.random.default_rng(seed)
    model = ActorCritic(rng=rng)
    optimizer = Adam(model.parameters(), lr=0.03)
    history = []
    for _update in range(updates):
        states, actions, rewards, values, next_values, terminated, ends, old_logp = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        mc_returns, discount_weights, episode_returns = [], [], []
        for _episode in range(episodes_per_update):
            state, start = env.reset(), len(rewards)
            while True:
                observation = Tensor(np.eye(5)[[state]])
                logits, value = model(observation)
                p = logits.softmax(axis=1).data[0]
                action = int(rng.choice(2, p=p))
                following, reward, terminal, truncated = env.step(action)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                values.append(float(value.data[0]))
                next_values.append(float(model(Tensor(np.eye(5)[[following]]))[1].data[0]))
                terminated.append(terminal)
                ends.append(terminal or truncated)
                old_logp.append(
                    float((logits - logits.logsumexp(axis=1, keepdims=True)).data[0, action])
                )
                state = following
                if terminal or truncated:
                    segment = rewards[start:]
                    following_return = 0.0
                    discounted = []
                    for r in reversed(segment):
                        following_return = r + gamma * following_return
                        discounted.append(following_return)
                    mc_returns.extend(reversed(discounted))
                    discount_weights.extend(gamma ** np.arange(len(segment)))
                    episode_returns.append(float(sum(segment)))
                    break
        observation = Tensor(np.eye(5)[states])
        actions = np.array(actions)
        old_logp = np.array(old_logp)
        if method == "ppo":
            advantage, returns = generalized_advantage(
                np.array(rewards),
                np.array(values),
                np.array(next_values),
                np.array(terminated),
                np.array(ends),
                gamma=gamma,
                lam=lam,
            )
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            for _epoch in range(4):
                optimizer.zero_grad()
                logits, value = model(observation)
                loss, diagnostics = ppo_loss(logits, value, actions, old_logp, advantage, returns)
                loss.backward()
                optimizer.step()
        else:
            optimizer.zero_grad()
            logits, _value = model(observation)
            logp = logits - logits.logsumexp(axis=1, keepdims=True)
            loss = (
                -(
                    logp[np.arange(len(actions)), actions]
                    * Tensor(np.array(mc_returns) * discount_weights)
                ).sum()
                / episodes_per_update
            )
            loss.backward()
            optimizer.step()
            diagnostics = {
                "entropy": -(logits.softmax(1) * logp).sum(1).mean(),
                "clip_fraction": 0.0,
                "approx_kl": 0.0,
            }
        history.append(
            {
                "mean_rollout_return": float(np.mean(episode_returns)),
                "entropy": float(diagnostics["entropy"].data),
                "clip_fraction": diagnostics["clip_fraction"],
                "approx_kl": diagnostics["approx_kl"],
            }
        )
    probabilities = model.policy(Tensor(np.eye(5))).softmax(1).data.copy()
    return model, probabilities, history
