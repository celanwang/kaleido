# 从 Q-learning 到 PPO：估值、采样与更新约束

LineWorld 是 5 状态、2 动作的可控 MDP；到右端终止给 +1，其他步 −0.02，12 步上限为外部截断。
Q-learning 的终止目标不 bootstrap，截断仍 bootstrap。这个玩具环境用于机制验证，不代表复杂控制能力。

REINFORCE 用同一批策略采样、固定 Monte Carlo returns，再更新 log π；显式包含 γ^t 权重，
优化折扣回报的 episode sum，而不是将长轨迹误当额外独立 episode。
PPO 固定 rollout 的 old_logp/values/advantage/returns，做四轮 clipped surrogate 更新。
正负 advantage 的 clipping 行为不同，不能简单把所有 ratio 截断后直接乘 advantage。
价值损失和 entropy 同时记录；没有 value clipping、KL early stopping 或生产级采样器。

GAE δ=r+γ(1−terminated)V(next)−V(s)，递推在 episode_end（含截断）断开。
截断既要用最后观测 bootstrap，又不能把新 episode 的优势串进来。
λ 控制偏差/方差，优势归一化只作用于策略项，critic 目标保持原回报尺度。
rollout 常量均为 NumPy 数据，与新图脱离，不对旧策略/奖励反传。

测试：`uv run --locked --all-extras pytest tests/test_rl.py`，含独立手算边界、PyTorch loss/梯度、
Q-learning 与两种策略训练。demo 用独立评测随机种子，保存每个 episode 结果，报告随机策略基线。
参考：[PPO](https://arxiv.org/abs/1707.06347)、[GAE](https://arxiv.org/abs/1506.02438)。
