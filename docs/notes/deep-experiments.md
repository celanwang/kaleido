# P2 实验协议与对照边界

全程 NumPy/CPU/float64 训练。PyTorch 仅承担小数组的前向、梯度和更新参考；
不是将 PyTorch 模型训练后包装为自研结果。数据、模型和输出均以无 pickle 的压缩 NPZ 保存，
每个数组记录 dtype/shape/content SHA-256，原有 Run 协议记录代码与依赖来源。

Digits 来自锁定 sklearn 内置文件，像素除16利用已知量程，不估计全数据统计量。
seed=42/7/2026 各自分层 1078/359/360；MLP 64→32→10 与 4 核 CNN 用验证 NLL 选模。
SGD/Adam 的 MLP 初始权重、批次顺序和更新预算一致，但学习率分别固定为0.1/0.01，
不是等调参预算的最优优化器比较。CNN 模型容量不同，结果不解释为架构普遍排名。

VAE 64→16→4，20轮，β=1/0 共用初态、批次、epsilon 流与固定评测噪声。
验证和测试负 ELBO 平均3个固定 epsilon 样本；两种训练目标都以 β=1 的验证负 ELBO 选模。
额外保留最终等预算模型与 KL/重建，区分“训练目标消融”与“最佳检查点效果”。
噪声均值路径的重建 MSE 另列，不能代替 Monte Carlo ELBO；像素交叉熵是软 Bernoulli 代理。
保存16个固定 prior z 与生成输出，以便检查潜变量先验采样行为。

RNN 的 train/valid/test=128/64/128，长度6的随机二值序列，标签为首位。
从最后隐藏状态预测，最新位预测是基线。长度12样本为独立长度压力测试，不能当自然 OOD 基准。

注意力保存 Q/K/V、因果概率与输出，比较 PyTorch 输入梯度；改变后半部分 V 时前半输出不变。
零 Q/K 的因果注意力应恰好计算 prefix mean，这是机制 demo，不是语言模型能力实验。

RL 用同一 LineWorld，Q-learning 600 episodes，REINFORCE/PPO 各60×12 episodes，
PPO 每批四轮更新。随机策略和训练后策略在独立 RNG 的200 episodes上评测，同时列随机采样与贪心结果。
三方法训练成本不同，不作样本效率排名；固定环境布局没有跨环境泛化意义。
策略项优势归一化，critic 目标不归一化；终止与截断的处理由手算和自动测试约束。

测试：`uv run --locked --all-extras pytest`；实际运行：
`uv run --locked --all-extras python -m kaleido.demos.deep_learning`。
测试覆盖 checkpoint 重载、哈希、互斥划分、验证选择、指标复算和重复运行的数值一致性。
三种子结果只描述这些数据划分/初始化的变化，不能作为独立总体试验的置信区间。
