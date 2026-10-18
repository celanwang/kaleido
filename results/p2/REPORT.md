# P2 验收：自动微分核心到模型与策略训练

日期：2026-10-18（Asia/Shanghai）。参考实验来自干净代码提交
[`73599db`](https://github.com/celanwang/kaleido/commit/73599db97a2faa86fd9ecad158e98ba07ebeb1d9)。
源码/配置/依赖锁 SHA-256：`66269e907c57f742752aa6f37f6fbd175dcc5f03e3f9486b22251a1ec4fb6810`。
原始结果位于 [reference/](reference/)，没有以拟合预期替代实际运行。

## 完成的模块

| 模块 | 交付与边界 | 验证证据 |
|---|---|---|
| 神经网络核心 | 不可变值上的 Parameter/Module、模型快照、MLP、稳定 CE、索引/softmax/logsumexp；动量 SGD 与耦合 L2 Adam | 18 项数值梯度、事务更新、PyTorch 前向/梯度/单步/多步测试 |
| 注意力 | 单序列单头 scaled dot-product，四投影模块、精确因果 mask | 4 项有限差分、Q/K/V 梯度、未来扰动不变性与边界测试 |
| 代表性模型 | valid CNN、共享参数 tanh RNN、显式 epsilon 的对角潜变量 VAE | 4 项卷积、BPTT、重参数化/解析 KL 与 PyTorch 对照 |
| 强化学习 | Q-learning、REINFORCE、GAE、clipped PPO，区分终止和截断 | 6 项手算、损失/梯度参考与真实策略学习测试 |
| 完整实验 | 固定划分、验证选模、等预算消融、模型重载、数组哈希、独立策略评测 | 8 项归档/指标/选择/重跑与边界测试 |

P2 新增 40 项，累计 **185 项测试通过**，无跳过；Ruff check/format、三阶段 CLI、sdist/wheel 构建通过。
PyTorch 仅参考，所有列出的模型训练由 NumPy core 实现。
每个模块交付代码、笔记和运行 demo，入口见 [README](../../README.md)。

## 训练与评测协议

- seed=42/7/2026；Digits 1797×64，像素除16，每种子分层 1078/359/360。
- MLP 64→32→10；CNN 为4个3×3 valid核、ReLU与144→10线性头；各12轮、每批64，共204次更新。
- MLP SGD/Adam 共用初态与批次，固定学习率0.1/0.01，不声称等调参预算下某优化器更优。
- VAE 64→16→4，20轮共340次更新；β=1/0共用初态、批次及噪声流，评测平均3组固定 epsilon。
- MLP/CNN 用验证 NLL，两个 VAE 训练目标都用 β=1 验证负 ELBO 选模，测试不参与选择。
- RNN 用独立生成的128/64/128条长度6序列，训练150次，另有长度12压力测试。
- RL 固定 LineWorld；Q-learning 600 episodes；两种策略方法各60×12 episodes，PPO每批4轮。
  每个方法另用独立 RNG 的200 episodes评测随机采样策略和贪心策略。

三种子结果描述数据划分与初始化变化，不作总体置信区间或统计显著性解释。
完整固定超参数在 [protocol.json](reference/protocol.json)，运行配置在 [manifest.json](reference/manifest.json)。

## Digits：真实留出集

均值 ± 三组划分的样本标准差；准确率标准差以百分点表示。

| 模型 | 测试准确率 | 测试 NLL |
|---|---:|---:|
| 训练集多数类基线 | 10.278% | — |
| MLP + 动量 SGD | 95.648% ± 0.160 | 0.146651 ± 0.013660 |
| MLP + Adam | 96.111% ± 0.735 | 0.138655 ± 0.008143 |
| CNN + Adam | 96.481% ± 0.893 | 0.134869 ± 0.011956 |

模型容量与学习率不同，不能从小表推断普遍架构排名。验证选中的轮数均保存于原始指标，
包含低于最终轮次的检查点，避免把最终迭代误称为验证最优模型。

## VAE：重建与先验约束

β=1 模型的平均测试 Monte Carlo 负 ELBO=25.8973，KL=1.6445 nats/sample；
均值潜变量路径重建 MSE=0.048191，训练集均值像素基线约0.07335。
灰度为软 Bernoulli 目标，这是图像建模代理，不能解读为精确离散图像似然。

| 同样340次更新后的训练目标 | 测试重建 NLL | 测试 KL |
|---|---:|---:|
| β=1 | 24.2528 | 1.6445 |
| β=0 | 21.4604 | 39.3519 |

移除 KL 后重建更好，但 posterior 与 prior 偏离大。两种训练目标用共同负 ELBO 选模时，
β=0 的最佳轮数是1/1/2，不能把这些早期模型与最终等预算消融混为一谈。
归档同时保存选中与最终模型，以及固定 prior z/生成结果；生成质量没有人工评分或外部基准结论。

## 序列与注意力

RNN 在长度6独立测试集的三组准确率均为100%；长度12分别为88.281%、100%、100%。
训练长度上的正确性不保证长度外推，且压力测试只有固定种子，不能扩大为一般性结论。
最新位基线明显低于模型，说明任务确实要求保留首位，而非读取最后输入。

注意力前向与 PyTorch 的最大差为4.44×10⁻¹⁶，Q/K/V梯度最大差5.55×10⁻¹⁷。
改变未来 V 后过去输出差为0，零 Q/K 的因果注意力与 prefix mean 一致。
这是机制验证；尚无多头、位置编码、Transformer块或语言模型训练。

## LineWorld：估值到策略优化

| 策略 | 三种子平均成功率 | 平均回报 |
|---|---:|---:|
| 均匀随机 | 40.5% | 0.20333 |
| Q-learning 贪心 | 100% | 0.94000 |
| REINFORCE 随机采样 | 100% | 0.92970 |
| PPO 随机采样 | 100% | 0.93993 |

训练后贪心策略均达到4步最短路径回报0.94。各算法训练预算不同，不能比较样本效率。
模型只在一个5状态MDP验证，不声称学会复杂控制或跨环境泛化。
GAE 的终止/截断、PPO 正负优势 clipping 已通过独立手算与 PyTorch 梯度检查，
不以奖励上升替代算法正确性验证。

## 归档与复现

```sh
uv sync --locked --all-extras
uv run --locked --all-extras pytest
uv run --locked --all-extras python -m kaleido.demos.deep_learning --output runs/p2-reproduce
```

环境：Python3.12.15、NumPy2.5.4、sklearn1.9.1、SciPy1.18.1、PyTorch2.14.1、uv0.12.23。
CPU/float64；本机单次计算与写入约3.91秒，不含依赖导入，不作为设备性能基准。
跨平台浮点末位、文件时间戳、运行ID与耗时可变化；同机重复运行的数值/数组指纹已验证一致。

| 产物 | 内容 |
|---|---|
| [data.npz](reference/data.npz) / [data.json](reference/data.json) | 完整输入、标签、划分ID、来源与每数组指纹 |
| [models.npz](reference/models.npz) / [models.json](reference/models.json) | 模型参数、选中与最终VAE、参数清单与指纹 |
| [outputs.npz](reference/outputs.npz) / [outputs.json](reference/outputs.json) | logits、重建、噪声、prior样本、注意力与逐episode评测 |
| [metrics.json](reference/metrics.json) / [histories.json](reference/histories.json) | 各种子原始指标、验证与训练曲线 |
| [checks.json](reference/checks.json) / [manifest.json](reference/manifest.json) | 注意力误差、干净代码版本、配置和依赖环境 |

NPZ 只含数值数组，读取使用 `np.load(path, allow_pickle=False)`。
归档已复核全部数组指纹、验证选择、模型重载、逐样本预测与episode指标。
模型快照用于推理，不包含训练恢复所需的优化器矩和RNG状态；该边界不隐藏为“完整恢复”。

## 阶段出口

P2 按代表性案例完成；VAE代表生成分支，GAN/扩散与LSTM/GRU为可选扩展。
自研core在MLP/选定注意力出口后冻结，仅修正确性问题。
P3 使用成熟框架完成手写BPE、Decoder-only、小规模真预训练与KV cache对照，
先做资源/语料探测，再定训练预算；见 [路线图](../../docs/ROADMAP.md)。
