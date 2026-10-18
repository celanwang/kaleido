# P0 验收：最小实验规范到自动微分训练闭环

运行日期：2026-10-18（Asia/Shanghai）。代码版本
[`22e2e47`](https://github.com/celanwang/kaleido/commit/22e2e47a46a63d142334936be219373ab1633ae7)，
运行时 Git 工作区干净。完整产物在 [reference/](reference/)，报告不以手工估计代替实际数据。

## 交付与验证

| 模块 | 代码与产物 | 正确性证据 |
|---|---|---|
| 实验协议 | `src/kaleido/experiments.py`，独立运行目录与失败记录 | 7 项协议测试 |
| 张量与反向自动微分 | `src/kaleido/core/`，广播、二维矩阵乘、归约与初等非线性 | 62 项 NumPy/解析/中心差分/边界测试 |
| 回归闭环 | `src/kaleido/demos/linear_regression.py`，配置化训练、逐样本输出 | 17 项隔离/重跑/多 seed 收敛/失败测试 |

合计 **86 项测试通过**；Ruff check/format、完整 CLI 运行和 sdist/wheel 打包均通过。
每个模块有笔记，统一回归 demo 实际使用实验记录与自动微分；数学笔记串联矩阵求导、
Hessian、似然与信息论。此阶段完成的是主线所需最小地基，未声称覆盖全部数学主题。

## 实验协议

- 已知生成规律的合成线性任务，3 特征尺度为 1/3/10，高斯噪声标准差 0.2，seed=42。
- 固定且不重叠的训练/验证/测试样本数为 320/96/96；标准化仅拟合训练数据。
- 全批次梯度下降 300 次，学习率 0.05，CPU float64。
- 验证 MSE 最低的第 57 次更新用于效果评估；300 次更新的最终参数另与 OLS 对照。
- OLS 与均值预测均只使用训练标签；测试集未用于检查点选择或调参。

## 实测结果

| 条件或检查 | 结果 |
|---|---:|
| 自动微分回归：训练 MSE（选中检查点） | 0.0412158531 |
| 自动微分回归：验证 MSE | 0.0421651188 |
| 自动微分回归：测试 MSE | 0.0445505910 |
| 训练均值基线：测试 MSE | 38.2828417274 |
| NumPy OLS：测试 MSE | 0.0433428036 |
| 最终迭代参数与 OLS 最大绝对误差 | 2.3803×10⁻¹³ |
| 训练损失中心差分梯度最大绝对误差 | 3.9462×10⁻⁹ |
| 标准化后的稳定学习率上界 | 0.9527456421 |
| 原尺度的稳定学习率上界 | 0.0108759775 |
| 原尺度、相同步长消融 | 第 6 次更新超过 10¹² MSE，提前终止 |
| 测试标签人为 +5：MSE | 25.2358057254 |

解释：最终优化收敛到最小二乘解，留出误差接近已知噪声方差 0.04 的量级。
验证选择与最后迭代的参数不同，不能把最终 OLS 参数误差误称为选中模型的参数误差。
原尺度下 0.05 超过 Hessian 推导出的稳定上界，消融将发散与机制联系起来；
降低原尺度学习率可以稳定，不能由本实验推出标准化在所有任务必然更优。
标签漂移是人为压力测试，说明分布假设失效后小训练误差无法保证小测试误差。

## 来源与复现

```sh
uv sync --locked
uv run --locked pytest
uv run --locked python -m kaleido.demos.linear_regression --output runs/p0-reproduce
```

Python 3.12.15，NumPy 2.5.4，uv 0.12.23；精确依赖见根目录 uv.lock。
参考运行端到端计算与文件写入约 0.041 秒，仅为本机单次观察，不是性能基准。
完整平台、命令、配置与状态见 [manifest.json](reference/manifest.json)。

- 源码/配置/锁文件 SHA-256：`9100f543b5a7059829679af0c34040cd5a560f654a05fdee107fc2e07c1cbfcd`
- 数据及划分 SHA-256：`064a5f60c2267d5ced9fa9769fe7b09e12fe0ef47555807fb66cd1be0e005fbb`
- [metrics.json](reference/metrics.json)：全部未舍入指标。
- [data.json](reference/data.json)：完整生成数据、划分与训练标准化参数。
- [history.json](reference/history.json)：主实验与消融的每次迭代。
- [model.json](reference/model.json) / [predictions.json](reference/predictions.json)：参数与每个测试样本预测。

相同锁定环境下重复运行得到相同数值；时间戳、耗时、路径和运行 ID 会变化。
不同 BLAS/平台的浮点末位可不同。多 seed 测试只验证优化收敛，
本报告不包含多次独立训练的效果置信区间，也不对真实数据集或语言模型能力作结论。

## 后续边界

P0 完成，P1–P6 仍待实现。下一步先整理经典估计器 API，做线性/逻辑回归与 sklearn 对照。
core 保持 CPU float64 和二维 matmul 边界；规模训练、分类稳定算子、Qwen、GRPO 与推理优化
均不属于本轮交付。阶段安排与验收条件见 [ROADMAP](../../docs/ROADMAP.md)。
