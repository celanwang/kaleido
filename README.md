# Kaleido

万花筒：以机器学习全域为结构，以可复现实验理解设计、取舍与失效边界。

原始方向见 [DESIGN](DESIGN.md)，阶段与验收见 [ROADMAP](docs/ROADMAP.md)。
当前已实现 P0 地基、P1 经典机器学习与 P2 深度学习代表案例。P3–P6 尚未实现。
自研核心用于理解，后续规模训练使用成熟框架。
P0 的 [验收报告与完整参考产物](results/p0/REPORT.md)保留原始阶段证据。
P1 的 [验收报告与完整参考产物](results/p1/REPORT.md)包含三组划分与六类算法对照。
P2 的 [验收报告与完整参考产物](results/p2/REPORT.md)覆盖神经模型、生成与强化学习；全仓 185 项测试通过。

## 运行

环境：Python 3.12（本机验证 3.12.15），uv 0.12.23，CPU；无需模型下载、GPU 或外部服务。

```sh
uv sync --locked
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked python -m kaleido.demos.linear_regression
```

P1 的 sklearn 对照与离线真实数据 demo 使用可选依赖：

```sh
uv sync --locked --extra reference
uv run --locked --extra reference pytest
uv run --locked --extra reference python -m kaleido.demos.classical_ml
```

P1 读取 `configs/p1_classical.json`，默认三组固定划分；支持 `--config` 和 `--output`。
所有数据均为 sklearn 内置文件，运行时无需下载。基础实现只依赖 NumPy，sklearn 用于参考与数据。

P2 的 PyTorch 参考和完整训练 demo：

```sh
uv sync --locked --all-extras
uv run --locked --all-extras pytest
uv run --locked --all-extras python -m kaleido.demos.deep_learning
```

P2 读取 `configs/p2_deep.json`，默认三组种子；Digits 用训练/验证/测试独立划分，
MLP/CNN 按验证 NLL 选检查点，VAE 用固定噪声的验证负 ELBO。PyTorch 仅用于参考，
训练由 NumPy core 完成；Linux 参考依赖使用 CPU wheel。安装依赖需要包索引或缓存，demo 数据无需下载。
产物位于 `runs/p2-deep/<id>/`：JSON 记录指标、协议、历史与每个数组哈希；
`data.npz`、`models.npz`、`outputs.npz` 保存完整输入、模型参数、预测与 prior 样本，读取时用
`np.load(path, allow_pickle=False)`。这些是推理模型快照，不含恢复训练所需的优化器/RNG 状态。

默认读取 `configs/p0_linear.json`；可用 `--config <path>` 和 `--output <directory>` 指定。
从仓库根目录运行。每次生成独立 `runs/p0-linear/<id>/`，不覆盖历史，运行产物默认不入 Git。

| 产物 | 内容 |
|---|---|
| manifest.json | 状态、完整配置、Git revision/dirty、源码指纹、环境与命令 |
| data.json | 完整合成数据、固定划分、数据哈希、训练集标准化参数 |
| history.json | 主实验及原尺度消融的训练/验证曲线 |
| model.json | 验证集选中的参数、原尺度参数与 OLS 参考 |
| predictions.json | 每个测试样本的真值和三种预测 |
| metrics.json / cost.json | 对照指标、失效案例、梯度检查、计算预算与耗时 |

## 实现与笔记

- `src/kaleido/experiments.py`：[实验协议](docs/notes/experiment-protocol.md)。
- `src/kaleido/core/`：[反向自动微分](docs/notes/autodiff.md)，float64、广播与二维矩阵乘。
- `src/kaleido/demos/linear_regression.py`：[回归实验](docs/notes/linear-regression.md)，
  训练均值与 NumPy OLS 对照、尺度消融和标签漂移压力测试。
- [最小数学地基](docs/notes/math-foundations.md)：矩阵求导、谱与步长、高斯似然、熵和交叉熵。
- `src/kaleido/classical/`：统一估计器接口、[线性/逻辑回归](docs/notes/classical-linear.md)、
  [CART 分类树](docs/notes/cart.md)、[PCA](docs/notes/pca.md)、[K-means 与 EM](docs/notes/clustering-em.md)。
- `src/kaleido/demos/classical_ml.py`：[P1 实验协议](docs/notes/classical-experiments.md)，
  Diabetes/Breast Cancer/Iris 留出集、验证选参、sklearn 对照与机制反例。
- `src/kaleido/core/nn.py` / `optim.py`：[参数、MLP、SGD/Adam 与稳定分类损失](docs/notes/neural-core.md)。
- `src/kaleido/core/attention.py`：[单头因果注意力](docs/notes/attention.md)。
- `src/kaleido/core/models.py`：[CNN、RNN、VAE 代表案例](docs/notes/cnn-rnn-vae.md)。
- `src/kaleido/rl.py`：[Q-learning、REINFORCE、GAE 与 PPO](docs/notes/policy-gradients.md)。
- `src/kaleido/demos/deep_learning.py`：[P2 实验协议](docs/notes/deep-experiments.md)。

监督模型 `fit(X,y)` 返回自身，提供 `predict(X)`；分类器另提供 `predict_proba(X)`。
Scaler/PCA 使用 `fit/transform/inverse_transform`，聚类提供 `fit/predict`。
目标 y 为一维，X 为有限二维实数。逻辑回归限二分类，树限 Gini 分类，EM 限对角协方差；
没有样本权重、缺失值处理或完整 sklearn 兼容层。SVM、集成模型留为扩展。

数值梯度验证、独立解析公式和固定划分用于判断正确性。回归为已知生成规律的合成实验，
用于机制验证，不能推广为真实业务效果或大模型能力结论。core 不支持 GPU、批量 matmul
或高阶导数；参数通过创建新叶子张量更新，避免修改旧计算图的前向值。
P2 只支持单序列单头注意力、stride=1/valid 卷积、tanh RNN 和选定 VAE；
GAN/扩散、LSTM/GRU 等作为扩展，自研框架在当前学习案例后冻结，P3 使用成熟框架承担训练。
