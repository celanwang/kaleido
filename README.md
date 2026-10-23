# Kaleido

万花筒：以机器学习全域为结构，以可复现实验理解设计、取舍与失效边界。

原始方向见 [DESIGN](DESIGN.md)，阶段与验收见 [ROADMAP](docs/ROADMAP.md)。
当前已实现 P0–P6 主线，P6 已合并，v1.0.0 正在完成最终交付审阅；未实施的扩展见 [CHANGELOG](CHANGELOG.md)。
自研核心用于理解，后续规模训练使用成熟框架。
P0 的 [验收报告与完整参考产物](results/p0/REPORT.md)保留原始阶段证据。
P1 的 [验收报告与完整参考产物](results/p1/REPORT.md)包含三组划分与六类算法对照。
P2 的 [验收报告与完整参考产物](results/p2/REPORT.md)覆盖神经模型、生成与强化学习，阶段验收时185项测试通过。
P3 的 [验收报告与训练权重](results/p3/REPORT.md)覆盖真实三种子预训练、逐token评测与cache实测，阶段验收时209项测试通过。
P4 的 [验收报告与adapter](results/p4/REPORT.md)保存固定Qwen三种子SFT/GRPO/采样筛选对照；阶段验收时235项测试通过。
P5 的 [评测报告与逐题证据](results/p5/REPORT.md)整合P3/P4、新结构题、公开子集和Judge校准；阶段验收时258项测试通过。

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

P3 正式预训练与快速接线检查：

```sh
uv run --locked --all-extras python -m kaleido.demos.language_model
uv run --locked --all-extras python -m kaleido.demos.language_model --config configs/p3_smoke.json
```

正式预算在 `configs/p3_pretrain.json`，三种子各1000步；先实测CPU/MPS再选设备，
原文已随仓库固定，无需运行时下载。原始声明与使用说明在 [data/p3](data/p3/README.md)。
模型使用PyTorch从随机初始化训练，BPE与Decoder关键逻辑显式实现。
输出含词表、章节划分/指纹、计数基线、每token NLL、模型、生成、cache数值检查及重复计时。
smoke仅验证接线，不代表完成正式训练。所有示例均从仓库根目录运行，可通过 `--output` 指定产物目录。

P4固定Qwen后训练与离线fixture：

```sh
uv run --locked --all-extras python -m kaleido.demos.post_training
uv run --locked --all-extras python -m kaleido.demos.post_training --fixture --config configs/p4_smoke.json
```

正式配置在 `configs/p4_posttrain.json`，模型revision/SHA在 `configs/p4_model.json`。
首次正式运行需下载约1GB官方权重到HF缓存，随后校验并使用固定原生tokenizer/template。
默认MPS/float32、q/v LoRA，依次完成基座评测、SFT、GRPO和固定SFT采样筛选监督对照。
仓库不存完整基座；adapter重放仍需原revision。显式fixture是随机tiny模型，无效果结论。

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
- `src/kaleido/lm/`：[BPE与语料](docs/notes/bpe-corpus.md)、[Decoder与cache](docs/notes/decoder-cache.md)、
  [预训练实验协议](docs/notes/pretraining-experiment.md)。
- `src/kaleido/posttrain/`：[任务与独立判定](docs/notes/verifiable-tasks.md)、
  [原生Qwen/LoRA](docs/notes/qwen-lora.md)、[GRPO公式](docs/notes/grpo-objective.md)、
  [后训练实验协议](docs/notes/posttrain-experiment.md)。

监督模型 `fit(X,y)` 返回自身，提供 `predict(X)`；分类器另提供 `predict_proba(X)`。
Scaler/PCA 使用 `fit/transform/inverse_transform`，聚类提供 `fit/predict`。
目标 y 为一维，X 为有限二维实数。逻辑回归限二分类，树限 Gini 分类，EM 限对角协方差；
没有样本权重、缺失值处理或完整 sklearn 兼容层。SVM、集成模型留为扩展。

数值梯度验证、独立解析公式和固定划分用于判断正确性。回归为已知生成规律的合成实验，
用于机制验证，不能推广为真实业务效果或大模型能力结论。core 不支持 GPU、批量 matmul
或高阶导数；参数通过创建新叶子张量更新，避免修改旧计算图的前向值。
P2 只支持单序列单头注意力、stride=1/valid 卷积、tanh RNN 和选定 VAE；
GAN/扩散、LSTM/GRU 等作为扩展，自研框架在当前学习案例后冻结，P3 使用成熟框架承担训练。

P5 统一评测与离线接线检查：

```sh
uv run --locked --all-extras python -m kaleido.demos.model_evaluation
uv run --locked --all-extras python -m kaleido.demos.model_evaluation --fixture --config configs/p5_smoke.json
```

正式评测加载 P3 三个预训练权重及 P4 九个冻结 adapter，保留同一固定 Qwen 基座、原生模板，
不训练或重选检查点。新题包括64道结构隔离算术、固定32题GSM8K直接回答子集及12道颜色检索。
题目级配对 bootstrap 保留全部固定种子；按题族/难度报告格式、内容与截断。
本地 Judge 经过已知gold校准和顺序交换，未达门槛时自动关闭模型排名。
本轮独立新题未证明稳定GRPO收益；Judge一致率仅50%，排名已关闭。
颜色提示与解析约定修复前后的记录均保留；修复后的结果属于描述性验证。
完整数据、限制和复现命令见 [P5报告](results/p5/REPORT.md) 与 [协议笔记](docs/notes/evaluation-protocol.md)。

P6 统一入口、复现与导出：

```sh
uv run --locked --all-extras kaleido list
uv run --locked --all-extras kaleido doctor
uv run --locked --all-extras kaleido verify --deep
uv run --locked --all-extras kaleido reproduce --profile smoke
uv run --locked --all-extras kaleido reproduce --profile formal
uv run --locked --all-extras kaleido export --weights results/p3/reference/model-42.npz --tokenizer results/p3/reference/tokenizer.json --output runs/decoder-42
uv run --locked --all-extras kaleido predict --bundle runs/decoder-42
```

`predict` 从 stdin 读取 JSON-lines，例如 `{"prompt":"Alice was","max_new_tokens":12}`。
P3 导出为 CPU 动态长度计算图，支持独立 wheel 安装后在仓库外推理；不需要训练模块、语料或 Qwen 缓存。
完整研究复现仍需 checkout，源码外运行时用全局 `--root /path/to/kaleido` 指定。
正式复跑保留配置、失败日志及与历史工件的差异；固定种子不被解释为 MPS 权重位级一致的保证。
检查、打包与发布均由维护者在本地手动执行，不自动上传 PyPI。
见 [复现指南](docs/REPRODUCING.md)、[P6验收报告](results/p6/REPORT.md) 和 [v1.0.0范围](docs/releases/v1.0.0.md)。

最终交付套件把普通 wheel、冻结模型、锁定运行时和标准库安装脚本一并打包，可在没有源码的目录安装：

```sh
python3.12 install.py --verify-only
python3.12 install.py --env /absolute/path/to/new-venv
```

可直接使用 `results/final/reference/delivery/` 中的套件，或通过 `kaleido package` 在本地生成套件后执行以上命令。
离线安装可添加 `--offline --cache-dir /path/to/prepared/uv/cache`。
完整入口与验收见 [最终交付报告](results/final/REPORT.md)。公开 Release 由维护者手动创建与审阅。
