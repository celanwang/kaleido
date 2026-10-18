# Kaleido

万花筒：以机器学习全域为结构，以可复现实验理解设计、取舍与失效边界。

原始方向见 [DESIGN](DESIGN.md)，阶段与验收见 [ROADMAP](docs/ROADMAP.md)。
当前实现 P0：本地实验协议、NumPy 张量自动微分、合成回归训练闭环。
P1–P6 尚未实现。自研核心用于理解，后续规模训练使用成熟框架。

## 运行

环境：Python 3.12（本机验证 3.12.15），uv 0.12.23，CPU；无需模型下载、GPU 或外部服务。

```sh
uv sync --locked
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked python -m kaleido.demos.linear_regression
```

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

数值梯度验证、独立解析公式和固定划分用于判断正确性。回归为已知生成规律的合成实验，
用于机制验证，不能推广为真实业务效果或大模型能力结论。core 不支持 GPU、批量 matmul、
索引反传或高阶导数；参数通过创建新叶子张量更新，避免修改旧计算图的前向值。
