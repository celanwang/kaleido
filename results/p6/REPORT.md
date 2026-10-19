# P6：复现、导出部署与 v1.0.0 发布候选

2026-10-19，Apple M4 Max / 48 GiB / macOS 15.8。P6 工程实现与验收完成：统一入口、工件审计、
全阶段正式复跑、P3 模型导出与独立 wheel 推理均已实际执行。
**v1.0.0 是发布候选；本次没有公开 GitHub Release、创建发布 tag 或上传 PyPI。**
合并功能 PR 后，从 main 手动运行发布流程可重新校验并生成 draft，公开发布仍需维护者审阅。

## 已实现的统一入口

`python -m kaleido` 与安装后的 `kaleido` 提供同一组命令：

| 命令 | 作用与边界 |
|---|---|
| list | 列出 P0–P5 模块、正式/接线配置、依赖与模型下载需求 |
| doctor | 检查 Python 3.12 和直接依赖版本；支持同版本 +cpu 构建标签，不下载模型 |
| verify --deep | 校验已审查的输入与历史工件，并检查有 sidecar 的 NPZ 数组摘要 |
| run p0 … p5 | 显式选择配置、输出目录、formal/smoke 模式，记录实际配置与配置文件摘要 |
| reproduce | 每个阶段在独立进程运行，保留总 manifest、步骤、实时落盘 stdout/stderr |
| export | 导出真实 P3 检查点，验证原模型/计算图 logits 与生成 token |
| predict | 不需要 checkout 的本地 JSON-lines 推理，逐请求校验并保留非零失败退出码 |

源码外执行研究命令须用全局 `--root PATH` 指向 checkout；wheel 不假装包含语料和历史模型。
旧 demo 入口仍可运行，没有改写 NumPy core 或 P3/P4 训练算法。

[审查清单](../../artifacts/inventory-v1.json) 覆盖 140 个输入/历史文件、1,361 个 sidecar 数组摘要，
[本轮校验](reference/verification.json) 全部通过。缺失、摘要错误、越界路径会失败，校验不会自动重建预期摘要。
封存工具仅供维护者审查后主动更新；这些哈希提供相对于清单的完整性，不提供外部签名真实性。
P6 新证据另有[递归索引](reference/artifacts.json)，约9.3 MB，旧 P0–P5 证据未被替换。

## 完整正式复跑

在干净提交 `50ebad6ec27cecfa9c63487ca19af2aab3468d26` 上，按原正式配置依次执行 P0–P5，
不是把 fixture 当训练证据。总运行约267.74秒；每个子阶段都记录 completed、formal、非 fixture。
执行源摘要 v2 为 `188d8e6d8d869fbfbe93dded83e4d2a3df43831dea39ee362a2ab85bb685f84b`。
v2 在原 Python/配置/依赖范围外加入 scripts、workflow 和输入清单，旧阶段原摘要不改写。

| 阶段 | 本轮配方成本 / 秒 | 与历史证据的实际对照 |
|---|---:|---|
| P0 | 0.031 | 回归任务指标一致，保留分布漂移与发散反例 |
| P1 | 0.270 | 三组划分的代表算法与参考指标一致 |
| P2 | 3.885 | 任务指标一致，356 个数据/模型/输出数组摘要一致 |
| P3 | 35.972 | 三种子各1000步；NLL 在预设容差内，权重及部分逐token损失不逐位一致 |
| P4 | 170.296 | 三种子 SFT/GRPO/筛选监督正式重训，九个 adapter 共864个数组摘要及288条测试输出一致 |
| P5 | 53.238 | 在发布的冻结工件上重新生成1,080条回答，与原 P5 逐题记录一致；Judge 仍未达门槛 |

[reproduction](reference/reproduction.json) 保存完整配置回执、各阶段环境、指标、成本、文件索引与对照。
P5 仍评测发布的 P3/P4 检查点，**不会自动消费同轮新训练模型**；不能把同级目录排列称为新的训练血缘。
另一次在最终部署代码上执行的[六阶段 smoke](reference/smoke.json)完整通过；P4/P5 在其中明确标记 fixture。
上述耗时是本次配方运行成本，不是通用性能基准。

## P3 重训：保留差异，不承诺位级训练复现

| 种子 | 原测试 NLL | 重训测试 NLL | 差值 | 权重最大绝对差 |
|---|---:|---:|---:|---:|
| 42 | 2.776067243632 | 2.776067236202 | −7.43e−9 | 7.3001e−4 |
| 7 | 2.773708656812 | 2.773708682355 | +2.55e−8 | 1.1882e−3 |
| 2026 | 2.802522993219 | 2.802522677423 | −3.16e−7 | 9.7602e−4 |

初始化摘要相同，任务指标在预先给定的 atol=rtol=1e−5 范围内，三组生成样例的 token 也一致。
但模型状态不逐位相同，因此整体数组对照明确为 false；原始 JSON 对照也为 false，
没有把这种失败重命名为完全复现。其90项差异为84个权重摘要、6个训练耗时/吞吐值；其他已比较任务指标无超容差差异。

最大差异均位于 QKV bias 的 key 段，排除该段后最大参数差分别约1.33e−6、1.97e−6、1.89e−5。
这只是本轮定位，不是特定内核原因的证明，也不意味着其他差异必然无影响。
[新权重及逐token损失](reference/changed/p3/)与[诊断](reference/training_diagnostics.json)一并保留，旧模型保持原样。
本版本承诺可追溯的配置、数据和显式对照，不承诺 MPS 训练位级一致或跨设备相同结果。
固定 seed 与一般可复现性也不等价，参见 [PyTorch 复现说明](https://docs.pytorch.org/docs/stable/notes/randomness.html)。

## 真实模型导出与本地部署

导出对象是原发布的 P3 seed42 检查点（478,720参数），不是随机 fixture，也不是本轮重训后替换的模型。
导出源为干净提交 `56d17aec7f9857d9179daa453b07d8fa7b0a50d2`，其源摘要为
`3e686b5cd620c13fb7ee79e5b48085d5f536ed2666e83bed0079b00d2515231e`。
与正式复跑提交的差异主要为隔离安装修复及用户文档，不改训练算法；两个来源分别记录。

[部署 bundle](reference/bundle/manifest.json) 包含 CPU float32 `torch.export` 计算图、byte-BPE、
固定生成协议、验证样例和文件清单。加载计算图无需 `DecoderLM` 或训练模块。
动态长度1..128，batch1；长度1、2、7、127、128的 logits 与原模型最大绝对误差均为0，greedy一致。
三条短/长提示的完整裁窗生成逐token一致，长提示覆盖超过 context 的路径。

导出后再做实际 wheel 安装：新建独立 venv，按 uv.lock **离线同步** CPU 运行时，
不安装 editable 项目，然后安装生成的 `kaleido-1.0.0-py3-none-any.whl`。
从仓库外以 `python -I` 运行；断言导入位置位于新 venv、源码目录不在 sys.path、
HF 包未安装、训练模块未导入，并重放三条样例。[wheel_check](reference/wheel_check.json)全部通过。
实测 wheel 为109,187 bytes，SHA256 为 `6018c77002f00a42ac56ed5f27555013f964a5d2810949ea14cc4e2334e41a7b`。

Linux CI 初次发现：requirements 安装仍可能缺包索引缓存，即使对应 wheel 已缓存。
修复后通过 `UV_PROJECT_ENVIRONMENT` 指定新环境，用 `uv sync --locked --offline --no-install-project`
直接消费锁文件中的工件信息，不再依赖二次索引解析；Mac 与 Linux 检查通过。
准备依赖的安装阶段需要缓存或网络，推理本身不需要 checkout、语料、Qwen 或 HF 网络。

部署边界是本地单进程 JSON-lines：单行≤64KiB、prompt≤16KiB、新token预算0..128；
异常行单独失败，后续行继续处理，有失败则退出码非零。已有导出目录拒绝覆盖，失败不发布半成品目录。
完整裁窗 greedy 没有导出 KV cache，没有量化，也没有速度提升承诺；不是 HTTP 并发服务。
导出格式仅验证锁定的 PyTorch 2.14.1，不承诺跨版本/跨框架/加速器兼容。

## v1.0.0 范围与发布状态

项目版本、[CHANGELOG](../../CHANGELOG.md)、[发布说明](../../docs/releases/v1.0.0.md)与
[复现指南](../../docs/REPRODUCING.md)已就绪。
[手动发布流程](../../.github/workflows/release.yml)只允许 main、匹配项目版本且尚不存在的 tag；
重新执行校验、测试、smoke、导出、独立 wheel 推理，然后生成 wheel、sdist、Decoder zip、
验证报告与 SHA256SUMS，最后只创建 GitHub draft release。

[本次发布状态](reference/release_state.json)明确为 candidate_pending_merge_and_manual_draft：
**发布流程的 GitHub draft 创建步骤尚未执行，本次未生成 tag/公开 Release/上传 PyPI。**
主线代码可审阅、合并；公开发布是之后的独立维护者动作，不能把包内版本号当作已经发布。

v1.0.0 的范围是已验证的研究主线，不是原 DESIGN 全域扩展全部完成。
SVM/集成、GAN/扩散、DPO/PPO、量化、投机采样、通用人工偏好 Judge 等仍未实施。
P4/P5 未证明稳定 GRPO 迁移收益，Judge 校准失败的负结果继续保留。

## 验收命令与结果

```sh
uv sync --locked --all-extras
uv run --locked --all-extras kaleido doctor
uv run --locked --all-extras kaleido verify --deep
uv run --locked --all-extras kaleido reproduce --profile smoke
uv run --locked --all-extras kaleido reproduce --profile formal
uv run --locked --all-extras pytest
uv run --locked --all-extras ruff check .
uv run --locked --all-extras ruff format --check .
uv build
```

全仓274项测试、lint/格式、wheel/sdist构建通过。CI 使用统一入口，并在 Linux 验证独立 wheel 推理。
新测试覆盖清单破损/缺失/越界、失败证据保留、CPU版本标签、动态图边界、逐请求恢复、
导出失败清理、发布工件完整性和实际重训差异；不以 smoke 替代正式训练或公开发布证据。
