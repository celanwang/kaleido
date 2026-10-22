# v1.0 最终可靠性交付

2026-10-23。目标：**完成可独立安装、可完整校验、可追溯运行的 v1.0 交付闭环。**
P0–P6 已合并，本轮围绕实际交付缺口加固，不扩展训练算法或改写旧实验结论。

| 已确认的缺口 | 最终实现 |
|---|---|
| 总校验原来只覆盖 P0–P5 | 现在覆盖 P0–P6、最终交付套件和依赖输入；未入清单的新工件也会失败 |
| 源配置在执行后才计算摘要 | 执行前固定 bytes，随 run 保存 configuration.json、receipt.json；源文件被修改时明确标记 |
| 子进程零退出和目录存在就可被当作完成 | 核对分配目录、持久回执、manifest、原始配置及 profile/fixture，错误停止后续步骤 |
| 模型只校验基座权重 | 原生配置、生成配置、tokenizer、词表/merges 和许可也固定摘要，先验证再构造模型 |
| 元数据边界不足 | 拒绝重复键、非对象、非有限数字和非法 schema；声明的维度须与导出图 embedding 一致 |
| 独立安装仍借用源码项目 | 新套件附带 wheel、Decoder、PEP751 pylock、标准库安装脚本及清单，安装和推理均可脱离 checkout |
| 交付资产尚不可下载审阅 | CI 增加套件复制安装、逐token重放和 ZIP/摘要/报告上传；main draft 发布流程同步加入套件 |

## 交付物与运行

[独立套件](reference/delivery/README.md)及[清单](reference/delivery/manifest.json)已随仓库保存。
CI 配置会在检查通过后上传 `kaleido-cpu-delivery` ZIP、外部摘要和独立安装报告。
本次本机验证已完成；GitHub Linux 新套件作业当前等待 runner，尚未生成远端附件，不能把旧 P6 CI 通过当作本次套件 CI 通过。
解压并进入套件目录后：

```sh
python3.12 install.py --verify-only
python3.12 install.py --env /absolute/path/to/new-venv
```

需要 uv0.12.23；首次按 pylock 中的工件 URL/摘要下载锁定依赖。
准备好缓存后可加 `--offline --cache-dir /absolute/path/to/cache`。依赖 wheel 不内嵌，
无需 checkout、Qwen、语料或 HF 包；已有环境拒绝覆盖，失败环境保留用于诊断。
安装后运行新环境中的 `kaleido predict --bundle /absolute/path/to/decoder-42`，输入 JSON-lines。
本机实际把套件复制到临时目录，以基础 Python 的隔离模式启动，创建新 venv，离线安装并重放三条非空样例。
[consumer-check](reference/consumer-check.json)记录普通 wheel 安装位置、无 HF/训练导入、逐token一致和套件摘要。

协议仍为锁定 PyTorch2.14.1、CPU/batch1/完整裁窗 greedy；没有导出 KV cache、量化或 HTTP 服务。
支持范围以 macOS arm64 和 Linux x86_64 的验证为准，跨版本/加速器部署未作承诺。
摘要提供相对于审查清单的一致性，不是独立数字签名。

## 实际验收

- 干净提交 `254064a91864af10d866371a8d9693f10d7879d2` 完成真实 Qwen 重评。
  十个模型的1,080条记录全部与冻结 P5 原始记录一致；新增 pin 只加固来源，没有改变能力结论。
- P0–P5 六阶段统一 smoke 完成；P4/P5 在该检查中明确为 fixture。
- 独立安装来自同一干净代码提交，消费者只有已复制套件和准备好的依赖缓存，未借用源码环境同步项目。
- 全仓292项测试、lint、格式和 wheel/sdist 构建通过。新增回归覆盖 P6 遗漏、配置执行中修改、
  假完成回执、非法元数据、错误维度、缓存污染、独立交付校验与非覆盖行为。

[acceptance](reference/acceptance.json)保留来源、配置回执、六阶段状态及逐模型重放摘要。
加固验证时发现 HF 正常快照是指向 blob 的链接，初次严格路径检查不兼容；已修正为固定文件 allowlist
允许该布局、仍核验目标内容，增加链接目标污染回归。失败运行保留，未以修改期望摘要掩盖错误。

运行命令：

```sh
uv run --locked --all-extras kaleido verify --deep
uv run --locked --all-extras pytest
uv run --locked --all-extras ruff check .
uv run --locked --all-extras ruff format --check .
uv build
```

审查清单由维护者明确封存，校验过程不会自动更新期望值。
历史 P0–P6 原始证据保持原样，最终报告与套件作为新增证据封存。

## 最终范围与发布边界

本轮实现闭环交付，不把 SVM/集成、GAN/扩散、DPO/PPO、量化、投机采样等未来研究项标为完成。
P3 重训非位级一致、GRPO 未证明稳定迁移收益、Judge 校准失败的结论继续保留。
没有为“完成版本”重训到更好分数，也没有自动执行未知代码或调用付费评判服务。

**v1.0.0 仍是待审阅的发布候选，本轮未公开 Release、未创建发布 tag、未上传 PyPI。**
最终修复合并到 main 后，手动发布流程会重新验收并创建 draft；公开发布是独立维护者动作。
合并前可直接审阅并下载 CI 的套件资产，不需要先公开发布。
