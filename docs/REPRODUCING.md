# 复现、审计与本地部署

研究环境要求 Python 3.12、uv 0.12.23。所有命令在完整 checkout 根目录执行：

```sh
uv sync --locked --all-extras
uv run --locked --all-extras kaleido doctor
uv run --locked --all-extras kaleido verify --deep
uv run --locked --all-extras pytest
```

`doctor` 检查直接依赖版本（允许同版本 `+cpu` 构建标签）；传递依赖由 uv.lock 锁定。
`verify` 不下载、不重建清单，对照 `artifacts/inventory-v1.json` 校验数据、配置和 P0–P5 历史工件。
摘要错误或缺文件应先调查，不应直接执行封存脚本来消除错误。

## 分层运行

```sh
uv run --locked --all-extras kaleido list
uv run --locked --all-extras kaleido reproduce --profile smoke
uv run --locked --all-extras kaleido reproduce --profile formal
uv run --locked --all-extras kaleido run p3 --profile formal --output runs/my-p3
```

smoke 对 P0–P2 使用原小预算，P3 使用缩小的预训练配置，P4/P5 明确为随机 fixture。
formal 使用原正式预算，P3 从零训练三个 Decoder，P4 训练三种子的 SFT/GRPO/筛选监督，
需要约1GB固定 Qwen 基座缓存；P5 重评仓库中发布的 P3/P4 工件，不自动改用同轮新模型。
每次运行新建唯一目录，不覆盖旧结果；`--config PATH` 可覆盖单阶段配置，实际文件摘要与配置进入回执。
自定义配置的 formal 模式不意味着它与已发表的标准预算相同，应核对配置回执。

统一复跑目录保存总 manifest、steps、逐阶段 stdout/stderr 及各阶段完整工件。
可以在运行中查看 `p4.stdout.log`。非零退出会保留证据并停止后续阶段；
SIGKILL/断电可能留下 running 记录，须视为未完成。没有实现优化器/RNG 断点恢复，不能当作成功续训。

固定随机种子并非位级训练复现承诺。P6 本机复跑中 P3 NLL 接近历史值，但权重摘要不同，
实际新权重与差异见 P6 验收报告。P0–P5 旧证据保持原样，不用新结果覆盖它们。
用以下命令生成显式对照，新权重或 trace 不同时会保留到 `changed/`：

```sh
uv run --locked --all-extras python scripts/summarize_reproduction.py --suite runs/p6-formal/RUN_ID --output runs/replay-summary/reproduction.json
```

## 导出与推理

```sh
uv run --locked --all-extras kaleido export --weights results/p3/reference/model-42.npz --tokenizer results/p3/reference/tokenizer.json --output runs/decoder-42
uv run --locked --all-extras kaleido predict --bundle runs/decoder-42
```

向最后一个命令输入一行 `{"prompt":"Alice was","max_new_tokens":12}`，它会输出一行 JSON。
导出前验证 logits 和生成 token；已有输出目录拒绝覆盖，加载前验证 bundle 文件集合和摘要。
本地协议是 CPU/batch1/full-forward greedy，动态长度 1..128，超过 context 时裁窗，
不包含 KV cache、量化或 HTTP 服务。单行≤64KiB，prompt≤16KiB，新token预算0..128，
错误请求独立返回错误并继续处理，任何失败使退出码非零。

## 独立安装验证

```sh
uv build
uv run --locked --all-extras python scripts/check_wheel.py --wheel dist/kaleido-1.0.0-py3-none-any.whl --bundle runs/decoder-42 --report runs/wheel-check.json
```

检查器建立新 venv，用 `UV_PROJECT_ENVIRONMENT` 明确指定目标，按 uv.lock 离线同步运行时，
不装 editable 项目，再安装实际 wheel；从仓库外以 `python -I` 推理并逐 token 对照。
需要先同步完整研究环境以准备依赖 wheel 缓存；冷缓存安装阶段可以联网，推理不依赖网络或 HF 包。
仅安装 wheel 不包含语料/历史实验：研究命令需要 checkout，本地 `predict` 不需要。

## 发布状态与边界

v1.0.0 是当前发布候选。功能 PR 合并后，可在 main 手动运行 `prepare release draft`，
输入与 pyproject 一致的新 tag。流程重新测试、smoke、导出和独立 wheel 推理，生成 wheel、sdist、
Decoder bundle、验证报告及 SHA256SUMS；只创建 draft，不自动公开或上传 PyPI，也不覆盖既有 tag。
正式训练与历史结论以各阶段报告为准，CI smoke 不能代替正式训练证据。

导出格式仅验证锁定的 PyTorch 2.14.1，跨版本/跨框架/加速器部署未作承诺。
未实施的算法与推理扩展见 CHANGELOG；GRPO 未证明稳定迁移收益、Judge 校准失败的结论继续保留。
