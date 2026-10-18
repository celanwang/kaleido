# 实验记录：从首个 demo 开始

问题：一条 loss 曲线无法说明结果来自哪个数据、配置或代码。P0 先建立本地记录，避免后期补记。

每次运行在 `runs/<UTC 时间>-<随机 ID>/` 生成独立目录，不覆盖历史。
`manifest.json` 记录配置、命令、Python/NumPy/平台、Git revision/dirty 状态，以及源码、配置、
依赖锁文件的 SHA-256。Git 不可用时保留 null，不能伪造“干净版本”。
实际数据另外记录包含 dtype/shape/content 的 SHA-256，不能只记录 seed。

`Run` 上下文只在正常退出时标记 completed，异常标记 failed 并重新抛出。
指标拒绝 NaN/Inf；写 JSON 先生成临时文件再原子替换，避免留下半个指标文件。
进程被强制杀死可能停留在 running，应视为未完成。环境记录用于追溯，不承诺跨平台逐位一致。

验证：`uv run --locked pytest tests/test_experiments.py` 检查成功/失败、重复运行不覆盖、
非有限指标与数据指纹。真实使用见 `python -m kaleido.demos.linear_regression`（P0 第三模块）。

取舍：文件协议无需服务；此时不引入通用 Trainer、数据库或插件系统。
源码指纹覆盖 `src/`、`configs/`、pyproject、uv.lock、Python 版本文件；不替代完整 Git 历史或数据归档。
