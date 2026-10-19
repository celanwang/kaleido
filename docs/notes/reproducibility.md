# 复现入口与可信边界

统一的是配置、来源和证据，而不是所有算法的训练 API。`kaleido run p0` 到 `p5` 分派到
原有配方，`--root` 显式选择源码 checkout；安装后的 wheel 不假装自带原始数据和实验工件。
`list` 与部署推理可以脱离仓库，研究复现命令需要有完整 checkout。

`reproduce --profile smoke` 逐阶段在独立进程执行接线检查，P4/P5 显式使用随机 fixture。
`--profile formal` 使用原正式预算，会训练 P3/P4 并需要固定 Qwen 权重缓存。
P5 始终评测发布的冻结 P3/P4 工件，不自动改用本轮同级目录里的新训练结果；重跑结果需要
分别比较，不能把目录排列冒充新的训练血缘。配置、模式和路径全部进入运行索引。

失败保留总 manifest、已完成步骤和 stdout/stderr；下一次运行创建新 ID，不覆盖失败证据。
这不是断点续训：原 P3/P4 快照未保存优化器/RNG 状态，没有声称可精确恢复中断训练。

`verify --deep` 对照已评审的 inventory 校验输入数据、配置、历史工件和依赖文件的 bytes/SHA256，
并校验有 sidecar 的 NPZ 数组 dtype/shape/内容哈希。校验绝不自动重建预期哈希。
`seal_inventory.py` 是维护者在审查变更后手动运行的封存工具；同一个可被改写的索引不提供外部真实性
证明，也不能代替科学正确性测试。缺文件、路径越界、摘要不符都返回失败。

`doctor` 检查完整复现所需 Python 与依赖版本，不下载模型或隐式探测设备；安装依赖仍使用
`uv sync --locked --all-extras`。锁文件是研究环境约束，wheel 的推理消费者只需其实际使用的依赖。

参考：[uv 项目与构建](https://docs.astral.sh/uv/guides/projects/)、
[锁定与同步](https://docs.astral.sh/uv/concepts/projects/sync/)。
