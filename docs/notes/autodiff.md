# 自动微分：将链式法则变成反向图遍历

目标：对标量损失 L 求所有参数梯度。有限差分每个参数需要两次前向，适合验错；
反向模式从 L 的梯度 1 出发，沿图传播 vector-Jacobian product（VJP），避免构造完整 Jacobian。

例如 C=AB，上游梯度 G 给出 dA=GBᵀ、dB=AᵀG；均方误差给出
∇W=2Xᵀ(XW+b−y)/n。测试用独立 NumPy 解析公式对照，并逐坐标做中心差分。

广播的前向复制是概念上的展开，反向要把复制位置的梯度求和：先消去增加的前导维度，
再沿原先大小为 1 的轴求和。形状正确不意味着数值正确，因此测试覆盖多轴广播与标量。

计算图是 DAG：共享节点需汇总所有上游贡献，按反向拓扑顺序传播。
每次 backward 使用新的局部 adjoint 表，最终累积到 `.grad`，避免第二次反传把旧中间梯度再乘一次。
`zero_grad()` 只清当前张量；每步以新叶子张量更新参数，数值不可原地修改，旧图仍引用原前向值。

支持：实数 float64、加减乘除、常数幂、二维矩阵乘、sum/mean、reshape/T、exp/log/ReLU。
非标量 backward 必须传同形状 seed。ReLU 在 0 选子梯度 0，数值检查避开折点。
拒绝非有限数值、非法 log、非二维 matmul；不做高阶导数、索引反传、GPU 或通用深度学习框架。

验证：`uv run --locked pytest tests/test_tensor.py`；真实训练见回归 demo。
测试覆盖 NumPy 前向、中心差分、线性回归与两层网络解析梯度、共享图、重复 backward、
3000 层图和边界条件。有限差分只能证明采样点附近的一阶一致性，不能代替结构和边界审查。

数学联系：矩阵乘对应线性映射，反传对应其转置；sum 的反向是广播，广播的反向是 sum；
概率模型的负对数似然常通过 log 与 exp 构建，但稳定 softmax/logsumexp 留到分类模块实现。

参考：[micrograd](https://github.com/karpathy/micrograd) 的标量反向图、
[NumPy 广播规则](https://numpy.org/doc/stable/user/basics.broadcasting.html)。
