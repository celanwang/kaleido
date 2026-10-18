# 注意力：可学习的内容寻址

实现 softmax(QKᵀ/√d)V。点积方差随维度增长，缩放控制 logits 的量级；
softmax 产生权重，V 承载聚合内容。Q/K/V 分工体现“如何匹配”和“取回什么”的分离。

本阶段单序列、单头、二维 Q/K/V；SelfAttention 包含四个可训练投影。
因果 bool mask 直接进入稳定 softmax，被遮挡项权重与梯度严格为零。
测试既比较 PyTorch 前向与三个输入梯度，也改变未来位置并检查过去输出不变。
不靠很大的有限负常数近似屏蔽，也不允许全屏蔽行。

这不是完整 Transformer：没有多头、位置编码、归一化、前馈块、KV cache。
这些由 P3 的具体语言建模任务驱动，不提前扩展通用框架。
测试：`uv run --locked --all-extras pytest tests/test_attention.py`；
`deep_learning` demo 保存注意力矩阵、前缀扰动与梯度参考误差。
参考：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)。
