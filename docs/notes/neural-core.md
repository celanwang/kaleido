# 从张量到可训练模块

Parameter 身份固定，持有的 Tensor 前向值不可变。优化器替换叶子 Tensor，旧计算图仍引用原值；
梯度保留直到显式 zero_grad，与常用优化器的累积语义一致。模型按属性和列表递归枚举参数，
共享参数只枚举一次；state_dict 保存独立副本，加载先检查所有 key/shape/有限值再整体更新。
不提供设备管理、动态图编译、混合精度或完整 PyTorch API。

Linear/MLP 沿现有二维矩阵乘演进。新增索引 VJP 用 add.at 汇总重复索引，索引也复制避免图被改写。
多分类交叉熵使用 logsumexp，不先算概率再 log；softmax 的 VJP 为 p·(g−Σgp)。
显式 bool mask 中 True 表示可见，每行至少一个可见元素，避免全屏蔽行产生 NaN。

SGD 支持动量，Adam 显式一阶/二阶矩与偏置修正，eps 在平方根之外。
weight_decay 为耦合 L2，不是 AdamW。缺少梯度的参数不更新也不推进自己的 Adam 时钟。
所有参数更新先计算、验证，再整体提交；非有限梯度不会留下半次更新。

运行：`uv run --locked --all-extras pytest tests/test_deep_core.py`。
测试与 PyTorch CPU float64 对照 MLP 前向/梯度/Adam 单步、SGD/Adam 多步和缺失梯度。
真实训练入口为 `python -m kaleido.demos.deep_learning` 的 Digits MLP/CNN 实验。
参考：[Adam 官方文档](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html)。
