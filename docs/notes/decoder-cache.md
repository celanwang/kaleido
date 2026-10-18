# Decoder与KV cache：先相同，再比较快慢

PyTorch只提供张量、自动微分与优化器。模型从随机初始化开始，显式实现pre-LN残差块、
多头Q/K/V、causal softmax、GELU前馈、learned绝对位置和共享输入/输出embedding。
没有加载预训练权重，也没有使用Transformer黑盒模块。有限规模训练使用float32。

训练输入与target错位一位，EOS既作为章首起点也作为章尾目标。padding target为−100，
不把padding和真实EOS混为一个训练目标；batch行不跨章。评测每个target仅计一次，
长章按固定context块重置位置；困惑度必须连同tokenizer、EOS和块策略说明。

KV cache保存每层已计算的K/V，新增query的因果mask必须带past offset。
多token chunk的mask同样带offset，不能误用无偏移的is_causal=True。
learned绝对位置下，超窗后不能只丢弃旧K/V：旧值仍包含原位置及更早上下文。
达到context上限时从最近窗口完整重建cache，使语义与朴素截窗解码一致；报告重建次数。

参考对照SDPA前向/梯度、完整与chunk cache logits、未来扰动、跨窗greedy与固定seed采样。
基准固定greedy输出长度，禁EOS提前结束，每次重新prefill，warmup后多次测量，并同步设备。
结果包含prefill+解码与cache重建，不是只挑最短内核时间，也不预设cache必定加速。
测试：`uv run --locked --all-extras pytest tests/test_decoder.py`。
