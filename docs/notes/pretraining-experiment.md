# P3：训练与推理是一条证据链

原文快照和版权来源见 `data/p3/`；模型不读取源码仓库或评测结果作为语料。
BPE只在训练40章拟合，词表512，预处理后训练/验证/测试字节数427379/58683/40227。
每章独立编码为 EOS+正文+EOS；随机训练窗口按文档token数加权，不跨章，短窗口pad target为−100。
词频和bigram基线只用训练转移计数，α=0.1平滑，不在测试集估频。

资源探测以同形状真实采样batch进行前向/反传/裁剪/AdamW，warmup2步后测5步。
选择实测最快可用设备；固定device可覆盖auto。CPU的进程RSS包含解释器、导入与候选probe，
MPS记录current/driver allocation，都不冒称模型专属峰值。训练成本另含验证和选模。

默认模型478720参数，128context、2层、128宽、4头、共享embedding、dropout0、float32。
seed=42/7/2026，各1000步×16×128的名义预算，记录扣除padding后的真实target数。
AdamW lr=8e-4，warmup50步，cosine降至10%，betas=(0.9,0.95)，weight_decay0.01，norm裁剪1。
每100步用固定验证窗口选最低token加权NLL，test只作初始/最终报告，不用于调参。

评测每个章内target恰好一次、含EOS，不计BOS或padding。固定不重叠context块重置位置，
所以困惑度只能与相同tokenizer和窗口协议对照；附带含EOS成本的UTF-8 bits/byte。
保存每token NLL、每窗口/章节求和，避免对不同长度batch的均值再平均。

模型NPZ+JSON仅含推理参数与逐数组指纹；replay从seed完整重训，不声称能从中间步精确恢复。
生成提示固定，temperature0.8/top_k40/CPU独立generator。输出仅作样例，不是指令遵循或能力基准。
推理性能单独用CPU单线程batch1，warmup后5次固定greedy长度，包含prefill；
一组不超窗，一组接近context上限展示重建成本。不以MPS训练吞吐冒充生成吞吐。

`configs/p3_pretrain.json` 为正式实验，`p3_smoke.json` 仅CI快速接线验证，不作为阶段完成证据。
运行 `uv run --locked --all-extras python -m kaleido.demos.language_model`。
P3使用PyTorch成熟底层，P0–P2 NumPy core保持冻结。
