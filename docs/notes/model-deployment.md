# 从训练快照到可验证的部署工件

NPZ 检查点需要模型结构代码；P6 使用 `torch.export` 导出训练后 P3 Decoder 的 CPU float32
张量计算图，再打包 byte-BPE、生成协议、校验清单和样例。消费者只加载图与 tokenizer，
不导入 `DecoderLM` 或训练循环，也不需要 checkout、语料、Qwen 权重或 HF 网络访问。

导出边界是 batch=1、动态长度 1..128、完整前向、greedy。输入合法性在运行时检查，
图只承担张量计算。每步对末尾 context 窗口重新计算位置，保持 P3 的裁窗语义；这没有导出 KV cache，
没有量化，也未声称加速。PyTorch 文档提示导出格式可能跨版本不兼容，因此运行时严格匹配
生产者的 PyTorch 基础版本，不把 `.pt2` 称为通用跨框架格式。

发布目录前先加载导出图，比较边界长度 logits 与 greedy，再对短提示和超过 context 的提示
生成逐 token 对照；全部通过后才发布目录，已有目录拒绝覆盖，失败清除临时目录。
加载时验证文件集合与逐文件摘要。摘要保证相对于清单的一致性，不提供独立签名真实性。

`kaleido predict --bundle PATH` 使用本地 stdin/stdout JSON-lines，每行一条请求与结果，
支持 `prompt` 和 `max_new_tokens`（0..128）。限制单行64KiB和prompt 16KiB，逐请求拒绝异常并
继续处理下一行；有任何失败时退出码非零。它是单进程本地部署接口，不是 HTTP 并发服务。

最终验收必须把 wheel 装到独立环境、移到仓库外运行，再与原模型逐 token 对比。
仅在 editable checkout 内 import 成功，不构成这个验收。

参考：[torch.export 计算图与序列化](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export.html)、
[导出 API 与版本限制](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export/api_reference.html)。
