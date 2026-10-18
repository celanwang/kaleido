# 同一Qwen基座上的低秩更新

官方Qwen2.5-0.5B-Instruct固定40位revision与safetensors SHA-256；不加载远端Python代码。
原生tokenizer、chat template和EOS/pad配置保留，不沿用P3手写BPE。
显式请求chat template返回token列表，并验证完整assistant序列的前缀与generation prompt相同。
SFT只监督assistant回答及第一个原生EOS，不将prompt或EOS后的模板换行算进目标。

冻结所有基座参数，在每层q_proj/v_proj添加BA·α/r，A随机、B零初始化，初始输出与基座完全一致。
模型和adapter使用float32，优化器仅接收adapter；无dropout，eval模式下仍可反传。
回答似然按shift一位的原生logits计算；left-pad prompt、right-pad completion，mask按长度而非pad ID构造，
避免pad与EOS重合时丢掉真实终止监督。原生logits_to_keep只计算回答位置的词表投影。

自写batch采样仍使用原生Qwen前向与KV cache；测试与原生generate的greedy输出对照。
快照只保存adapter及逐数组hash，重放仍需固定基座；不把1GB基座权重提交仓库。
测试含零初始化、冻结权重、完整原生似然对照、EOS/padding和adapter保存/加载。
