# P4实验协议

固定官方Qwen检查点与原生template/tokenizer，float32、q/v LoRA rank8/alpha16。
先在训练题探测生成/更新成本，恢复零输出adapter后冻结基座验证/测试记录；
正式三种子只改变adapter初始化、训练采样和rollout RNG，基座和192题保持相同。

SFT采样64个batch×4题，lr2e−4；按验证严格准确率选检查点，同分看回答teacher NLL。
GRPO和rejection分支都从该SFT快照开始、重置优化器，lr5e−5。
每分支24轮×2题×4回答，上限32生成token；GRPO每批2次更新，rejection也计划48次batch8更新。
RFT整个采样阶段固定SFT策略，按独立核验通过的生成ID去重，再监督这些原样生成序列，
不以规则gold答案替换被筛选文本；没有接受样本则明确记录未更新。

两分支使用相同题目顺序、组数、生成seed和token上限；GRPO随策略更新而采样，RFT策略固定。
因此实际完成长度、更新target数、接受样本数和耗时仍不同，必须报告，不能称为等总计算的最优方法比较。
所有评测greedy、原始logits无repetition penalty；采样temperature1且不截断分布。
原生生成默认的采样/惩罚参数被显式覆盖，模板与tokenizer不变；probe验证此设置下与原生generate一致。

验证每16个SFT步、每4个GRPO轮、每8个RFT步执行；测试只作冻结基座和各分支选中模型的最终报告。
严格准确率、数值正确率、格式率和截断率分开，记录每个回答的token ID与解析结果。
参考模型为SFT快照，非原始基座；零方差组、sampled KL、clip fraction和grad norm每轮保存。

单seed测试仅32题，Wilson区间按题报告；不同seed复用同一测试集，不作为96个独立题计算置信区间。
label均衡是受控生成分布，不是自然数学题基准；格式适配不能冒充推理能力增长。
原文基座约1GB从固定revision下载到HF缓存，仓库只存adapter、压缩原生tokenizer和逐题证据。

正式运行：`uv run --locked --all-extras python -m kaleido.demos.post_training`。
CI仅用显式 `--fixture --config configs/p4_smoke.json` 的随机tiny Qwen，不能作为正式后训练效果证据。
