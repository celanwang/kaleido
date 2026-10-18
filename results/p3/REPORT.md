# P3 验收：从字节BPE到真实预训练与缓存解码

日期：2026-10-19（Asia/Shanghai）。正式三种子实验来自干净代码提交
[`cee0ef8`](https://github.com/celanwang/kaleido/commit/cee0ef8f2e3f64b9d9504da864a088f767dfe02d)。
源码/配置/依赖锁 SHA-256：`612e2fc7d6a563fd7cda29b5b5c9ccae1aab892536713c747807a6dac9f6647d`。
运行原始记录位于 [reference/](reference/)，包括三份实际训练权重，未以smoke测试代替训练。

## 阶段出口与验证

| 模块 | 交付 | 证据 |
|---|---|---|
| 数据与BPE | 有完整声明和指纹的原文，整章划分、去重与近重复审计；字节fallback、确定合并、严格保存/加载 | 4项来源/边界、Unicode、EOS、序列化与划分测试 |
| Decoder/cache | 显式pre-LN多头Decoder、GELU、绝对位置、共享embedding、带offset的KV cache | 5项SDPA前向/参数梯度、因果性、跨窗生成、微型过拟合与模型回放测试 |
| 训练/评测 | 资源测量、随机章内窗口、AdamW、验证选模、逐token加权评测、词频/bigram基线 | 3项target错位/padding、加权统计、快照校验测试 |
| 完整接线 | 可配置训练、固定提示、重复计时、CPU种子复现、全局模式恢复 | 7项小预算端到端与边界测试 |
| 正式归档 | 完整tokens与模型，重算指标和CPU加载MPS权重 | 5项冻结语料/基线/每token指标与三份权重验证 |

P3新增24项，累计 **209项测试通过，无跳过**。Ruff check/format、包构建和全部阶段CLI通过。
CI使用 `p3_smoke.json` 做接线检查，同时直接加载本目录正式模型复算测试NLL；
1000步×3的阶段完成证据是本目录权重、历史与每token损失。

## 语料与数据边界

三部原文为Alice、Through the Looking-Glass、Wizard of Oz，完整来源、作者、原始声明与许可见
[data/p3](../../data/p3/README.md)。原文快照随仓库固定，运行不访问网络。
源页面标注美国公有领域，不据此作所有地区版权状态的结论。

按真实章节提取48个文档；删除2段重复长段落，无重复整章。
按书分层固定40/4/4章，UTF-8正文分别427379/58683/40227字节。
跨划分最大5词shingle Jaccard为0.003254；阈值0.8仅排查近乎重复，不证明没有短语或语义重叠。
这是同书未见章节测试，不是未见作者/书籍的泛化实验。

BPE只拟合训练40章，256字节+EOS+255个合并构成512词表。
训练/验证/测试target数241323/33566/23061，含章末EOS，不含章首EOS与padding。
每个章独立训练，不跨章拼接；测试按固定128-token块重置上下文，每个target仅计一次。
同一tokenizer/窗口/EOS协议下比较NLL和困惑度；另报告含EOS成本的bits/UTF-8 byte。

## 资源实测与正式预算

本机M4 Max、48GB；模型478720参数，参数float32体积1914880字节。
同batch=16、context=128，warmup2步、实测5步的中位训练步吞吐：

| 候选 | target token/秒 |
|---|---:|
| CPU单线程 | 51216 |
| CPU四线程 | 85179 |
| MPS | 276724 |

正式选择MPS。probe的进程峰值RSS约715MB，包含解释器、依赖和多个候选，**不是模型专属峰值**。
MPS当前分配约7.69MB、driver分配约116.20MB，均保留原始字段，不能与RSS混为同一口径。

模型为2层、128宽、4头、context128、共享embedding、dropout0；不加载预训练模型。
每种子1000步，lr8e-4、warmup50、cosine降到10%，AdamW betas0.9/0.95、decay0.01、grad norm裁剪1。
真实更新target共6143697，来自同一241323-token训练集的重复随机窗口，不是614万独立语料token。
每种子含验证/选模训练约9.7–9.9秒，完整三种子流程约40.51秒；不含依赖导入，不作跨设备性能基准。

## 留出集结果

验证每100步一次，三种子最低验证NLL均出现在1000步；没有利用测试集选择检查点或超参数。

| 条件 | 测试NLL | 困惑度 | bits/UTF-8 byte |
|---|---:|---:|---:|
| 训练频率unigram | 4.203430 | 66.9155 | 3.47647 |
| 训练转移bigram | 2.984450 | 19.7756 | 2.46831 |
| Decoder seed42 | 2.776067 | 16.0558 | 2.29596 |
| Decoder seed7 | 2.773709 | 16.0179 | 2.29401 |
| Decoder seed2026 | 2.802523 | 16.4862 | 2.31784 |

随机初始化模型的测试NLL为6.245–6.283。训练后平均NLL=2.784100、样本标准差0.015999；
平均困惑度16.1866、标准差0.2601。这是同一测试划分上的初始化/采样变化，不是总体置信区间。
所有正式模型低于bigram基线，说明在该预算/协议下优于这两个固定计数参考。
没有做去上下文消融，不能把收益单独归因于长程依赖，更不能推断问答、推理或指令遵循能力。

## 生成质量的实际限制

固定三条提示，每个种子temperature0.8/top_k40生成48个token，保留完整ID、seed与解码结果。
例如seed7的一段续写包含：

```text
the more. She looked her a frowing a ffick, and she bit herself, which roud she loo
```

能观察到词语、标点与局部句式，也有伪词、重复和不连贯。NLL下降不等于高质量长文本。
这里只训练了不到半MB正文、约48万参数，不对通用语言模型能力作结论。

## KV cache：正确性与实测成本

完整/chunk cache logits在MPS float32的最大绝对差为1.24×10⁻⁵，测试路径greedy一致；
同设备加载NPZ权重后logits差为0。32-token、batch1的cache体积65536字节，随长度/批量线性增长。
三份模型另在CPU重算全部23061个测试target，和MPS逐token损失在明确浮点容差内一致。

推理计时使用CPU单线程、batch1，固定greedy长度、无EOS提前退出，先warmup2次再测5次，
每次从空cache开始，包含prefill和实际窗口重建。下表为中位数：

| seed | 未超窗：朴素/缓存，32token | 加速比 | 临界窗口：朴素/缓存，16token | 加速比 |
|---|---:|---:|---:|---:|
| 42 | 21.076/5.817 ms | 3.623× | 20.439/19.095 ms | 1.070× |
| 7 | 16.197/5.460 ms | 2.967× | 18.873/16.093 ms | 1.173× |
| 2026 | 10.981/5.455 ms | 2.013× | 22.068/17.442 ms | 1.265× |

临界窗口每次生成需14次cache重建；learned绝对位置下直接丢弃旧K/V会改变语义，不能为速度省略。
朴素/缓存的最终greedy ID完全一致。短任务计时受频率与系统负载影响，且本次顺序测量未随机交错；
保留全部时间样本，范围只描述此机器/负载，不承诺普遍加速，更不把MPS训练吞吐当作生成吞吐。

## 复现与归档

```sh
uv sync --locked --all-extras
uv run --locked --all-extras pytest
uv run --locked --all-extras python -m kaleido.demos.language_model
```

正式配置为 `configs/p3_pretrain.json`，`device=auto`可能随负载选择不同设备。
要重做同设备实验，将配置复制后固定 `device=mps`；跨设备/版本不保证训练逐位一致。
CPU小预算完整重跑已测试一致；MPS权重使用CPU逐token复算验证，未声称跨后端训练轨迹逐位相同。
Python3.12.15、NumPy2.5.4、PyTorch2.14.1；精确依赖见uv.lock。

| 产物 | 内容 |
|---|---|
| [corpus.json](reference/corpus.json)、[tokenizer.json](reference/tokenizer.json)、[tokens.npz](reference/tokens.npz) | 文档审计、train-only合并与完整token序列 |
| [metrics.json](reference/metrics.json)、[histories.json](reference/histories.json)、[baselines.json](reference/baselines.json) | 真实token预算、初始/选中指标、验证曲线与参考 |
| [model-42.npz](reference/model-42.npz)、[model-7.npz](reference/model-7.npz)、[model-2026.npz](reference/model-2026.npz) | 三份真实推理权重，邻接JSON逐数组校验 |
| [evaluation.npz](reference/evaluation.npz)、[evaluation.json](reference/evaluation.json) | 初始/最终逐token NLL、窗口/章节求和与哈希 |
| [generations.json](reference/generations.json)、[benchmarks.json](reference/benchmarks.json)、[checks.json](reference/checks.json) | 原样生成、所有计时、cache/重载误差 |
| [resources.json](reference/resources.json)、[manifest.json](reference/manifest.json) | 候选设备实测、干净代码版本、完整配置和环境 |

NPZ用 `allow_pickle=False` 读取。权重仅用于推理，不含优化器/RNG的中间训练恢复状态。
NumPy core保持冻结；P3完成，下一阶段在固定Qwen检查点上建立SFT→GRPO与独立任务评测闭环，
不把本阶段小模型或手写BPE替换为Qwen后训练基座。见 [路线图](../../docs/ROADMAP.md)。
