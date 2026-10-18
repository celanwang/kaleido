# P4验收：固定Qwen上的SFT、GRPO与采样筛选对照

日期：2026-10-19（Asia/Shanghai）。正式实验来自干净代码提交
[`8c5bb9a`](https://github.com/celanwang/kaleido/commit/8c5bb9a2e7d54aebfd16edb4855f382bc1f75064)。
源码/配置/依赖锁 SHA-256：`535f181f1070870fb7dedf12f9054cbb10f38e5ff4be2be422525565dfbf1e23`。
完整原始记录在 [reference/](reference/)。**闭环已实现，本次预算未证明GRPO稳定提升推理能力。**

## 交付与验证

| 模块 | 实际交付 | 测试 |
|---|---|---|
| 任务/判定 | 题族隔离、均衡标签、训练AST/parser与评测展开oracle/regex分开 | 3项数据、reward hack、截断、标签损坏与区间边界 |
| 原生Qwen/LoRA | 固定revision/权重SHA，原生模板、EOS/padding，冻结基座与q/v adapter | 4项零初始化、原生似然/生成、模板前缀与checkpoint |
| GRPO公式 | 显式组优势、old/ref detach、clipping、sampled KL、长度归一化 | 3项独立值/梯度、零方差、mask与detach |
| 完整配方 | SFT、在线GRPO、固定SFT rejection，再监督真实接受的生成ID | 10项离线接线、重跑、接受/空池、特殊token与配置边界 |
| 正式归档 | 三种子逐题评测、原生tokenizer、9份adapter、全部rollout与预算 | 6项冻结工件、独立重判、RNG预算回放、参考与选模 |

P4新增26项，累计 **235项测试通过、无跳过**；Ruff check/format、各阶段CLI和sdist/wheel通过。
CI不下载0.5B模型：随机tiny Qwen仅验证接线；正式效果依据实际Qwen运行，另离线审计其完整工件。
没有使用P3小模型或手写BPE冒充Qwen。模块笔记见 [实验协议](../../docs/notes/posttrain-experiment.md)。

## 同一基座、模板与资源

- 官方 `Qwen/Qwen2.5-0.5B-Instruct`，revision `7ae557604adf67be50417f59c2c2f167def9a775`。
- 原始safetensors SHA-256：`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`。
- 基座494032768参数冻结，540672个q/v LoRA参数可训练（约0.109%）；rank8、alpha16、全程float32。
- 保留原生tokenizer、chat template和EOS/pad配置，无remote code。原始配置/声明与压缩tokenizer归档。
- 本机M4 Max/48GB，MPS。probe热身后batch4单步中位0.07765秒，32个回答监督token。
  probe时MPS current约2.022GB、driver约2.999GB；这些不是整个训练过程的峰值显存。
- probe中原生generate与自写greedy相同；采样明确关闭原生默认repetition penalty/分布截断，
  所有条件使用同一评测设置，不将本结果与模型卡成绩直接比较。

基座约988MB下载到HF缓存，不提交仓库；原始 [Apache许可证](reference/QWEN-LICENSE.txt)随工件保留。

## 数据与实验条件

受控整数表达式求最小非负余数：训练4种AST结构，验证/测试各2种未见结构，128/32/32题。
每题族余数0–7均衡、模数8–17，常数答案在任一split只能得到12.5%。
平衡在冻结测试输出前完成。结构隔离不排除代数等价与基础运算重叠，不称为自然数学分布泛化。

训练奖励要求完整 `<answer>N</answer>`、数值正确且未截断；只去掉真正末尾EOS，
其他special token不会因decode而隐藏。训练与最终评测使用不同解析和数学求值路径，不执行生成代码。
宽松数值正确率只用于诊断。SFT仅监督回答及原生EOS，prompt/padding不计目标。

每个seed从同一基座开始，SFT64步×4题；选中SFT后，GRPO/rejection从同一adapter和新优化器分出。
两分支相同24轮×2题×4回答、同样seed和32-token上限；GRPO每批2次更新，rejection计划48次batch8更新。
RFT先用固定SFT收集完整采样池，核验、去重，再监督原样生成ID；不把gold答案塞进接受池。
所有检查点只按验证严格准确率选择，同分按回答teacher NLL，测试不反馈训练或超参数。

## 实际独立测试结果

每种子复用同一32道测试题，指标为验证选中检查点的greedy结果：

| seed | SFT | GRPO分支选中模型 | 采样筛选再监督 |
|---|---:|---:|---:|
| 42 | 5/32 =15.625% | 5/32 =15.625% | 3/32 =9.375% |
| 7 | 5/32 =15.625% | 5/32 =15.625% | 5/32 =15.625% |
| 2026 | 3/32 =9.375% | 7/32 =21.875% | 6/32 =18.750% |
| 三种子算术均值 | 13.542% | 17.708% | 14.583% |

基座严格准确率0%、格式率0%、宽松数值正确率3.125%；后训练选中模型格式率均100%、截断率0。
因此可以确认格式适配，但不能把严格准确率从0起算的变化全部解释为算术能力增长。
SFT均值仅略高于12.5%常数基线；GRPO均值优势由一个种子贡献，另外两个选回SFT起点。

例如5/32的Wilson95%区间为[6.86%,31.75%]，7/32为[11.02%,38.76%]，明显重叠。
区间也未校正模板相关性；三个seed不是96个独立题，不能合并后声称显著。
本实验没有证明方法排序、稳定推理提升或更广任务泛化。

## 选模、零信号与失败分析

| seed | SFT选中步 | GRPO选中轮（0=SFT起点） | RFT选中步（0=SFT起点） |
|---|---:|---:|---:|
| 42 | 48 | 0 | 40 |
| 7 | 48 | 0 | 0 |
| 2026 | 48 | 20 | 24 |

GRPO在每seed都实际执行48次优化器更新尝试，没有因选回SFT而省略训练。
原始rollout的48个题组统计：

| seed | 全错组 | 全对组 | 混合奖励组 | 零方差组比例 |
|---|---:|---:|---:|---:|
| 42 | 32 | 2 | 14 | 70.833% |
| 7 | 35 | 2 | 11 | 77.083% |
| 2026 | 17 | 7 | 24 | 50.000% |

全错/全对组的相对优势为0，只有混合组提供组内区分信号；KL仍可约束偏移。
seed7有81/192条采样解析成答案5，而当轮题目中正确答案5仅5/48组，说明输出分布偏斜，
不能用训练奖励某段上升代替持出评测。不同轮的题目也不同，前后奖励均值不是同一固定探针。
seed2026相对SFT在测试上新增5道正确、丢失1道；RFT seed42新增3道、丢失5道。
配对变化保存在 [analysis.json](reference/analysis.json)，包含负变化而非只列平均收益。
seed2026的选中GRPO测试严格准确率提高，但回答teacher NLL由0.3363升至0.5048，
进一步说明单个准确率不能替代概率质量和稳定性评估。

GRPO的population组标准差+1e−4、per-completion mean、clip0.2、β0.04与sampled KL公式明确记录。
冻结ref是SFT检查点，old在同批2次更新内不变；不是当前TRL默认配置的隐式替代。
采样temperature1、无top-k/top-p/repetition penalty，使behavior与raw policy likelihood口径一致。

## 采样筛选对照与真实成本

| seed | 原始接受/192 | 去重接受 | GRPO生成/更新token | RFT生成/更新token |
|---|---:|---:|---:|---:|
| 42 | 32 | 19 | 1538 /3076 | 1538 /3072 |
| 7 | 26 | 15 | 1536 /3072 | 1536 /3072 |
| 2026 | 58 | 26 | 1535 /3070 | 1540 /3072 |

三种子SFT共6144个回答监督token；GRPO9218，RFT9216。
两后训练分支的生成上限、题目顺序和计划更新数匹配，实际token预算接近，但不是相同总计算：
GRPO还需old/ref前向，RFT固定SFT采样且重放去重池，信息分布与算力代价不同。
三种子含验证的配方耗时：SFT28.29秒、GRPO70.17秒、RFT57.58秒；
缓存已热的正式流程含模型加载、probe、基座/最终评测和归档约174.46秒，
不含此前首次权重下载或依赖安装。仅为这台机器的观察。

## 归档、核验与复现

```sh
uv sync --locked --all-extras
uv run --locked --all-extras pytest
uv run --locked --all-extras python -m kaleido.demos.post_training
```

Python3.12.15 / PyTorch2.14.1 / Transformers5.20.1 / HF Hub2.2.0 / Safetensors0.8.0。
首次下载固定基座约1GB，后续优先本地缓存；正式配置为 `configs/p4_posttrain.json`。
CPU fixture重跑一致已测试；正式MPS记录seed和环境，不承诺跨设备训练轨迹或采样逐位相同。
adapter是推理快照，不含优化器/RNG中途恢复状态。

| 产物 | 内容 |
|---|---|
| [model_source.json](reference/model_source.json)、[base_config.json](reference/base_config.json)、[base_generation_config.json](reference/base_generation_config.json) | 原始revision/权重SHA/模板指纹与配置 |
| [tokenizer/](reference/tokenizer/)、[prompts.json](reference/prompts.json) | gzip原生tokenizer、逐文件hash与全部原生prompt ID |
| [tasks.json](reference/tasks.json)、[base_evaluation.json](reference/base_evaluation.json)、[baselines.json](reference/baselines.json) | 固定题目、标签分布、冻结基座与常数参考 |
| [metrics.json](reference/metrics.json)、[test_samples.json](reference/test_samples.json)、[validation_samples.json](reference/validation_samples.json) | 指标、区间、每题生成与选模证据 |
| [rollouts.json](reference/rollouts.json)、[histories.json](reference/histories.json)、[analysis.json](reference/analysis.json) | 所有组奖励、token ID、KL/clip/梯度、配对变化 |
| [sft-42.npz](reference/sft-42.npz)、[grpo-42.npz](reference/grpo-42.npz)、[rejection-42.npz](reference/rejection-42.npz) | 示例seed三分支adapter；其余7/2026同目录，邻接JSON完整校验 |
| [checks.json](reference/checks.json)、[resources.json](reference/resources.json)、[manifest.json](reference/manifest.json) | 冻结/日程/参考关系、资源与干净代码版本 |
| [replay.json](reference/replay.json) | 发布adapter+发布tokenizer重新装入固定基座后的9组完整测试重放 |

离线归档测试重建原生tokenizer、独立重判全部评测/rollout，验证9份adapter与共同SFT来源，
并回放采样RNG复算生成/更新token和选模规则。另在本机重新加载固定真实基座、发布adapter及压缩tokenizer，
9组测试的greedy token ID与评分全部一致。CI不重跑1GB基座前向，不能据此声称完成CPU模型效果复验；
完整重放仍需锁定revision的基座。

P4按路线图闭环完成，但结论保持有限；P5将统一评测入口、配对/不确定性分析和校准，
以这份负结果检验评测框架。NumPy core继续冻结，代码执行任务仍需独立沙箱方案。
