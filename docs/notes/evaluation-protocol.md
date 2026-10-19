# 评测协议：先固定问题，再看结果

P5 不训练或重选 P4 检查点。不同模型共享题目 ID、提示、原生对话模板和生成预算。
生成 adapter 构造提示时只传问题与 system；参考答案不进入提示。archive adapter 只能重放原协议，
不能把旧 32-token 输出伪装为新 64-token 实验。缺失、重复或额外 ID 都终止运行。
所有指标重新解析原始文本，不信任旧的 `correct` 字段。

数值题严格要求完整 `<answer>NUMBER</answer>` 且无截断；另外记录内容正确率、
格式率和截断率，以区分答案与格式收益。Decimal 比较保留全部限定精度，不用浮点容差。
上下文颜色检索 v2 在提示中明确要求单一颜色词，评分仅接受该格式；单一颜色命中另作内容诊断。
拒绝否定句等自由文本，避免把提及 gold 当作事实正确。这仍是窄域诊断，不等于一般事实性。

新增结构算术用 AST 与独立展开式交叉验证，并对 P4 全部题目检查公式及 AST 结构隔离。
GSM8K 固定上游 revision、原始文件哈希与 MIT 许可，再按固定 RNG 选择 32 道整数题。
仅向模型传问题，既不传原始解题过程也不传 gold。这是自定义直接回答子集，
不能称为完整 GSM8K 或标准 CoT 分数；无法证明基座预训练没有见过公开题。

每个种子内用精确双侧 McNemar 检查同题胜负；预先定义的比较集合用 Holm 校正。
跨种子均值的区间按题目重采样，同一道题保留三个固定种子的所有结果，按题族内重采样，
不把 32 题 × 3 种子当作 96 个独立样本。该 percentile 区间条件于已选种子和题族，
不估计未见题族或训练随机性的总体不确定性。P3 则按完整章节重采样，以 token 总数加权
NLL，禁止独立重采样相邻窗口；四个相关章节的区间只能作为描述。

Judge 使用独立的规则 gold 校准/审计对，交换 A/B 顺序，记录无效输出与顺序不一致。
预设门槛：agreement ≥ 85%、parse ≥ 95%、swap consistency ≥ 90%；校准与审计均须通过。
失败则关闭模型排名。规则 gold 不冒充人工标注；即使通过，也只适用于给定参考的颜色问答，
不外推为一般人类偏好。与被测模型同源的小 Qwen Judge 也不构成独立能力证据。

参考：[原始 GSM8K](https://github.com/openai/grade-school-math)、
[MT-Bench Judge 偏差研究](https://arxiv.org/abs/2306.05685)、
[bootstrap 定义](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)、
[精确 McNemar](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html)。

协议修复记录：v1 只要求 concise，却拒绝 `azure stones` 等合理短答，造成任意格式惩罚。
该运行不用于质量排名，原始证据保存在 `results/p5/protocol-audit/context-v1.json`。
v2 明确颜色词约定并收紧为精确颜色解析；题目、模型与训练设置不变，重新运行全部面板。
修复发生在查看 v1 输出后，因此 v2 是描述性协议验证，不能宣称未经观察的确认性测试。
数值题与 Judge 协议未根据成绩调整。
