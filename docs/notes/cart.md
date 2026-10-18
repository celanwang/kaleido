# CART：从线性边界到分段区域

分类树反复选择某一特征的阈值，把空间切成轴对齐区域；叶子输出训练类别频率。
节点不纯度为 Gini=1−Σp²。枚举每列排序后相邻不同值之间的切点，
用前缀类别计数计算加权子节点不纯度，选择最小值；不重复枚举相同特征值。

`min_impurity_decrease` 使用节点样本数/全体样本数加权的下降，与 sklearn 定义一致。
`max_depth` 与 `min_samples_leaf` 控制容量，不做代价复杂度后剪枝、缺失值或样本权重。
递归深度显式限制为 64，适合本项目的小数据案例；排序会在每个节点重复，未做工业优化。

一个容易写错的边界是 XOR：根节点切任一轴都没有即时 Gini 收益，但下一层可以完美分类。
默认允许零收益切分，正的 impurity 门槛则可能完全阻断 XOR。
与之相比，原始两特征上的逻辑回归无法表达 XOR。这个例子说明表示能力与局部贪心规则的取舍。

阈值使用避免加法溢出的中点，相邻浮点数若没有可表示中点则退到左值。
同分优先列顺序、切点顺序，标签同票取排序后首类。sklearn 的同分选择与 float32 内部精度
可能不同，所以只在唯一切分案例要求逐点一致，真实数据比较效果和失误样本，不强求树结构相同。

测试：`uv run --locked --extra reference pytest tests/test_tree.py`。
可运行 demo：`DecisionTreeClassifier(max_depth=2).fit([[0,0],[0,1],[1,0],[1,1]],[0,1,1,0])`
再调用 `.predict()`；完整真实数据与容量对照见 `python -m kaleido.demos.classical_ml`。
参考：[sklearn 决策树](https://scikit-learn.org/stable/modules/tree.html)。
