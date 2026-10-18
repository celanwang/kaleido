# 线性与逻辑回归：统一接口，显式目标

`classical` 的估计器返回 `self`，监督模型为 `fit(X,y)/predict(X)`；变换为
`fit(X)/transform(X)/inverse_transform(X)`。只接受有限的二维实数特征，预测检查特征数与拟合状态。
分类支持有限数值或字符串标签，逻辑回归限二分类；不引入通用 Trainer。

线性回归中心化 X 与 y 后用 NumPy `lstsq` 解最小二乘，偏置由中心还原。
底层矩阵分解交给 NumPy，目标、预处理和估计器行为显式实现；不重复造 SVD。
秩亏时解最小范数系数，预测可能唯一而系数不可识别。`rank_` 暴露这一限制。

逻辑回归用 Bernoulli 负对数似然 `mean(softplus(z)-yz)+λ||w||²/2`，偏置不惩罚。
梯度 `Aᵀ(sigmoid(Aθ)-y)/n + λ[w,0]`，Hessian 是加权 Gram 矩阵加正则。
实现 Newton 方向与 Armijo 回溯；达到梯度容差才标记 `converged_`，否则发警告。
Newton 需要 O(d²) 存储和矩阵求解，适合这里的小型经典任务，不用于大模型训练。

sklearn 对照必须匹配目标：n 个等权样本时 `C=1/(nλ)`，不是 `C=1/λ`。
λ=0 时对应 `C=inf`；可分数据可能不存在有限极小值，不能以有限迭代停止当作收敛证据。
分类只以决策分数正负给出标签，概率不等于校准保证。

稳定性：softplus 用 `logaddexp(0,z)`，sigmoid 使用 `exp(-abs(z))`，避免极端 logits 溢出。
core 新增 sigmoid/softplus VJP 和 logits BCE，与手推梯度对照，避免对饱和概率直接取 log。
StandardScaler 只在训练集拟合，常量列 scale=1；测试数据不能参与均值/方差估计。

运行测试：`uv run --locked --extra reference pytest tests/test_classical_linear.py`。
可运行例子：

```python
from kaleido.classical import LogisticRegression, StandardScaler
x = StandardScaler().fit_transform([[0., 1.], [1., 0.], [2., 1.], [3., 0.]])
model = LogisticRegression().fit(x, [0, 0, 1, 1])
print(model.predict_proba(x), model.converged_)
```

真实数据与线性边界的失败案例由 `python -m kaleido.demos.classical_ml` 统一运行。
参考：[sklearn 线性与逻辑目标](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)。
