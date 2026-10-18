# PCA：保方差与保任务信息是两回事

中心化后 X=USVᵀ，前 k 个右奇异向量形成最大方差子空间，也最小化同秩线性重建误差。
样本协方差特征值为 s²/(n−1)。实现通过 NumPy SVD 得到分解，自行处理中心、截断、
方差比例、投影与逆投影；不做白化。

PCA 不默认标准化：协方差 PCA 与相关矩阵 PCA 是不同问题。缩放特征会改变它保留的方向。
只在训练集拟合均值和子空间，测试只投影；否则降维也会泄漏分布信息。
特征向量正负可任意，重根处连具体基向量也不唯一，参考比较的是投影矩阵/重建与特征值，
不是每个坐标符号。常量数据总方差为零时明确返回零方差比例，而非生成 NaN。

失效案例：一个方差很大的无关维度和一个方差小但完全决定标签的维度。
一维 PCA 会很好地重建整体能量，同时丢掉分类信息。降维率不能直接当作任务收益。

```python
from kaleido.classical import PCA

model = PCA(1).fit([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
print(model.transform([[3.0, 4.0]]), model.explained_variance_ratio_)
```

验证：`uv run --locked --extra reference pytest tests/test_pca.py`；
真实数据与低方差信号反例见 `python -m kaleido.demos.classical_ml`。
参考：[sklearn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)。
