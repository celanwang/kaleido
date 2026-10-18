# K-means 到 EM：硬分配与软分配

K-means 最小化 Σ||x−μ_label||²，交替最近中心分配与簇内均值更新。
用 D² 加权的 k-means++ 初始化，多个固定随机种子重启选训练 inertia 最低者。
空簇从非单样本簇搬入当前最远样本；不同样本数量少于簇数时拒绝拟合。
停止准则为最大中心位移的绝对容差，和 sklearn 的相对容差定义不同；
对照提供完全相同初始中心，比较收敛解，不能只凭相同 seed 假设初始化相同。

高斯混合把类别当潜变量。E 步根据 π_k N(x|μ_k,Σ_k) 得到责任度；
M 步按责任度更新权重、均值与方差。实现对角协方差，E 步在 log 域用 logsumexp，
方差加 `reg_covar` 避免奇异；质量近零的分量使该次重启失败，不以偷偷除极小数掩盖塌缩。
多个 K-means 启动选择最大训练平均 log likelihood；记录每轮 likelihood、失败次数与收敛状态。

未正则化 EM 的精确更新具有似然不下降性质；加方差下限后不承诺每轮严格单调。
测试在可控数据上检查近似单调，同时与同一完整初始化（权重、均值、方差）的 sklearn diag EM 对照。
K-means 偏好近球形、近等尺度簇；对角 EM 允许不同轴向尺度但不能表示旋转相关性。
局部极值、标签排列与分量不可识别都不因“算法收敛”消失。分量数在验证集选，不使用测试标签。

```python
from kaleido.classical import GaussianMixture, KMeans

x = [[0.0, 0.0], [0.0, 1.0], [8.0, 8.0], [8.0, 9.0]]
print(KMeans(2).fit_predict(x))
print(GaussianMixture(2).fit(x).predict_proba(x))
```

验证：`uv run --locked --extra reference pytest tests/test_cluster.py`；
Iris 留出集和旋转高斯反例见 `python -m kaleido.demos.classical_ml`。
参考：[sklearn 混合模型](https://scikit-learn.org/stable/modules/mixture.html)。
