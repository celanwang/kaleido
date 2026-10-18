# 代表案例：局部性、记忆与潜变量

CNN 采用 NCHW、stride=1、valid cross-correlation，核共享减少参数，并引入平移局部性。
滑窗前向，输入 VJP 以核位置累加，核 VJP 对所有样本/位置求和；不是对输入做循环平移。
Digits CNN 使用 4 个 3×3 核、ReLU、展平和线性头，保留空间信息，不建池化/卷积参数大全。

SimpleRNN 用 tanh(Wx+Uh+b)，按时间展开同一组参数，反传自动合并所有时间步的共享参数梯度。
合成 first-bit 任务要求从最后隐藏状态恢复第一个输入，并报告更长序列的失效。
梯度连乘解释长期依赖的消失/爆炸；本阶段不实现 LSTM/GRU 全家族。

VAE 选择对角 Gaussian posterior、单位 Gaussian prior、Bernoulli decoder。
显式传入固定 epsilon，通过 z=μ+exp(logvar/2)ε 使采样路径可求导。
负 ELBO=按样本求和的像素 BCE + KL(q||p)，KL=Σ(μ²+exp(logvar)−logvar−1)/2。
logvar 限于 [-10,10] 保持小实验稳定，并记录这一建模边界。
Digits 像素先除16得到软目标，这对应连续灰度的 Bernoulli 交叉熵代理，不声称是离散图像的精确似然。
验证使用固定噪声实现可重复的 Monte Carlo 负 ELBO，不能用“无噪声重建”冒充 ELBO。
β=0 作为 KL 消融，展示重建与先验约束的取舍；VAE 作为生成代表，GAN/扩散留为扩展。

测试：`uv run --locked --all-extras pytest tests/test_cnn_rnn_vae.py`；
真实训练：`python -m kaleido.demos.deep_learning`。卷积、BPTT 与重参数化/解析 KL 均和 PyTorch 对照。
参考：[Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)。
