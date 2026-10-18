# GRPO的具体公式与信息边界

每题采样G=4条回答，r∈{0,1}。A=(r−组均值)/(组population std+1e−4)，不使用critic。
同组全对或全错时A=0，报告零方差组比例；零信号并不等于实现失败，也不能隐藏。

本实现是显式GRPO变体：token ratio=exp(logπ−logπ_old)，PPO式正负优势clipping；
KL估计为 exp(logπ_ref−logπ)−(logπ_ref−logπ)−1，ref固定为SFT选中adapter。
每条完成按自己的有效长度平均，再对完成平均，保留原始GRPO的长度加权取舍。
不是当前TRL默认配置的隐式同义词；β、clip、组标准差口径和损失归一化均写进协议。

old/ref/advantage都detach。padding在指数之前屏蔽，真实EOS保留。
采样temperature=1、不做top-k/top-p截断，使behavior分布与记录的raw policy likelihood匹配。
每批做2次更新，ref与old都固定；采样KL是old分布上的近似约束，不宣称精确全词表KL。

损失测试用逐序列/逐token独立公式比较值与梯度，覆盖正负clipping、不同长度、零方差、
mask和detach边界。参考：[DeepSeekMath](https://arxiv.org/abs/2402.03300)、
[TRL公式说明](https://huggingface.co/docs/trl/grpo_trainer)。
