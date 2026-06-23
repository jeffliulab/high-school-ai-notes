# 2.5 优化与正则化

> **难度** ⭐⭐⭐☆☆ · **前置**：[2.2 梯度下降与反向传播](2-2-2-backprop.md)、[3.4 凸优化基础](../ch1-intro/1-3-4-convex-optimization.md)

!!! abstract "读完这一节，你会"
    - 说清 SGD、mini-batch、动量、Adam 这几个优化器分别在解决什么问题，并能从零写出它们的更新公式
    - 理解 L2 / L1 正则、early stopping、dropout 三种"防过拟合"手段背后的直觉，知道什么时候该用哪个
    - 会给训练加上学习率调度，并能从零实现与 PyTorch 库版互相对照

## 梯度方向对了，可步子怎么迈才是真问题

在 [2.2 梯度下降与反向传播](2-2-2-backprop.md) 里我们已经把"梯度"这件事彻底讲清楚了：梯度告诉你"往哪个方向走，损失下降最快"。那一节的更新公式你应该很熟了——

$$\theta \leftarrow \theta - \eta\,\nabla f(\theta),$$

其中 $\theta$ 是模型参数，$\eta$（读作 eta）是**学习率**（learning rate），也就是每一步迈多大。这个公式从 ch1 的 [3.4 凸优化基础](../ch1-intro/1-3-4-convex-optimization.md) 一路用到现在，是所有训练的引擎。

可问题来了：光知道下山的方向，并不等于就能顺利下到谷底。真实训练里你会遇到一堆麻烦——数据有几百万条，每走一步都把全部数据过一遍太慢了；山谷的形状又窄又长，朴素的梯度下降会在两侧来回横跳、半天下不去；学习率定大了越过谷底、定小了磨蹭一整天。**这一节要讲的"优化器"，就是一系列让"迈步"这件事更聪明的技巧。**

而即便你顺利走到了谷底、把训练集上的损失压得很低，还有第二个隐患在等着：模型可能把训练数据"背"得太死，连里面的噪声都记住了，一换成没见过的新数据就原形毕露。这种"只会做原题、不会举一反三"的毛病叫**过拟合**（overfitting）。后半节讲的**正则化**，就是专门治它的。

## 从全量梯度到 mini-batch：先把"每步多少数据"想清楚

最原始的梯度下降，每走一步都要把**所有**训练样本的梯度算一遍再求平均，这叫**批量梯度下降**（batch gradient descent）。它的方向最准，但代价是：数据一多，一步都迈不动。

另一个极端是**随机梯度下降**（SGD，stochastic gradient descent）：每次只随机抓**一条**样本算梯度就更新。这样走得飞快，但单条样本的梯度噪声很大，路线会抖得厉害。

实际工程里我们几乎总是取中间方案——**小批量梯度下降**（mini-batch gradient descent）：每次抓一小撮样本（比如 32 或 64 条，这个数量叫 batch size）算梯度。它既快又稳，是今天深度学习的默认做法。下面这张图把三者的"下山路线"画在了同一张等高线图上，你一眼就能看出区别。

<div class="diagram">
<svg viewBox="0 0 420 220" font-family="-apple-system, 'Segoe UI', sans-serif">
  <defs>
    <marker id="ba" markerWidth="8" markerHeight="8" refX="5" refY="3" orient="auto"><path d="M0 0 L5 3 L0 6 Z" fill="var(--dia-stroke-soft)"/></marker>
  </defs>
  <!-- 同心等高线（拉长的椭圆，模拟窄长山谷），中心 (300,110) -->
  <ellipse cx="300" cy="110" rx="105" ry="44" fill="none" stroke="var(--dia-rule)" stroke-width="1.5"/>
  <ellipse cx="300" cy="110" rx="72" ry="30" fill="none" stroke="var(--dia-rule)" stroke-width="1.5"/>
  <ellipse cx="300" cy="110" rx="40" ry="17" fill="none" stroke="var(--dia-rule)" stroke-width="1.5"/>
  <circle cx="300" cy="110" r="4" fill="var(--dia-accent-deep)"/>
  <text x="300" y="92" text-anchor="middle" font-size="11" fill="var(--dia-stroke-soft)">最优点</text>
  <!-- Batch：平滑直达（蓝） -->
  <path d="M62 60 Q180 78 258 104" fill="none" stroke="var(--dia-blue)" stroke-width="2" marker-end="url(#ba)"/>
  <circle cx="62" cy="60" r="4" fill="var(--dia-blue)"/>
  <text x="40" y="50" font-size="11" fill="var(--dia-blue)">Batch：稳但慢</text>
  <!-- mini-batch：略抖但快（绿） -->
  <path d="M62 110 L110 122 L150 108 L196 120 L240 110 L268 112" fill="none" stroke="var(--dia-green)" stroke-width="2" marker-end="url(#ba)"/>
  <circle cx="62" cy="110" r="4" fill="var(--dia-green)"/>
  <text x="40" y="146" font-size="11" fill="var(--dia-green)">mini-batch：又快又稳</text>
  <!-- SGD：抖得厉害（赭） -->
  <path d="M62 168 L96 142 L124 178 L162 138 L198 176 L236 134 L266 124" fill="none" stroke="var(--dia-accent)" stroke-width="1.6" marker-end="url(#ba)"/>
  <circle cx="62" cy="168" r="4" fill="var(--dia-accent)"/>
  <text x="40" y="200" font-size="11" fill="var(--dia-accent)">SGD：快但抖</text>
</svg>
</div>
<p class="figure-caption">图 2.5-1：在同一片等高线山谷上，三种"每步用多少数据"的下山路线对比。</p>

需要澄清一个常见的叫法混乱：今天大家口头说的"SGD"，其实绝大多数时候指的就是 mini-batch 版本，而不是严格意义上"每次一条样本"的那个原始 SGD。后面我们也沿用这个习惯。

## 动量：给小球一点惯性，别在山谷里来回横跳

mini-batch 已经不错了，但碰到"窄长山谷"这种地形（一个方向很陡、另一个方向很平），朴素 SGD 还是会在陡的两侧来回横跳，真正该前进的平缓方向反而挪得很慢。

怎么办？我们给下山的小球加一点**惯性**。现实里小球滚下山坡时，不会每一步都完全听当前坡度的，它还带着前几步攒下来的速度。把这个想法搬进更新公式，就是**动量法**（momentum）：

$$v \leftarrow \beta\,v + \nabla f(\theta), \qquad \theta \leftarrow \theta - \eta\, v.$$

这里 $v$ 是"速度"，它不是只看当前梯度，而是把历史梯度按系数 $\beta$（一般取 0.9）做了个**指数加权平均**。这样一来，那些来回横跳的抖动方向因为正负相消、被平均掉了，而始终一致指向谷底的方向却被一步步累加、越走越快。一句话记住：**动量让一致的方向加速、让抖动的方向相互抵消。**

## Adam：每个参数都配一个自己的学习率

动量解决了"方向"的问题，但还有个老大难：所有参数共用同一个学习率 $\eta$ 真的合理吗？有的参数对应的梯度一直很大，有的一直很小，用同一个步长显然不公平。

**Adam**（Adaptive Moment Estimation，自适应矩估计）就是来解决这个的。它同时干了两件事：一边像动量那样累积梯度的"一阶矩"（平均方向 $m$），一边累积梯度平方的"二阶矩"（大小 $v$），然后用后者去**自动缩放**每个参数的步长——梯度一直很大的参数，步长就被自动调小；梯度很小的，步长就放大。它的更新大致是这样：

$$
m \leftarrow \beta_1 m + (1-\beta_1)\nabla f,\quad
v \leftarrow \beta_2 v + (1-\beta_2)(\nabla f)^2,\quad
\theta \leftarrow \theta - \eta\,\frac{\hat m}{\sqrt{\hat v}+\epsilon}.
$$

其中 $\hat m,\hat v$ 是对 $m,v$ 做了"偏差修正"后的版本（训练刚开始时 $m,v$ 从 0 起步会偏小，修正一下），$\epsilon$ 是个很小的数（如 $10^{-8}$）防止除以零。你不必把这套公式背死，记住一句话就够了：**Adam = 动量 + 每个参数自适应的学习率**，它收敛快、对学习率不太挑剔，是今天最常用的默认优化器。

## 从零写一遍这几个优化器，再和 PyTorch 对照

道理讲完了，我们动手把它们写出来。光看公式容易飘，亲手实现一遍你才会真正相信"哦，原来就这么几行"。下面在一个简单的二次函数 $f(\theta)=\theta^2$ 上跑，它的真梯度是 $2\theta$，最优点在 $\theta=0$，方便我们检查谁先到。

```python
import numpy as np

def grad(theta):           # f(theta) = theta^2 的解析梯度
    return 2 * theta

# ---------- 朴素 SGD ----------
theta, lr = 5.0, 0.1
for _ in range(50):
    theta -= lr * grad(theta)          # θ ← θ - η·∇f
print("SGD     :", round(theta, 4))

# ---------- 动量 ----------
theta, lr, beta, v = 5.0, 0.1, 0.9, 0.0
for _ in range(50):
    v = beta * v + grad(theta)         # 速度 = 惯性 + 当前梯度
    theta -= lr * v
print("Momentum:", round(theta, 4))

# ---------- Adam（含偏差修正） ----------
theta, lr = 5.0, 0.1
b1, b2, eps = 0.9, 0.999, 1e-8
m, v = 0.0, 0.0
for t in range(1, 51):
    g = grad(theta)
    m = b1 * m + (1 - b1) * g          # 一阶矩：平均方向
    v = b2 * v + (1 - b2) * g**2       # 二阶矩：梯度大小
    m_hat = m / (1 - b1**t)            # 偏差修正
    v_hat = v / (1 - b2**t)
    theta -= lr * m_hat / (np.sqrt(v_hat) + eps)
print("Adam    :", round(theta, 4))
```

跑出来你会看到，动量比朴素 SGD 更快地逼近 0，而 Adam 因为自适应步长也很快收敛。这就把上面那些公式从"纸面"变成了"能动的东西"。

在真正训练神经网络时，我们当然不会手写这些循环，而是直接调 PyTorch 的优化器——它内部做的就是上面这套事，只不过帮你管好了所有参数：

```python
import torch

model = torch.nn.Linear(10, 1)
# 换一行就换一个优化器，公式都在库里实现好了：
opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
# opt = torch.optim.Adam(model.parameters(), lr=1e-3)   # 最常用的默认选择

for x, y in dataloader:               # 每次拿一个 mini-batch
    opt.zero_grad()                   # 清掉上一步的梯度
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()                   # 反向传播算梯度（见 2.2）
    opt.step()                        # 按选定的优化器更新参数
```

注意那三行固定套路 `zero_grad → backward → step`：先清空旧梯度，再算新梯度，最后才更新。漏掉 `zero_grad()` 是新手最常见的 bug，梯度会一轮轮累加导致训练发疯。

## 正则化的总思路：给"想太多"的模型加一道约束

换个话题，来对付前面提到的过拟合。先看清问题的本质：过拟合的模型，往往参数值被训练得又大又极端，硬是穿过每一个训练点（连噪声也不放过），曲线扭得很夸张。下面这张图把"刚刚好"和"过拟合"两条曲线画在一起，对比很直观。

<div class="diagram">
<svg viewBox="0 0 420 200" font-family="-apple-system, 'Segoe UI', sans-serif">
  <!-- 坐标轴 -->
  <line x1="40" y1="170" x2="400" y2="170" stroke="var(--dia-stroke-tertiary)" stroke-width="1"/>
  <line x1="40" y1="20" x2="40" y2="170" stroke="var(--dia-stroke-tertiary)" stroke-width="1"/>
  <!-- 数据点（带噪声，沿一条平缓趋势散布） -->
  <circle cx="80" cy="130" r="3.5" fill="var(--dia-stroke-soft)"/>
  <circle cx="130" cy="108" r="3.5" fill="var(--dia-stroke-soft)"/>
  <circle cx="175" cy="118" r="3.5" fill="var(--dia-stroke-soft)"/>
  <circle cx="220" cy="86" r="3.5" fill="var(--dia-stroke-soft)"/>
  <circle cx="265" cy="96" r="3.5" fill="var(--dia-stroke-soft)"/>
  <circle cx="310" cy="64" r="3.5" fill="var(--dia-stroke-soft)"/>
  <circle cx="355" cy="74" r="3.5" fill="var(--dia-stroke-soft)"/>
  <!-- 刚刚好：平滑趋势线（绿） -->
  <path d="M80 124 Q220 96 355 70" fill="none" stroke="var(--dia-green)" stroke-width="2.2"/>
  <!-- 过拟合：扭来扭去穿过每个点（赭，虚线） -->
  <path d="M80 130 L130 108 L175 118 L220 86 L265 96 L310 64 L355 74" fill="none" stroke="var(--dia-accent)" stroke-width="1.8" stroke-dasharray="4 3"/>
  <!-- 图例（放在右上空白处，不压曲线） -->
  <line x1="250" y1="168" x2="278" y2="168" stroke="var(--dia-green)" stroke-width="2.2"/>
  <text x="284" y="172" font-size="11" fill="var(--dia-green)">刚刚好</text>
  <line x1="250" y1="186" x2="278" y2="186" stroke="var(--dia-accent)" stroke-width="1.8" stroke-dasharray="4 3"/>
  <text x="284" y="190" font-size="11" fill="var(--dia-accent)">过拟合</text>
</svg>
</div>
<p class="figure-caption">图 2.5-2：过拟合的曲线为了穿过每个点而剧烈扭曲，平滑的拟合反而更可能在新数据上表现好。</p>

既然过拟合常伴随"参数太大、曲线太弯"，那治它的总思路就很自然了：**在损失函数里额外加一项惩罚，让模型在"拟合数据"和"保持参数简单"之间权衡。** 这就是正则化。

## L2 与 L1：往损失里加一项"别太大"的惩罚

最常用的是 **L2 正则**（也叫权重衰减，weight decay）。做法是在原来的损失后面加上所有参数平方和，再乘一个系数 $\lambda$：

$$L_{\text{total}} = L_{\text{data}} + \lambda \sum_i \theta_i^2.$$

这一项的意思是"参数越大，惩罚越重"。于是优化器在压低数据损失的同时，也会本能地把参数往小了拉——参数小了，曲线就平滑了，过拟合自然缓解。系数 $\lambda$ 是个旋钮：调大了模型更"老实"但可能欠拟合，调小了约束就弱。

还有一个孪生兄弟 **L1 正则**，把平方换成绝对值 $\lambda\sum_i|\theta_i|$。它有个 L2 没有的特殊本事：会把一批不重要的参数直接压到**恰好等于 0**，相当于自动做了特征筛选。所以当你怀疑很多特征是没用的、想让模型自己挑出关键的几个时，L1 特别合适，这一点和 [1.10 进阶特征工程](2-1-10-feature-engineering.md) 里的思路是相通的。

在代码里加 L2 几乎不费力——PyTorch 的优化器直接有个 `weight_decay` 参数：

```python
# weight_decay 就是 L2 正则的系数 λ，一行搞定
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

如果你想看它"从零"长什么样，其实就是手动在损失里加上那一项：

```python
l2 = sum((p ** 2).sum() for p in model.parameters())   # ∑ θ²
loss = data_loss + 1e-4 * l2                            # 总损失 = 数据损失 + λ·L2
```

## Early stopping 与 dropout：两种不靠改损失的防过拟合手段

除了往损失里加惩罚项，还有两种思路完全不同、却同样好用的招数。

第一个是**早停**（early stopping），它简单到有点不像"技术"。训练时我们一直盯着**验证集**（没参与训练的那部分数据）上的损失：一开始它会随训练一起下降，但到某个点之后，训练损失还在降、验证损失却开始回升——这正是模型开始"背题"、过拟合抬头的信号。早停做的就是：在验证损失最低的那一刻把训练叫停，保留那一版模型。换句话说，**在模型开始学坏之前就喊卡**。

第二个是 **dropout**，专门给神经网络用，名字直译就是"随机丢弃"。训练时，每一步都随机让一部分神经元"罢工"（输出置零），比如丢掉一半。为什么这招管用？因为模型不能再依赖某几个"明星神经元"包打天下了——任何一个都可能随时缺席，于是它被迫让每个神经元都学到点有用的、互不依赖的本事。这相当于在悄悄训练很多个"残缺"的子网络，最后让它们一起投票，泛化能力自然更强。要特别记住：**dropout 只在训练时开启，预测时必须关掉**（用全部神经元），PyTorch 里靠 `model.train()` 和 `model.eval()` 来切换。

```python
import torch.nn as nn

net = nn.Sequential(
    nn.Linear(784, 256), nn.ReLU(),
    nn.Dropout(0.5),          # 训练时随机丢弃 50% 的神经元
    nn.Linear(256, 10),
)

net.train()   # 训练模式：dropout 生效
# ... 训练若干轮 ...
net.eval()    # 评估模式：dropout 自动关闭，用全部神经元做预测
```

## 学习率调度：步子该从大到小慢慢收

最后补一个很实用、考试也常考的小话题——**学习率调度**（learning rate scheduling）。

前面我们把学习率 $\eta$ 当成一个固定的数，但更聪明的做法是让它随训练**动态变化**。想想下山的画面就懂了：刚出发离谷底远，大胆迈大步、快速接近；快到谷底了再迈大步就会越过去，这时应该把步子收小、稳稳落底。所以一个非常常见的策略就是**让学习率随训练逐渐衰减**，比如每过若干轮就乘以 0.1。

PyTorch 把这件事封装成了"调度器"（scheduler），用起来和优化器配套：

```python
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
# 每过 10 个 epoch，把学习率乘以 0.1（即衰减为原来的十分之一）
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)

for epoch in range(30):
    for x, y in dataloader:
        opt.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
    sched.step()              # 注意：每个 epoch 结束后才调一次，不是每个 batch
```

那个 `sched.step()` 放的位置很关键——它应该在每个 epoch（把全部数据过一遍叫一个 epoch）结束后调用，而不是每个 batch。放错地方，学习率会衰减得快得离谱，这也是初学者常踩的坑。

## 容易踩的坑

- **忘了 `opt.zero_grad()`**：PyTorch 的梯度是默认累加的，不在每步开头清零，梯度会一轮轮叠起来，loss 直接发散。这是新手第一号 bug。
- **学习率没调好**：太大就在谷底两侧横跳甚至发散，太小则磨蹭半天不动。换了优化器（尤其从 SGD 换 Adam）后，老的学习率往往不再适用，要重新调。
- **dropout 忘了切 `eval()`**：评估或预测时如果还停在 `train()` 模式，dropout 会继续随机丢神经元，结果每次都不一样、还偏低。同理 BatchNorm 也吃这套，养成习惯：评估前先 `model.eval()`。
- **正则化系数 $\lambda$ 一刀切**：$\lambda$ 太大模型会从过拟合直接滑向欠拟合，连训练集都学不好。它该和学习率一样，当成一个要在验证集上调的超参数。
- **`scheduler.step()` 放进了 batch 循环**：本该每个 epoch 衰减一次，放进内层循环就变成每个 batch 都衰减，学习率瞬间归零。

## 它在后面会怎么用到

- 这一节的优化器和正则化，是你训练**任何**神经网络的标准配置。马上在 [2.4 PyTorch 基础](2-2-4-pytorch-basics.md) 搭出来的网络里就要用到 `Adam` 和 `zero_grad → backward → step` 这套流程。
- 训到 CNN 时（[3.1 卷积与池化](2-3-1-conv-pooling.md)、[3.2 图像分类实战](2-3-2-image-classification.md)），dropout、weight decay 和学习率调度几乎是默认就要加上的"防过拟合三件套"。
- 更深入的调参与实验管理，会在集训营的 [1.2 调参与实验管理](../ch4-camp/4-1-2-experiment-management.md) 里系统展开；而"为什么这些方法在凸/非凸情形下都还能用"，根子在 [3.4 凸优化基础](../ch1-intro/1-3-4-convex-optimization.md)。

## 练习

??? note "基础练习"
    1. 把"从零写一遍优化器"那段代码跑起来，把 50 步打印成一条 $\theta$ 随步数变化的列表，比较 SGD、动量、Adam 谁最先逼近 0。再把动量系数 $\beta$ 从 0.9 改成 0.5，观察收敛变快还是变慢，并解释为什么。
    2. 用一句话分别说清：early stopping 靠盯哪个数据集来判断该停？dropout 在训练和预测时分别开还是关？

??? note "进阶练习"
    1. 给一个会过拟合的小网络（比如在很少的数据上训练）依次加上 L2（`weight_decay`）和 dropout，画出加之前和加之后"训练损失 vs 验证损失"两条曲线，亲眼确认验证损失被压下来了。
    2. 自己实现一个最简单的学习率调度：不用 PyTorch 的 scheduler，而是在每个 epoch 手动把 `opt.param_groups[0]['lr']` 乘以 0.9。验证它和 `StepLR` 行为一致。

## 小结

- 一句话总结这一节：**优化器让"迈步"更聪明（mini-batch 快、动量稳方向、Adam 自适应步长 + 学习率调度收尾），正则化让模型不"背题"（L2/L1 罚大参数、early stopping 适时喊卡、dropout 随机丢神经元）。**
- 训练任何网络，记住那条不变的引擎 $\theta\leftarrow\theta-\eta\nabla f$，以及 `zero_grad → backward → step` 三连。

想把这些优化器的几何直觉看得更透，强烈推荐读 [*An overview of gradient descent optimization algorithms*](https://www.ruder.io/optimizing-gradient-descent/)（Sebastian Ruder 的这篇综述把 SGD 到 Adam 的演进讲得极清楚）；正则化与 dropout 的原始动机可以看 [*Deep Learning*](https://www.deeplearningbook.org/) 第 7、8 章。
