# 2.1 感知机与 MLP

> **难度** ⭐⭐⭐☆☆ · **前置**：[3.1 线性代数](../ch1-intro/1-3-1-linear-algebra.md)、[1.2 线性回归](2-1-2-linear-regression.md)

!!! abstract "读完这一节，你会"
    - 看懂单个神经元在做什么：一次线性加权，再过一道"非线性的关卡"（激活函数）
    - 明白为什么把很多神经元堆成多层（MLP）就能拟合各种弯弯曲曲的函数
    - 用 NumPy 从零写出一个前向传播，再用 PyTorch 的库版本对照，二者结果一致

## 从线性回归说起：为什么还不够

在 [1.2 线性回归](2-1-2-linear-regression.md) 里，我们已经见过最朴素的模型长什么样：给每个输入特征配一个权重，加权求和，再加一个偏置。写成公式就是 $y = \mathbf{w}^\top\mathbf{x} + b$。这其实就是一条直线（或者高维里的一个平面），它能把"线性可分"的问题处理得很好。

可问题在于，**现实世界里大多数关系都不是直线**。比如"异或"这种简单逻辑——两个输入相同时输出 0，不同时输出 1——你画在坐标纸上就会发现，无论怎么放一条直线，都没法把这两类点干净地分开。换句话说，纯线性模型有一个天花板：**它只会画直线，画不了曲线。**

那怎么突破这个天花板？答案出奇地简单：在线性加权之后，再加一道"非线性的关卡"。这道关卡，就是这一节的第一个主角——**神经元**。

## 一个神经元：线性一下，再"掰弯"一下

我们先把单个神经元拆开看。它做的事情其实只有两步，前一步你已经很熟了，后一步才是新东西。

**第一步是线性加权**，和线性回归一模一样：把输入向量 $\mathbf{x}$ 的每个分量乘上对应的权重再加起来，再补一个偏置 $b$。我们把这个中间结果记作 $z$：

$$z = \mathbf{w}^\top\mathbf{x} + b = \sum_{i} w_i x_i + b.$$

**第二步是过激活函数**，这是神经元和线性回归唯一的区别，却也是它全部威力的来源。我们拿一个非线性函数 $\sigma$ 把 $z$ "掰弯"一下，得到最终输出 $a$：

$$a = \sigma(z).$$

这里的 $\sigma$ 可以是好几种函数。最经典的一个叫 **Sigmoid**，它的形状是一条 S 形曲线，能把任意实数压到 $0$ 到 $1$ 之间，特别适合表示"概率"；另一个在现代网络里用得最多的叫 **ReLU**（修正线性单元），它的规则朴素到一句话就能说完：负数变成 0，正数原样保留。我们下面这张图把单个神经元的数据流画了出来，你可以顺着箭头走一遍。

<div class="diagram">
<svg viewBox="0 0 420 200" font-family="-apple-system, 'Segoe UI', sans-serif">
  <defs>
    <marker id="na" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="var(--dia-stroke-tertiary)"/></marker>
  </defs>
  <!-- 三个输入节点 -->
  <circle cx="50" cy="45" r="14" fill="none" stroke="var(--dia-blue)" stroke-width="2"/>
  <circle cx="50" cy="100" r="14" fill="none" stroke="var(--dia-blue)" stroke-width="2"/>
  <circle cx="50" cy="155" r="14" fill="none" stroke="var(--dia-blue)" stroke-width="2"/>
  <text x="50" y="50" text-anchor="middle" font-size="13" fill="var(--dia-blue)">x₁</text>
  <text x="50" y="105" text-anchor="middle" font-size="13" fill="var(--dia-blue)">x₂</text>
  <text x="50" y="160" text-anchor="middle" font-size="13" fill="var(--dia-blue)">x₃</text>
  <!-- 加权连线 -->
  <line x1="64" y1="45" x2="188" y2="100" stroke="var(--dia-stroke-tertiary)" stroke-width="1.5"/>
  <line x1="64" y1="100" x2="188" y2="100" stroke="var(--dia-stroke-tertiary)" stroke-width="1.5"/>
  <line x1="64" y1="155" x2="188" y2="100" stroke="var(--dia-stroke-tertiary)" stroke-width="1.5"/>
  <text x="120" y="58" font-size="11" fill="var(--dia-stroke-soft)">w₁</text>
  <text x="120" y="143" font-size="11" fill="var(--dia-stroke-soft)">w₃</text>
  <!-- 求和单元 z -->
  <circle cx="205" cy="100" r="18" fill="var(--dia-accent-soft)" stroke="var(--dia-accent)" stroke-width="2"/>
  <text x="205" y="105" text-anchor="middle" font-size="13" fill="var(--dia-accent)">Σ</text>
  <text x="205" y="150" text-anchor="middle" font-size="11" fill="var(--dia-stroke-soft)">z = wᵀx + b</text>
  <!-- z 到激活 -->
  <line x1="223" y1="100" x2="290" y2="100" stroke="var(--dia-stroke-tertiary)" stroke-width="1.5" marker-end="url(#na)"/>
  <!-- 激活单元 -->
  <rect x="296" y="80" width="44" height="40" rx="4" fill="none" stroke="var(--dia-green)" stroke-width="2"/>
  <text x="318" y="105" text-anchor="middle" font-size="13" fill="var(--dia-green)">σ</text>
  <text x="318" y="150" text-anchor="middle" font-size="11" fill="var(--dia-stroke-soft)">激活函数</text>
  <!-- 输出 -->
  <line x1="340" y1="100" x2="400" y2="100" stroke="var(--dia-stroke-tertiary)" stroke-width="1.5" marker-end="url(#na)"/>
  <text x="392" y="92" text-anchor="middle" font-size="13" fill="var(--dia-stroke)">a</text>
</svg>
</div>
<p class="figure-caption">图 2.1-1：单个神经元 = 先把输入加权求和得到 z，再用激活函数 σ 把 z 掰弯，输出 a。</p>

历史上，最早的神经元模型叫**感知机（perceptron）**，由 Rosenblatt 在 1958 年提出。它的激活函数是一个简单的"阶跃"：$z$ 大于零就输出 1，否则输出 0。它能学会一些简单的线性分类，但正是因为只有这一个神经元、只会画一条直线，它对前面说的"异或"问题束手无策——这个著名的局限，一度让神经网络的研究冷了下来。

## 把神经元堆起来：多层感知机

单个神经元画不了曲线，那一群神经元呢？这就引出了这一节真正的主角——**多层感知机（Multi-Layer Perceptron，MLP）**，它也是最基础的一种"神经网络"。

它的想法非常直白：**把一排神经元并起来构成一"层"，再把好几层前后串起来。** 前一层的输出，就当作后一层的输入。我们一般把它分成三种角色：直接接收原始数据的叫**输入层**，最后吐出结果的叫**输出层**，夹在中间、不直接和外界打交道的叫**隐藏层（hidden layer）**——正是这些隐藏层，给了网络"画曲线"的能力。

借助 [3.1 线性代数](../ch1-intro/1-3-1-linear-algebra.md) 里讲过的矩阵语言，整整一层的计算可以一口气写完。设这一层有权重矩阵 $W$ 和偏置向量 $\mathbf{b}$，输入是上一层的激活值 $\mathbf{a}^{(l-1)}$，那么这一层的计算就是：

$$\mathbf{z}^{(l)} = W^{(l)}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}, \qquad \mathbf{a}^{(l)} = \sigma\!\left(\mathbf{z}^{(l)}\right).$$

你会发现，这和单个神经元的两步是同一套逻辑，只是把"加权求和"升级成了一次**矩阵乘向量**——一次算完整层所有神经元。一个三层的 MLP，无非就是把这两行公式连用三次。下面这张图把这种"逐层传递"画了出来。

<div class="diagram">
<svg viewBox="0 0 440 220" font-family="-apple-system, 'Segoe UI', sans-serif">
  <!-- 连线：输入层->隐藏层 -->
  <g stroke="var(--dia-stroke-tertiary)" stroke-width="0.8" opacity="0.6">
    <line x1="74" y1="70" x2="206" y2="50"/><line x1="74" y1="70" x2="206" y2="110"/><line x1="74" y1="70" x2="206" y2="170"/>
    <line x1="74" y1="150" x2="206" y2="50"/><line x1="74" y1="150" x2="206" y2="110"/><line x1="74" y1="150" x2="206" y2="170"/>
  </g>
  <!-- 连线：隐藏层->输出层 -->
  <g stroke="var(--dia-stroke-tertiary)" stroke-width="0.8" opacity="0.6">
    <line x1="226" y1="50" x2="356" y2="110"/><line x1="226" y1="110" x2="356" y2="110"/><line x1="226" y1="170" x2="356" y2="110"/>
  </g>
  <!-- 输入层 -->
  <circle cx="60" cy="70" r="14" fill="none" stroke="var(--dia-blue)" stroke-width="2"/>
  <circle cx="60" cy="150" r="14" fill="none" stroke="var(--dia-blue)" stroke-width="2"/>
  <!-- 隐藏层 -->
  <circle cx="216" cy="50" r="14" fill="var(--dia-accent-soft)" stroke="var(--dia-accent)" stroke-width="2"/>
  <circle cx="216" cy="110" r="14" fill="var(--dia-accent-soft)" stroke="var(--dia-accent)" stroke-width="2"/>
  <circle cx="216" cy="170" r="14" fill="var(--dia-accent-soft)" stroke="var(--dia-accent)" stroke-width="2"/>
  <!-- 输出层 -->
  <circle cx="370" cy="110" r="14" fill="none" stroke="var(--dia-green)" stroke-width="2"/>
  <!-- 文字标注：全部放在节点上下方空白处，不压线 -->
  <text x="60" y="200" text-anchor="middle" font-size="12" fill="var(--dia-blue)">输入层</text>
  <text x="216" y="205" text-anchor="middle" font-size="12" fill="var(--dia-accent)">隐藏层</text>
  <text x="370" y="200" text-anchor="middle" font-size="12" fill="var(--dia-green)">输出层</text>
  <text x="138" y="25" text-anchor="middle" font-size="11" fill="var(--dia-stroke-soft)">W⁽¹⁾, b⁽¹⁾</text>
  <text x="296" y="25" text-anchor="middle" font-size="11" fill="var(--dia-stroke-soft)">W⁽²⁾, b⁽²⁾</text>
</svg>
</div>
<p class="figure-caption">图 2.1-2：三层 MLP。数据从左往右逐层流动，每条连线是一个权重，每一层都做一次"线性 + 激活"。</p>

## 那个非线性，到底有多关键

讲到这里，你可能会问一个很自然的问题：既然每一层都做"线性变换"，我把激活函数 $\sigma$ 去掉，直接一层层线性叠下去，不也挺好？

答案是——那样叠多少层都没用。我们来推一遍就明白了。假设没有激活函数，第一层是 $\mathbf{z}_1 = W_1\mathbf{x}$，第二层是 $\mathbf{z}_2 = W_2\mathbf{z}_1$。把第一层代进第二层：

$$\mathbf{z}_2 = W_2(W_1\mathbf{x}) = (W_2 W_1)\,\mathbf{x}.$$

你看，两个矩阵相乘 $W_2 W_1$ 还是一个矩阵。换句话说，**叠了两层线性，等价于只有一层线性**——多出来的那一层完全是白搭，整个网络还是只会画直线。所以那个夹在中间的非线性激活，不是可有可无的装饰，而是让"叠加变深"真正产生意义的关键。

这件事还有一个非常深刻的理论支撑，叫**通用逼近定理（Universal Approximation Theorem）**。它说的是：只要隐藏层里的神经元足够多，一个仅含**一层**隐藏层的 MLP，就能以任意精度逼近几乎任何连续函数。

不过你别被这个"万能"吓住，它更多是个**存在性的安慰**而非实操指南：定理只保证"存在这样一组权重"，却没告诉你这组权重是多少、要多少个神经元、能不能训得出来。实践中我们宁愿把网络叠得**更深而不是更宽**——深层网络往往用少得多的神经元就能表达同样复杂的函数。你只需要从这个定理里记住一句话就够了：**线性负责"拉伸空间"，非线性负责"折叠空间"，两者交替进行，网络就能把原本缠在一起的数据掰开。**

## 动手实现：从零写一个前向传播

道理讲完了，我们动手把它跑起来。先用 NumPy 从零实现，这样你能清清楚楚看到每一行公式对应哪段代码——没有任何"黑箱"。我们搭一个最小的 MLP：2 个输入、1 个有 4 个神经元的隐藏层、1 个输出，激活函数隐藏层用 ReLU、输出层用 Sigmoid。

先把两个激活函数和参数准备好。注意 ReLU 就是一句 `np.maximum(0, z)`——把负数截成 0，正好对应我们前面说的"负数变 0，正数保留"：

```python
import numpy as np

def relu(z):
    return np.maximum(0, z)        # 负数 -> 0，正数原样保留

def sigmoid(z):
    return 1 / (1 + np.exp(-z))    # 把任意实数压到 (0, 1)，可当概率读

# 一个样本：2 个特征
x = np.array([0.5, -1.2])

# 第 1 层：4 个神经元，所以 W1 是 (4, 2)，b1 是 (4,)
W1 = np.array([[ 0.3, -0.8],
               [ 0.1,  0.5],
               [-0.6,  0.2],
               [ 0.9, -0.3]])
b1 = np.array([0.0, 0.1, -0.2, 0.05])

# 第 2 层（输出）：1 个神经元，W2 是 (1, 4)，b2 是 (1,)
W2 = np.array([[0.7, -0.4, 0.2, 0.6]])
b2 = np.array([0.1])
```

参数就位，现在把前面那两行公式 $\mathbf{z}=W\mathbf{a}+\mathbf{b}$、$\mathbf{a}=\sigma(\mathbf{z})$ 原封不动地敲成代码。你会看到代码和公式几乎是一一对应的：

```python
# 第 1 层：线性 -> ReLU
z1 = W1 @ x + b1        # 矩阵乘向量，得到 4 个 z
a1 = relu(z1)           # 过激活，得到隐藏层的输出

# 第 2 层：线性 -> Sigmoid
z2 = W2 @ a1 + b2       # 把隐藏层输出再加权求和
a2 = sigmoid(z2)        # 输出层用 Sigmoid，得到一个 0~1 的"概率"

print("隐藏层输出:", a1)
print("最终输出:", a2)   # 例如 [0.63]，可解读为"属于正类的概率约 0.63"
```

跑完你会得到一个落在 $0$ 到 $1$ 之间的数。这一整段，就是神经网络做预测时走的**前向传播（forward pass）**——数据从输入一路向前流到输出。现在我们手里的权重还是随便填的，所以预测没有意义；**怎么把这些权重调到"预测得准"，正是下一节反向传播要解决的事。**

## 库版对照：同一个网络，PyTorch 怎么写

从零实现是为了看清原理，但真正做项目时，没人会手写每一层。深度学习框架（这里用 [2.4 PyTorch 基础](2-2-4-pytorch-basics.md) 会细讲的 PyTorch）把"线性层 + 激活"都封装成了现成积木，我们只需要像搭乐高一样把它们叠起来。下面这段搭的是和上面**结构完全相同**的网络：

```python
import torch
import torch.nn as nn

# nn.Sequential：把各层按顺序串起来，正是我们图 2.1-2 的结构
model = nn.Sequential(
    nn.Linear(2, 4),     # 输入 2 维 -> 隐藏 4 维，等价于上面的 W1, b1
    nn.ReLU(),           # 隐藏层激活
    nn.Linear(4, 1),     # 隐藏 4 维 -> 输出 1 维，等价于 W2, b2
    nn.Sigmoid(),        # 输出层激活
)

x = torch.tensor([0.5, -1.2])
out = model(x)           # 一次调用就完成整个前向传播
print(out)               # 形如 tensor([0.5x], grad_fn=...)
```

对比着看，你会发现两段代码做的是同一件事：`nn.Linear(2, 4)` 内部就藏着一个 $4\times2$ 的权重矩阵和一个偏置，`model(x)` 这一次调用，背后跑的正是我们手写的那几行 `W @ x + b` 加激活。**库版的好处不只是省字**——它还顺手把每个参数的梯度通路都记下来了（注意输出里那个 `grad_fn`），下一节训练时就能直接用上，这是手写版暂时还没有的能力。

## 容易踩的坑

- **忘了加激活函数**：如果隐藏层之间不放非线性，叠再多层也等价于一层线性，网络永远学不会曲线。这是最隐蔽也最致命的错误。
- **权重形状对不上**：层与层之间，前一层的输出维度必须等于后一层的输入维度，否则 `W @ a` 会直接报 `shapes not aligned`。搭网络时在心里过一遍每一层的"进几维、出几维"。
- **所有权重初始化成同一个值（比如全 0）**：那样同一层里每个神经元算出来的东西完全一样，更新也完全一样，它们会永远"长得一模一样"，等于白白浪费了那么多神经元。所以要用随机初始化打破这种对称。
- **以为层数越多越好**：层数加深确实更强，但也更难训、更容易过拟合。基础阶段一两层隐藏层往往就够用，别一上来就堆十几层。
- **Sigmoid 用在很深的隐藏层里**：Sigmoid 在两端非常平、导数接近 0，深层网络里会让梯度越传越小（梯度消失）。这也是现代网络隐藏层普遍改用 ReLU 的原因。

## 它在后面会怎么用到

这一节搭好了"网络长什么样、怎么做预测"的骨架，但它现在还只是个不会学习的空壳。接下来几节会逐块给它装上"学习"的能力：

- **怎么把权重调准**：[2.2 梯度下降与反向传播](2-2-2-backprop.md) 会用链式法则把误差从输出层一层层回传，告诉每个权重该往哪调——这正是前向传播的"逆过程"。
- **激活函数和损失函数怎么选**：[2.3 激活函数与损失函数](2-2-3-activations-losses.md) 会系统比较 ReLU、Sigmoid、Tanh 各自的脾气，以及分类、回归该配什么损失。
- **每一层的矩阵运算**：底层全靠 [3.1 线性代数](../ch1-intro/1-3-1-linear-algebra.md) 的 $\mathbf{y}=W\mathbf{x}+\mathbf{b}$，而求梯度要用到 [3.2 微积分与梯度](../ch1-intro/1-3-2-calculus-gradients.md) 的链式法则。
- **用框架真正训起来**：[2.4 PyTorch 基础](2-2-4-pytorch-basics.md) 会接着库版代码往下走，把训练循环补完整。

## 练习

??? note "基础练习"
    1. 在上面的 NumPy 前向传播里，把隐藏层的 ReLU 换成"什么都不做"（即直接 `a1 = z1`），手动验证：此时整个两层网络等价于一个单层线性模型。试着把等效的那一个权重矩阵算出来。
    2. 给单个神经元设 $\mathbf{w}=(1,1)$、$b=-1.5$，激活用阶跃函数（$z>0$ 输出 1，否则 0）。分别代入 $(0,0)$、$(0,1)$、$(1,0)$、$(1,1)$，看看它实现的是哪一个逻辑门（与 / 或 / 异或）。
    3. 把库版里的 `nn.Sigmoid()` 去掉，观察输出的取值范围发生了什么变化，并解释为什么。

??? note "进阶练习"
    1. 用纯 NumPy 手写一个能处理"一批"样本的前向传播：输入是形状 $(N, 2)$ 的矩阵（$N$ 个样本一起算），输出形状 $(N, 1)$。提示：把 `W @ x` 改成 `X @ W.T`，想清楚每一步的形状。
    2. 经典难题：单个感知机为什么学不会异或？把异或的四个点画在坐标纸上，试着说明"找不到一条直线把两类分开"。再搭一个含 2 个隐藏神经元的小 MLP，手动凑一组权重让它正确实现异或——亲手体会"隐藏层把空间折叠开"是什么意思。

## 小结

- 一个**神经元**只做两件事：先线性加权求和得到 $z$，再用激活函数 $\sigma$ 把它掰弯——后者是它区别于线性回归的全部秘密。
- **MLP** 就是把神经元排成层、把层前后串起来；每一层都是一次 $\mathbf{z}=W\mathbf{a}+\mathbf{b}$ 加激活，整体是个"函数套函数"的复合函数。
- **非线性激活是命根子**：去掉它，叠多少层都退化成一层线性；有了它，配上足够的神经元，网络（理论上）能逼近任意连续函数。
- 这一节只完成了"前向预测"，**怎么把权重学准，留给下一节的反向传播**。

想把"网络在折叠空间"这件事看得更直观，强烈推荐去玩一下 [TensorFlow Playground](https://playground.tensorflow.org/)——它用动画让你亲眼看到隐藏层是怎么一步步把数据掰开的，比任何文字都管用。更系统的入门可读 [*Dive into Deep Learning*](https://d2l.ai/) 的多层感知机一章。
