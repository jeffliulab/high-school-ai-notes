# 2.4 PyTorch 基础

> **难度** ⭐⭐⭐☆☆ · **前置**：[4.2 NumPy 与张量](../ch1-intro/1-4-2-numpy-tensors.md)、[2.2 梯度下降与反向传播](2-2-2-backprop.md)

!!! abstract "读完这一节，你会"
    - 知道张量（tensor）和 NumPy 数组的关系，会在两者之间互相转换
    - 理解 `autograd` 是怎么帮你"免费"算出梯度的，会读懂 `loss.backward()` 干了什么
    - 能完整写出一个最小训练循环：定义模型 → 前向 → 算损失 → 反向 → 更新参数
    - 明白 `optimizer.zero_grad()` 为什么不能省，以及省了会发生什么

## 为什么要专门学一个 PyTorch

前面我们已经亲手推过梯度、也手写过反向传播（见 [2.2 梯度下降与反向传播](2-2-2-backprop.md)）。你大概也体会到了一件事：**网络只要稍微深一点，手推每个参数的梯度就会变成一场噩梦。** 链式法则本身不难，难的是层数一多，要乘的项一多，人就很容易抄错、漏乘、符号写反。

PyTorch 解决的正是这个痛点。你只管把"前向计算"按正常思路写出来——输入怎么一步步变成输出、再变成损失——剩下那套繁琐的"从后往前逐层求导"，它替你自动完成，而且又快又准。换句话说，**你负责描述模型在算什么，PyTorch 负责算梯度**。这就是为什么几乎所有深度学习代码都建立在它（或它的同类 TensorFlow、JAX）之上。

那 PyTorch 的世界里，最基本的"砖块"是什么？是张量。

## 张量：会算梯度、能上 GPU 的 NumPy 数组

如果你已经熟悉 NumPy（见 [4.2 NumPy 与张量](../ch1-intro/1-4-2-numpy-tensors.md)），那理解张量几乎不费力，因为可以记住一句话：**张量（tensor）就是一个 NumPy 数组，外加两项超能力——能放到 GPU 上飞快地算，还能自动记录运算、帮你求梯度。**

用法上它和 NumPy 也长得像，常用的创建方式几乎一一对应：

```python
import torch

a = torch.tensor([[1., 2.],
                  [3., 4.]])     # 直接从数据建张量
b = torch.zeros(2, 3)            # 全 0，形状 (2, 3)
c = torch.randn(2, 3)           # 标准正态随机数

print(a @ b.T)                  # 矩阵乘法照样用 @，和 NumPy 一致
```

和 NumPy 互转也只是一行的事，这在实际项目里很常用——数据清洗阶段用 Pandas/NumPy，喂进模型前再转成张量：

```python
import numpy as np

arr = np.array([1., 2., 3.])
t   = torch.from_numpy(arr)     # NumPy → 张量
back = t.numpy()                # 张量 → NumPy
```

有一点要特别留意：张量是有**数据类型（dtype）**的。神经网络默认用 32 位浮点 `float32`，而 NumPy 默认是 64 位的 `float64`，两者混用时偶尔会报类型不匹配的错。养成习惯，建张量时该写 `1.`（小数）就写 `1.`，必要时显式 `.float()` 转一下。

## autograd：你写前向，它记一张计算图

张量真正神奇的地方，是它能"记账"。

当你创建一个张量时加上 `requires_grad=True`，就等于告诉 PyTorch：**"这个量是我要优化的参数，请帮我盯着它，凡是用到它的运算都给我记下来。"** 从此以后，每做一步运算，PyTorch 都在背后悄悄搭一张**计算图（computation graph）**，把"谁是谁算出来的"这条链路完整记录下来。

为什么要记这张图？因为求梯度靠的是链式法则，而链式法则需要"从结果一层层往回追溯到源头"。只要图在，PyTorch 就能顺着它从损失一路回传到每个参数。下面这张图把这个过程画了出来，你可以先顺着实线箭头往右走一遍（前向），再顺着虚线箭头往左走一遍（反向）：

<div class="diagram">
<svg viewBox="0 0 420 210" font-family="-apple-system, 'Segoe UI', sans-serif">
  <defs>
    <marker id="fwd" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="var(--dia-blue)"/></marker>
    <marker id="bwd" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="var(--dia-accent)"/></marker>
  </defs>
  <!-- 节点 -->
  <circle cx="55"  cy="95" r="24" fill="none" stroke="var(--dia-stroke-soft)" stroke-width="1.5"/>
  <text x="55" y="100" text-anchor="middle" font-size="15" fill="var(--dia-stroke-soft)">w</text>
  <circle cx="200" cy="95" r="24" fill="none" stroke="var(--dia-stroke-soft)" stroke-width="1.5"/>
  <text x="200" y="100" text-anchor="middle" font-size="14" fill="var(--dia-stroke-soft)">z=wx</text>
  <circle cx="350" cy="95" r="24" fill="none" stroke="var(--dia-stroke-soft)" stroke-width="1.5"/>
  <text x="350" y="100" text-anchor="middle" font-size="15" fill="var(--dia-stroke-soft)">L</text>
  <!-- 前向箭头（上方走线，文字在更上方空白处） -->
  <line x1="80"  y1="88" x2="175" y2="88" stroke="var(--dia-blue)" stroke-width="2" marker-end="url(#fwd)"/>
  <line x1="225" y1="88" x2="325" y2="88" stroke="var(--dia-blue)" stroke-width="2" marker-end="url(#fwd)"/>
  <text x="210" y="40" text-anchor="middle" font-size="12" fill="var(--dia-blue)">前向：算出损失 L</text>
  <!-- 反向箭头（下方走线，文字在更下方空白处） -->
  <line x1="325" y1="115" x2="225" y2="115" stroke="var(--dia-accent)" stroke-width="2" stroke-dasharray="5 3" marker-end="url(#bwd)"/>
  <line x1="175" y1="115" x2="80"  y2="115" stroke="var(--dia-accent)" stroke-width="2" stroke-dasharray="5 3" marker-end="url(#bwd)"/>
  <text x="210" y="160" text-anchor="middle" font-size="12" fill="var(--dia-accent)">反向：backward() 把 dL/dw 回传到 w</text>
</svg>
</div>
<p class="figure-caption">图 2.4-1：前向计算搭出一张图（蓝色实线），调用 backward() 后梯度沿原路回传（橙色虚线）。</p>

我们用一个能手算验证的小例子，看看它到底准不准。取最简单的 $L=w^2$，在 $w=3$ 处，它的导数 $\dfrac{dL}{dw}=2w=6$。我们让 PyTorch 自己算一遍：

```python
w = torch.tensor(3.0, requires_grad=True)   # 盯住这个参数
L = w ** 2                                   # 前向：搭出计算图

L.backward()                                 # 反向：自动求导
print(w.grad)                                # 梯度被存进 w.grad，输出 tensor(6.)
```

跑完你会看到 `w.grad` 正好是 `6`，和我们手算的一模一样。注意梯度并不是 `backward()` 的返回值，而是被**累加进了 `w.grad` 这个属性里**——这个"累加"的细节后面会反复咬人，先记住它。

## 一个最小训练循环：五步走

光会算梯度还不够，训练模型是要把"算梯度"和"用梯度更新参数"这两件事，在一份数据上**反复循环很多遍**。这个循环是所有深度学习代码的骨架，无论模型多复杂，骨架永远是这五步：

<div class="diagram">
<svg viewBox="0 0 440 150" font-family="-apple-system, 'Segoe UI', sans-serif">
  <defs>
    <marker id="step" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="var(--dia-stroke-soft)"/></marker>
  </defs>
  <!-- 五个步骤方块 -->
  <rect x="10"  y="42" width="68" height="34" rx="5" fill="none" stroke="var(--dia-blue)"   stroke-width="1.5"/>
  <text x="44"  y="63" text-anchor="middle" font-size="12" fill="var(--dia-blue)">①前向</text>
  <rect x="100" y="42" width="68" height="34" rx="5" fill="none" stroke="var(--dia-blue)"   stroke-width="1.5"/>
  <text x="134" y="63" text-anchor="middle" font-size="12" fill="var(--dia-blue)">②算损失</text>
  <rect x="190" y="42" width="78" height="34" rx="5" fill="none" stroke="var(--dia-accent)" stroke-width="1.5"/>
  <text x="229" y="63" text-anchor="middle" font-size="11" fill="var(--dia-accent)">③清零梯度</text>
  <rect x="290" y="42" width="68" height="34" rx="5" fill="none" stroke="var(--dia-accent)" stroke-width="1.5"/>
  <text x="324" y="63" text-anchor="middle" font-size="12" fill="var(--dia-accent)">④反向</text>
  <rect x="380" y="42" width="50" height="34" rx="5" fill="none" stroke="var(--dia-green)"  stroke-width="1.5"/>
  <text x="405" y="63" text-anchor="middle" font-size="12" fill="var(--dia-green)">⑤更新</text>
  <!-- 顺序箭头 -->
  <line x1="78"  y1="59" x2="98"  y2="59" stroke="var(--dia-stroke-soft)" stroke-width="1.3" marker-end="url(#step)"/>
  <line x1="168" y1="59" x2="188" y2="59" stroke="var(--dia-stroke-soft)" stroke-width="1.3" marker-end="url(#step)"/>
  <line x1="268" y1="59" x2="288" y2="59" stroke="var(--dia-stroke-soft)" stroke-width="1.3" marker-end="url(#step)"/>
  <line x1="358" y1="59" x2="378" y2="59" stroke="var(--dia-stroke-soft)" stroke-width="1.3" marker-end="url(#step)"/>
  <!-- 回到开头的循环箭头（走下方，文字在更下方空白处） -->
  <path d="M405 78 L405 120 L44 120 L44 80" fill="none" stroke="var(--dia-stroke-tertiary)" stroke-width="1.3" stroke-dasharray="4 3" marker-end="url(#step)"/>
  <text x="225" y="138" text-anchor="middle" font-size="11" fill="var(--dia-stroke-tertiary)">重复很多个 epoch</text>
</svg>
</div>
<p class="figure-caption">图 2.4-2：最小训练循环的五个步骤——前向、算损失、清零梯度、反向、更新，循环往复。</p>

我们用一个具体任务把这五步落到代码上：拟合一条直线 $y=2x+1$。先准备数据和"零件"——模型、损失函数、优化器：

```python
import torch
import torch.nn as nn

# 造一批数据：真实关系是 y = 2x + 1，再加一点噪声
x = torch.randn(100, 1)
y = 2 * x + 1 + 0.1 * torch.randn(100, 1)

model = nn.Linear(1, 1)                 # 一个线性层：y = wx + b，参数 w、b 由它管理
loss_fn = nn.MSELoss()                  # 均方误差，回归任务的标准损失
optimizer = torch.optim.SGD(            # 优化器：负责"拿梯度去更新参数"
    model.parameters(), lr=0.1)
```

这里有三个新角色：`nn.Linear` 帮你把参数 $w$、$b$ 都建好、管好，省得你手动维护；`MSELoss` 就是我们在 [2.3 激活函数与损失函数](2-2-3-activations-losses.md) 里讲过的均方误差；`optimizer` 则是那个真正动手改参数的人——你把梯度算好，它按 $\theta\leftarrow\theta-\eta\nabla$ 帮你走一步。

零件齐了，循环就是把图 2.4-2 的五步照抄下来，重复几百遍：

```python
for epoch in range(200):
    pred = model(x)                 # ① 前向：模型算出预测
    loss = loss_fn(pred, y)         # ② 算损失：预测和真值差多少

    optimizer.zero_grad()           # ③ 清零：把上一轮残留的梯度抹掉
    loss.backward()                 # ④ 反向：autograd 自动算出每个参数的梯度
    optimizer.step()                # ⑤ 更新：优化器按梯度走一步

    if epoch % 50 == 0:
        print(f"epoch {epoch}, loss {loss.item():.4f}")

# 训练完，看看学到的 w 和 b 是不是接近真实的 2 和 1
print(model.weight.item(), model.bias.item())   # ≈ 2.0, ≈ 1.0
```

跑下来你会看到 loss 一路下降，最后打印出的 `weight` 和 `bias` 非常接近真实的 `2` 和 `1`——模型确实从带噪声的数据里把那条直线"猜"了出来。这就是一次完整的训练。

## 库版对照：手写循环 vs nn.Module

上面的写法对单层模型够用了，但真到了多层网络，把每一层都散着写会很乱。PyTorch 推荐的做法，是把模型封装成一个 `nn.Module` 的子类——你在 `__init__` 里把要用的层都登记好，再在 `forward` 里写清"数据怎么一层层流过去"。作为对照，下面把同一个线性回归改写成标准的类形式：

```python
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)   # 在这里登记要用到的层

    def forward(self, x):
        return self.fc(x)           # 在这里写前向：数据怎么流

model = LinearModel()
# 训练循环和上面那五步一字不差，这正是它的好处：骨架永远不变
```

你会发现训练循环那五步完全没动。**这正是 PyTorch 设计的精髓：不管模型从一层变成一百层，"前向→损失→清零→反向→更新"的骨架始终不变。** 你以后写卷积网络、Transformer，套的还是这同一个壳子——所以现在把这五步刻进肌肉记忆，绝对值得。

## 容易踩的坑

- **忘了 `optimizer.zero_grad()`**：还记得前面说梯度是"累加"进 `.grad` 的吗？如果不在每轮开头清零，这一轮的梯度会叠在上一轮上面，越积越大，训练直接崩掉。这是新手第一名的 bug。
- **顺序写反**：必须是先 `zero_grad()`、再 `backward()`、最后 `step()`。把 `step()` 放到 `backward()` 前面，等于拿着空梯度去更新，参数纹丝不动。
- **`loss` 直接当数字用**：`loss` 是个带计算图的张量，想打印或记录它的数值，要用 `loss.item()` 取出纯 Python 浮点数，否则你会把整张计算图都留在内存里，越跑越慢。
- **dtype 不一致**：`float32` 的模型遇上 `float64` 的输入会报错。数据进模型前统一 `.float()` 一下最省心。
- **评估时忘了关梯度**：只做预测、不训练时，记得套在 `with torch.no_grad():` 里，告诉 PyTorch 别再记账，能省下大量显存、也跑得更快。

## 它在后面会怎么用到

这五步训练循环，是后面**每一个**深度学习模型的共同骨架，差别只在"模型那一层换成什么"：

- 把 `nn.Linear` 堆成几层、中间夹上激活函数，就是 [2.1 感知机与 MLP](2-2-1-perceptron-mlp.md) 里的多层网络。
- 把线性层换成卷积层，就能跑 [3.1 卷积与池化](2-3-1-conv-pooling.md) 和 [3.2 图像分类实战](2-3-2-image-classification.md)。
- 优化器从最朴素的 SGD 换成 Adam、再加上正则化，是 [2.5 优化与正则化](2-2-5-optimization-regularization.md) 的主题。
- 到了 Round 2，同一套循环只要把数据和模型搬上 GPU，就是 [1.1 GPU 训练](../ch3-round2/3-1-1-gpu-training.md)。

## 练习

??? note "基础练习"
    1. 建一个 `w = torch.tensor(2.0, requires_grad=True)`，令 $L=3w^2+2w$，调用 `backward()`，验证 `w.grad` 是否等于你手算的 $6w+2=14$。
    2. 把正文里拟合 $y=2x+1$ 的代码跑起来，故意**删掉 `optimizer.zero_grad()` 这一行**，观察 loss 会怎样变化，亲眼见识一下"不清零"的后果。

??? note "进阶练习"
    1. 不用 `torch.optim`，自己手写更新步骤：在 `with torch.no_grad():` 里对每个参数做 `p -= lr * p.grad`，再手动 `p.grad.zero_()`。跑通后你就彻底明白 `optimizer.step()` 在背后做了什么。
    2. 把模型改写成一个两层的 `nn.Module`（中间夹一个 `nn.ReLU`），用它去拟合 $y=x^2$ 这种非线性关系，看看比单层线性模型强多少。

## 小结

- 张量就是"能上 GPU、能自动求导的 NumPy 数组"，用法和 NumPy 高度相似，互转只需一行。
- `requires_grad=True` 让 PyTorch 记下计算图，`loss.backward()` 沿图回传、把梯度存进每个参数的 `.grad`。
- 训练循环永远是五步：**前向 → 算损失 → 清零梯度 → 反向 → 更新**，模型再复杂，骨架不变。
- 头号大坑是忘了 `optimizer.zero_grad()`——梯度会累加，不清零必崩。

想系统过一遍，强烈推荐官方的 [PyTorch 60 分钟入门](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)，它用的正是这套张量 + autograd + 训练循环的脉络，跟着敲一遍就能上手。
