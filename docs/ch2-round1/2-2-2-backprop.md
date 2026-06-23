# 2.2 梯度下降与反向传播

> **难度** ⭐⭐⭐☆☆ · **前置**：[3.2 微积分与梯度](../ch1-intro/1-3-2-calculus-gradients.md)、[2.1 感知机与 MLP](2-2-1-perceptron-mlp.md)

!!! abstract "读完这一节，你会"
    - 说清"梯度下降"和"反向传播"分别是什么、谁负责什么，不再把两者混为一谈
    - 在一个两层小网络上，亲手把前向、反向各推一遍，算出每个参数的梯度
    - 用 NumPy 从零写出反向传播，再用 PyTorch 的 `autograd` 一行对照，确认自己没推错
    - 理解"自动微分"到底替你省了什么力气

## 一句话先把两个词分清楚

学这一节之前，先把两个最容易被搅在一起的词理清楚，否则后面会越看越糊涂。

**梯度下降（gradient descent）**是一个**更新参数的策略**：它说的是"既然梯度指向上坡，那我就朝反方向迈一小步，损失就降一点"。它的公式你在 [3.2 微积分与梯度](../ch1-intro/1-3-2-calculus-gradients.md)里已经见过：

$$\theta \leftarrow \theta - \eta\,\nabla_\theta L.$$

这里 $\theta$ 是参数，$\eta$（读作 eta）是**学习率**，控制每步迈多大；$\nabla_\theta L$ 是损失对参数的梯度。

可问题来了：神经网络动辄几万、几百万个参数，这个梯度 $\nabla_\theta L$ 到底**怎么算出来**？这就是**反向传播（backpropagation，简称 backprop）**要回答的。换句话说，反向传播只负责一件事——**高效地把损失对每一个参数的梯度求出来**；求完之后，怎么用这些梯度去更新参数，那是梯度下降的活儿。

所以记住这句话就够了：**反向传播算梯度，梯度下降用梯度。** 一前一后，配合成一个完整的"学习"动作。

## 把网络看成一张计算图

看到"反向传播"这个词先别紧张。它的本体，就是你高中学过的**链式法则**，只不过被一层层、机械地重复套用而已。要把这件事看清楚，最好的工具是**计算图（computational graph）**。

什么叫计算图？就是把"一个复杂的计算"拆成一连串最小的基本运算（加、乘、平方、取指数……），每个运算画成一个节点，数据顺着箭头从左流到右。比如最简单的一条链 $x \to u \to L$，就是 $x$ 先算出中间量 $u$，$u$ 再算出损失 $L$。

为什么要这么画？因为一旦拆成了图，求梯度就变成了一件特别"傻"的机械活：**前向**时数据从左往右流、把每个节点的输出算出来并存下；**反向**时梯度从右往左流、在每个节点上把"上游传来的梯度"乘上"这个节点自己的局部导数"，再传给下游。下面这张图就把这一来一回画了出来：

<div class="diagram">
<svg viewBox="0 0 380 200" font-family="-apple-system, 'Segoe UI', sans-serif">
  <defs>
    <marker id="bf" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="var(--dia-blue)"/></marker>
    <marker id="bb" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="var(--dia-accent)"/></marker>
  </defs>
  <!-- 三个节点 -->
  <circle cx="55"  cy="95" r="22" fill="none" stroke="var(--dia-stroke)" stroke-width="2"/>
  <circle cx="190" cy="95" r="22" fill="none" stroke="var(--dia-stroke)" stroke-width="2"/>
  <circle cx="325" cy="95" r="22" fill="none" stroke="var(--dia-stroke)" stroke-width="2"/>
  <text x="55"  y="100" text-anchor="middle" font-size="15" fill="var(--dia-stroke)">x</text>
  <text x="190" y="100" text-anchor="middle" font-size="15" fill="var(--dia-stroke)">u</text>
  <text x="325" y="100" text-anchor="middle" font-size="15" fill="var(--dia-stroke)">L</text>
  <!-- 前向：上方蓝箭头，左到右 -->
  <path d="M80 82 L162 82" stroke="var(--dia-blue)" stroke-width="2" marker-end="url(#bf)"/>
  <path d="M215 82 L297 82" stroke="var(--dia-blue)" stroke-width="2" marker-end="url(#bf)"/>
  <text x="121" y="40" text-anchor="middle" font-size="12" fill="var(--dia-blue)">前向：算输出</text>
  <!-- 反向：下方橙箭头，右到左 -->
  <path d="M300 108 L218 108" stroke="var(--dia-accent)" stroke-width="2" marker-end="url(#bb)"/>
  <path d="M165 108 L83 108" stroke="var(--dia-accent)" stroke-width="2" marker-end="url(#bb)"/>
  <text x="259" y="178" text-anchor="middle" font-size="12" fill="var(--dia-accent)">反向：传梯度 dL/du</text>
  <text x="123" y="178" text-anchor="middle" font-size="12" fill="var(--dia-accent)">dL/dx</text>
</svg>
</div>
<p class="figure-caption">图 2.2-1：计算图上的前向（蓝，从左到右算输出）与反向（橙，从右到左传梯度）。反向传播就是沿着这条链一路乘局部导数。</p>

放到链式法则上看就一目了然：要算 $L$ 对最左边 $x$ 的导数，只要把链上每一节的局部导数乘起来——

$$\frac{\partial L}{\partial x}=\frac{\partial L}{\partial u}\cdot\frac{\partial u}{\partial x}.$$

网络再深，无非是这条链变长、变出分叉，乘法的步数变多，**道理一分一毫都没变**。

## 手推一个两层小网络

光说不练没感觉。我们就拿一个最小的网络，**把前向和反向各推一遍**，你推完就会觉得反向传播其实挺老实。

设网络是这样的：输入 $x$，先做一次线性变换 $z = wx + b$，再过一个 **sigmoid 激活** $a = \sigma(z) = \dfrac{1}{1+e^{-z}}$，最后用平方误差和真实标签 $y$ 比较：

$$L = \tfrac{1}{2}(a - y)^2.$$

这就是一条 $x \to z \to a \to L$ 的计算图。我们的目标是求 $\dfrac{\partial L}{\partial w}$ 和 $\dfrac{\partial L}{\partial b}$，好拿去做梯度下降。

**先前向**，把每个中间量算出来并记住（反向时要用）：先算 $z=wx+b$，再算 $a=\sigma(z)$，最后算 $L$。这一步没什么玄机，照式子代进去就行。

**再反向**，我们从最右边的 $L$ 出发，一节一节往左传。每到一节，就乘上这一节的局部导数：

第一节，$L$ 对 $a$：因为 $L=\tfrac12(a-y)^2$，所以 $\dfrac{\partial L}{\partial a}=a-y$。

第二节，$a$ 对 $z$：sigmoid 有个特别漂亮的导数 $\sigma'(z)=a(1-a)$（在 [3.2 微积分与梯度](../ch1-intro/1-3-2-calculus-gradients.md)的进阶练习里证过）。于是把它乘进来：

$$\frac{\partial L}{\partial z}=\frac{\partial L}{\partial a}\cdot\frac{\partial a}{\partial z}=(a-y)\,a(1-a).$$

第三节，$z$ 对 $w$ 和 $b$：因为 $z=wx+b$，所以 $\dfrac{\partial z}{\partial w}=x$、$\dfrac{\partial z}{\partial b}=1$。再乘一次，就到底了：

$$\boxed{\ \frac{\partial L}{\partial w}=(a-y)\,a(1-a)\,x,\qquad \frac{\partial L}{\partial b}=(a-y)\,a(1-a).\ }$$

你看，整个过程就是**把 $\dfrac{\partial L}{\partial z}$ 这个"中间梯度"先算出来，再分别乘上 $x$ 和 $1$**。这种"算一个中间梯度、然后往两个参数上分发"的模式，正是反向传播在大网络里反复做的事——它把重复的中间结果只算一次，省下海量计算，这也是它比"对每个参数单独求一次导"高效得多的根本原因。

## 从零用 NumPy 写一遍

道理推完了，我们动手把它写成代码，**严格按上面的公式来**，你会发现代码和手推几乎一一对应。先准备数据和参数：

```python
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# 一条样本：输入 x，真实标签 y
x, y = 1.5, 1.0
# 待训练的参数，随便初始化
w, b = 0.3, 0.0
lr = 0.5            # 学习率 η
```

接着写**前向**：照着 $z \to a \to L$ 一路算下来，并把中间量留着备用：

```python
z = w * x + b          # 线性
a = sigmoid(z)         # 激活
L = 0.5 * (a - y)**2   # 平方误差损失
```

然后是这一节的主角——**反向**。我们就把刚才推的三节局部导数，按从右到左的顺序乘起来：

```python
dL_da = a - y               # 第一节：L 对 a
da_dz = a * (1 - a)         # 第二节：sigmoid 的局部导数
dL_dz = dL_da * da_dz       # 链式相乘，得到中间梯度 ∂L/∂z

dL_dw = dL_dz * x           # 第三节：分发到 w（乘 x）
dL_db = dL_dz * 1.0         # 分发到 b（乘 1）
```

最后用**梯度下降**更新参数，这就闭环了：

```python
w -= lr * dL_dw
b -= lr * dL_db
print(f"loss={L:.4f}  dL_dw={dL_dw:.4f}  dL_db={dL_db:.4f}")
```

如果你把上面这四块包进一个循环跑几百轮，会看到 `loss` 稳稳地往下掉——这就是一个最小的"训练"。**反向传播算梯度、梯度下降用梯度**，这句话你现在应该能在代码里逐行指出来了。

## 库版对照：把 autograd 这件事看明白

手推一遍是为了理解，可真到了几十层的大网络，没人愿意一节一节手算局部导数——太容易漏乘、错乘了。这时**自动微分（automatic differentiation，autodiff）**就登场了。

它的思路其实就是我们前面做的事，只不过交给框架去做：你**只写前向**，框架在背后悄悄记下整张计算图（每个运算用了谁、是什么运算），等你说一声"求导"，它就自动沿着图反向走一遍、把链式法则替你乘到底。PyTorch 里这套机制叫 `autograd`。我们用它把刚才的同一个例子算一遍，对照结果：

```python
import torch

x = torch.tensor(1.5)
y = torch.tensor(1.0)
# requires_grad=True：告诉 PyTorch「请盯着这两个量，我待会要对它们求导」
w = torch.tensor(0.3, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

# 只写前向，计算图由 PyTorch 自动搭建
z = w * x + b
a = torch.sigmoid(z)
L = 0.5 * (a - y)**2

L.backward()          # 一行：自动反向传播，把梯度算好放进 w.grad / b.grad
print(w.grad.item(), b.grad.item())
```

你会发现 `w.grad`、`b.grad` 打印出来的值，和上一节 NumPy 手算的 `dL_dw`、`dL_db` **一模一样**。这正说明 `backward()` 在背后做的，就是我们手推的那套链式相乘，分毫不差——只不过它对任意复杂的网络都能自动完成，你再也不用亲手推导。

那既然有 autograd，前面手推还有意义吗？太有意义了。**框架能替你算，但不能替你理解**：梯度为什么会爆炸、为什么会消失、某个激活函数为什么让训练卡死——这些问题的答案，全藏在你手推过的那条链里。竞赛里真正拉开差距的，往往就是这种"知其所以然"的判断力。

## 容易踩的坑

- **把梯度下降和反向传播当成一回事**：它俩是两个动作。反向传播负责"求出梯度"，梯度下降负责"用梯度更新参数"。面试和填空题最爱在这里设陷阱。
- **前向时忘了存中间量**：反向要用到 $a$、$z$ 这些前向算出的值。如果你为省内存把它们丢了，反向就没法乘局部导数了。框架帮你存着，自己手写时千万别漏。
- **PyTorch 里梯度会累加**：`backward()` 是把新梯度**加到** `.grad` 上，不是覆盖。所以训练循环里每轮开头要先 `optimizer.zero_grad()`（或手动清零），否则梯度越滚越大、训练直接崩。
- **该 detach 的地方没断开**：只要一个张量是从 `requires_grad=True` 的量算出来的，它就挂在计算图上。如果你只想看数值、不想让它参与求导，记得用 `.detach()` 或 `with torch.no_grad():`，否则既费内存又可能算错。
- **梯度爆炸 / 消失**：网络很深时，链式相乘会让梯度滚雪球般变得极大或极小。这不是 bug，而是链式法则的副作用，后面会用归一化、好的初始化、梯度裁剪来缓解（见 [2.5 优化与正则化](2-2-5-optimization-regularization.md)）。

## 它在后面会怎么用到

- **反向传播 = 链式法则的机械重复**，它的数学本体在 [3.2 微积分与梯度](../ch1-intro/1-3-2-calculus-gradients.md)里讲过；这一节是把那条法则真正用到网络上。
- 这里我们手动写了更新和清零，到了 [2.4 PyTorch 基础](2-2-4-pytorch-basics.md)，这些会被 `loss.backward()` 加 `optimizer.step()` 的标准三件套接管。
- 不同激活函数和损失函数，会给反向传播提供不同的"局部导数"，直接影响梯度好不好传（见 [2.3 激活函数与损失函数](2-2-3-activations-losses.md)）。
- 学习率怎么选、要不要加动量和正则，是梯度下降这一侧的学问（见 [2.5 优化与正则化](2-2-5-optimization-regularization.md)）。
- 到了 Round 2 的 Transformer，注意力机制同样靠反向传播来训练，你现在打下的底子那时会直接复用（见 [2.1 注意力机制](../ch3-round2/3-2-1-attention.md)）。

## 练习

??? note "基础练习"
    1. 把"从零 NumPy"那段包进一个 `for` 循环跑 300 轮，每 50 轮打印一次 `loss`，确认它确实在下降。再把学习率 `lr` 改成 `5.0` 和 `0.01`，分别观察损失曲线发生了什么。
    2. 不查上文，凭记忆默写出本节那个两层网络 $\dfrac{\partial L}{\partial w}$、$\dfrac{\partial L}{\partial b}$ 的最终表达式，再回来对答案。漏了哪一节的局部导数，就重点复习哪一节。
    3. 用一句话向同桌解释"反向传播和梯度下降的区别"，要求各自只说清它负责什么。

??? note "进阶练习"
    1. 把网络扩成**两层带隐藏单元**：$x \to h=\sigma(w_1x+b_1) \to a=\sigma(w_2h+b_2) \to L$。先在纸上手推 $\dfrac{\partial L}{\partial w_1}$（你会用到两次链式法则），再用 NumPy 实现，最后用 PyTorch 的 `backward()` 对照，确认三者一致。
    2. 写一个 **gradient check**：对同一个网络，分别用你手写的反向传播和[数值梯度](../ch1-intro/1-3-2-calculus-gradients.md)求一次梯度，输出两者的最大绝对误差。误差若大于 $10^{-5}$，说明你的反传写错了——这是调试反向传播最可靠的护身符。
    3. 把损失从平方误差换成二分类交叉熵，重新推一遍 $\dfrac{\partial L}{\partial z}$。你会惊喜地发现它化简成了非常干净的 $a-y$，想想这为什么是逻辑回归里偏爱交叉熵的原因之一。

## 小结

一句话记住这一节：**反向传播是链式法则在计算图上的机械重复，它把损失对每个参数的梯度高效地算出来；梯度下降再拿这些梯度，沿反方向一小步步更新参数。** 自己手推过一个小网络、再用 `autograd` 一行对上答案，你就同时拥有了"理解"和"工具"两样东西。

想把直觉建得更牢，强烈推荐看 3Blue1Brown 的《深度学习》系列第 3、4 集，它用动画把反向传播讲得无比清楚；想要权威推导，可读 [*Dive into Deep Learning*](https://d2l.ai/) 关于 backprop 与 autograd 的章节。
