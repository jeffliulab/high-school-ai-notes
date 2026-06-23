# 1.3.4 凸优化基础

> **难度** ⭐⭐⭐☆☆ · **前置**：[1.3.2 微积分与梯度](1-3-2-calculus-gradients.md)

"训练模型"="最小化损失函数"。如果损失是**凸函数**（碗形），随便从哪下山都能到全局最低；如果是非凸（坑坑洼洼），就可能卡在局部最低。理解这点，你才知道学习率为什么这么关键。

!!! abstract "学习目标"
    - 说清凸函数 vs 非凸函数的区别与后果
    - 写出梯度下降的更新公式，理解学习率 $\eta$ 的作用
    - 能用 NumPy 实现一维梯度下降，并复现"学习率过大→发散"

## 一、直觉：它解决什么问题

把损失想成地形。**凸**就是一个标准的碗：只有一个最低点，沿坡往下一定到底。**非凸**则有很多小坑，下山可能停在某个小坑（局部最优）而非真正谷底。

<div class="diagram">
<svg viewBox="0 0 360 170" font-family="-apple-system, 'Segoe UI', sans-serif">
  <!-- convex -->
  <path d="M30 30 Q95 185 160 30" fill="none" stroke="var(--dia-green)" stroke-width="2"/>
  <circle cx="95" cy="118" r="5" fill="var(--dia-accent)"/>
  <text x="55" y="150" font-size="12" fill="var(--dia-stroke-soft)">凸：唯一最低点</text>
  <!-- nonconvex -->
  <path d="M205 40 C225 120 245 60 265 100 S300 150 330 50" fill="none" stroke="var(--dia-blue)" stroke-width="2"/>
  <circle cx="245" cy="92" r="5" fill="var(--dia-accent)"/>
  <text x="225" y="150" font-size="12" fill="var(--dia-stroke-soft)">非凸：多个局部坑</text>
</svg>
</div>
<p class="figure-caption">图：凸函数只有一个全局最低点；非凸函数可能让梯度下降卡在局部最低。</p>

## 二、原理与数学推导

**凸函数定义**：对任意两点和 $\lambda\in[0,1]$，

$$f(\lambda x+(1-\lambda)y)\le \lambda f(x)+(1-\lambda)f(y),$$

直观就是"两点连线不低于函数曲线"。凸函数的**任何局部最小都是全局最小**。

**梯度下降**：从当前参数沿负梯度走一步，

$$\theta_{t+1}=\theta_t-\eta\,\nabla f(\theta_t),$$

其中**学习率** $\eta$ 决定步长：太小收敛慢，太大会越过谷底来回震荡甚至发散。

!!! example "例 1：手推一步"
    设 $f(x)=x^2$，则 $\nabla f=2x$。从 $x_0=4$、$\eta=0.1$ 出发：$x_1=4-0.1\cdot8=3.2$；$x_2=3.2-0.1\cdot6.4=2.56$……稳步逼近最低点 $0$。

## 三、代码实现

```python
import numpy as np

def gradient_descent(grad, x0, lr, steps):
    x = x0
    traj = [x]
    for _ in range(steps):
        x = x - lr * grad(x)      # θ ← θ - η ∇f
        traj.append(x)
    return np.array(traj)

grad = lambda x: 2 * x            # f(x) = x^2 的梯度

print(gradient_descent(grad, 4.0, lr=0.1, steps=5))   # 稳定收敛到 0
print(gradient_descent(grad, 4.0, lr=1.1, steps=5))   # lr 过大 → 来回放大、发散
```

运行第二行会看到数值越跳越大——这就是学习率过大导致的发散。

## 四、常见陷阱与调试

- **学习率最关键**：发散先把 $\eta$ 调小（÷10）；收敛太慢再调大。这是调参第一反应。
- **非凸≠没法练**：神经网络损失是非凸的，但实践中梯度下降配合好初始化/优化器依然好用——别因为"非凸"就放弃 GD。
- **特征未标准化**：各维尺度差异大时损失面"又长又窄"，GD 会来回横跳——先做标准化（见 [1.4.3 Pandas 与数据处理](1-4-3-pandas-data.md)）。
- **局部最优 vs 鞍点**：高维里更常见的障碍其实是鞍点，不是局部最优。

## 五、应用场景

- **所有训练**：从线性回归到 Transformer，更新公式都是 $\theta\leftarrow\theta-\eta\nabla f$（见 [2.2.5 优化与正则化](../ch2-round1/2-2-5-optimization-regularization.md)）。
- **凸模型**：线性回归、逻辑回归、SVM 的损失是凸的，保证能到全局最优（见 [2.1.2 线性回归](../ch2-round1/2-1-2-linear-regression.md)）。

## 六、练习题

??? note "基础练习"
    1. 对 $f(x)=x^2$，用 $\eta=0.5$ 从 $x_0=4$ 手算 3 步，观察是否更快收敛。
    2. 改 `gradient_descent` 记录每步的 $f(x)$，验证它单调下降（当 $\eta$ 合适时）。

??? note "进阶练习"
    1. 把 $f$ 换成二维 $f(x,y)=x^2+10y^2$（病态），观察 GD 在 $y$ 方向震荡；再对坐标做缩放后重试。
    2. 实现带动量的梯度下降 $v\leftarrow\beta v+\nabla f,\ \theta\leftarrow\theta-\eta v$，对比收敛速度。

## 七、小结与延伸阅读

- **凸 = 碗形**，局部最优即全局最优；非凸有多个坑。
- 训练的统一更新式：$\theta\leftarrow\theta-\eta\nabla f$，**学习率 $\eta$** 是第一调参旋钮。
- 发散先调小学习率，横跳先做特征标准化。

延伸：[*Mathematics for Machine Learning*](https://mml-book.github.io/) 第 7 章；Boyd《Convex Optimization》（进阶）。
