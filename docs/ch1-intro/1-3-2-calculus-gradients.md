# 1.3.2 微积分与梯度

> **难度** ⭐⭐☆☆☆ · **前置**：[1.3.1 线性代数](1-3-1-linear-algebra.md)、高中导数

训练模型本质上就一句话：**沿着梯度的反方向，一步步把损失"走"到最低点**。理解梯度，你就理解了梯度下降和反向传播的共同核心。

!!! abstract "学习目标"
    - 理解导数、偏导、梯度的几何含义（梯度 = 最陡上升方向）
    - 会用链式法则把复合函数的导数拆开
    - 能用 NumPy 数值法估计梯度，并解释它和解析梯度的关系

## 一、直觉：它解决什么问题

把"损失函数"想成一片山地：高度 = 损失，坐标 = 模型参数。我们要找最低的谷底。**梯度** $\nabla f$ 指向**上升最快**的方向，那么它的反方向 $-\nabla f$ 就是**下降最快**的方向——朝那个方向迈一小步，损失就降一点。

<div class="diagram">
<svg viewBox="0 0 360 190" font-family="-apple-system, 'Segoe UI', sans-serif">
  <defs>
    <marker id="ga" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="var(--dia-accent)"/></marker>
  </defs>
  <path d="M30 40 Q180 250 330 40" fill="none" stroke="var(--dia-blue)" stroke-width="2"/>
  <!-- descending balls -->
  <circle cx="70" cy="108" r="5" fill="var(--dia-accent)"/>
  <circle cx="110" cy="138" r="5" fill="var(--dia-accent)"/>
  <circle cx="150" cy="153" r="5" fill="var(--dia-accent)"/>
  <circle cx="180" cy="156" r="6" fill="var(--dia-accent-deep)"/>
  <path d="M76 112 L104 134" stroke="var(--dia-accent)" stroke-width="1.5" marker-end="url(#ga)"/>
  <path d="M116 142 L144 150" stroke="var(--dia-accent)" stroke-width="1.5" marker-end="url(#ga)"/>
  <text x="186" y="150" font-size="12" fill="var(--dia-stroke-soft)">最低点</text>
  <text x="40" y="36" font-size="12" fill="var(--dia-stroke-soft)">损失</text>
</svg>
</div>
<p class="figure-caption">图：训练 = 沿损失曲线的负梯度方向一步步"下山"到最低点。</p>

## 二、原理与数学推导

**导数**：$f'(x)=\lim_{h\to0}\dfrac{f(x+h)-f(x)}{h}$，即曲线在某点的斜率。

**偏导与梯度**：多元函数 $f(\mathbf{x})$ 对每个分量求偏导，拼成向量就是**梯度**：

$$\nabla f=\Big(\frac{\partial f}{\partial x_1},\dots,\frac{\partial f}{\partial x_n}\Big).$$

**链式法则**（反向传播的命根子）：若 $y=g(u),\ u=h(x)$，则

$$\frac{dy}{dx}=\frac{dy}{du}\cdot\frac{du}{dx}.$$

**Jacobian / Hessian**（了解即可）：向量值函数的一阶导排成矩阵是 Jacobian；二阶偏导排成矩阵是 Hessian，描述曲率。

!!! example "例 1：手算一个梯度"
    设 $f(x,y)=x^2+3y^2$。则 $\dfrac{\partial f}{\partial x}=2x,\ \dfrac{\partial f}{\partial y}=6y$，所以 $\nabla f=(2x,\,6y)$。在点 $(1,1)$ 处 $\nabla f=(2,6)$——$y$ 方向更陡。

## 三、代码实现

```python
import numpy as np

def f(x):                      # f(x, y) = x^2 + 3y^2
    return x[0]**2 + 3*x[1]**2

def numerical_grad(f, x, eps=1e-5):
    """中心差分估计梯度：对每个分量加减一点点看变化。"""
    g = np.zeros_like(x)
    for i in range(len(x)):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps; xm[i] -= eps
        g[i] = (f(xp) - f(xm)) / (2 * eps)
    return g

x = np.array([1.0, 1.0])
print(numerical_grad(f, x))    # ≈ [2. 6.]，与解析解 (2,6) 吻合
```

> 数值梯度只用来**验证**解析梯度是否写对（"gradient check"）；真正训练时用解析梯度（PyTorch 的 `autograd` 自动算）。

## 四、常见陷阱与调试

- **`eps` 取值**：太大不准、太小被浮点误差吞掉，$10^{-4}\sim10^{-6}$ 较稳。
- **链式法则方向**：复合函数从外往里逐层乘，漏一层就错——反向传播报错常源于此。
- **就地修改**：`xp = x` 不是拷贝！必须 `x.copy()`，否则改 `xp` 会污染 `x`。
- **梯度爆炸/消失**：深层网络里链式相乘会让梯度过大或过小，后面用归一化/裁剪缓解。

## 五、应用场景

- **梯度下降**：$\theta\leftarrow\theta-\eta\nabla f$，一切训练的更新公式（见 [1.3.4 凸优化基础](1-3-4-convex-optimization.md)）。
- **反向传播**：神经网络用链式法则逐层回传梯度（见 [2.2.2 梯度下降与反向传播](../ch2-round1/2-2-2-backprop.md)）。

## 六、练习题

??? note "基础练习"
    1. 求 $f(x)=\ln(1+e^{x})$（softplus）的导数，并说明它为什么总在 $(0,1)$ 之间。
    2. 用 `numerical_grad` 验证 $f(x,y)=\sin(x)+xy$ 在 $(0,1)$ 的梯度。

??? note "进阶练习"
    1. 用链式法则手推 $\sigma(x)=\frac{1}{1+e^{-x}}$ 的导数，证明 $\sigma'=\sigma(1-\sigma)$（逻辑回归会用到）。
    2. 写一个梯度检查函数，对比你手写的解析梯度与数值梯度的最大误差。

## 七、小结与延伸阅读

- 梯度指向上升最快方向，**负梯度**就是下山方向——这是训练的引擎。
- **链式法则**把复合函数的导数拆成逐层相乘，是反向传播的数学本体。
- 数值梯度用来查错，解析梯度用来训练。

延伸：[*Mathematics for Machine Learning*](https://mml-book.github.io/) 第 5 章；3Blue1Brown《微积分的本质》。
