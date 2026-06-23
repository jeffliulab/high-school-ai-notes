# 1.3.1 线性代数

> **难度** ⭐⭐☆☆☆ · **前置**：高中向量与矩阵基础

线性代数是机器学习的"母语"：一张数据表就是一个矩阵，一次神经网络前向传播就是几次矩阵乘法。把它学扎实，后面的 PCA、线性回归、神经网络都会变得透明。

!!! abstract "学习目标"
    - 能用"向量=数据点、矩阵=线性变换"的视角理解数据与模型
    - 会手算矩阵乘法、转置、逆，理解特征值/特征向量与 SVD 的含义
    - 能用 NumPy 完成上述运算，并说出它们在 ML 里对应什么

## 一、直觉：它解决什么问题

一条数据（比如一朵花的"花瓣长、花瓣宽、花萼长、花萼宽"）就是一个**向量** $\mathbf{x}\in\mathbb{R}^4$；$N$ 条数据堆起来就是一个 $N\times 4$ 的**矩阵** $X$。而"模型"做的事，本质就是对向量做**线性变换**——乘一个矩阵 $W$。

<div class="diagram">
<svg viewBox="0 0 360 190" font-family="-apple-system, 'Segoe UI', sans-serif">
  <defs>
    <marker id="va" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="var(--dia-accent)"/></marker>
    <marker id="vb" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="var(--dia-blue)"/></marker>
  </defs>
  <!-- axes -->
  <line x1="40" y1="165" x2="330" y2="165" stroke="var(--dia-stroke-tertiary)" stroke-width="1"/>
  <line x1="40" y1="165" x2="40" y2="20" stroke="var(--dia-stroke-tertiary)" stroke-width="1"/>
  <!-- original vector v -->
  <line x1="40" y1="165" x2="130" y2="105" stroke="var(--dia-accent)" stroke-width="2" marker-end="url(#va)"/>
  <text x="135" y="100" font-size="13" fill="var(--dia-accent)">v</text>
  <!-- transformed Av -->
  <line x1="40" y1="165" x2="245" y2="60" stroke="var(--dia-blue)" stroke-width="2" marker-end="url(#vb)"/>
  <text x="250" y="56" font-size="13" fill="var(--dia-blue)">Av</text>
</svg>
</div>
<p class="figure-caption">图：矩阵 A 把向量 v 线性变换为 Av（旋转 + 缩放）。神经网络的每一层都在做这件事。</p>

## 二、原理与数学推导

**向量与矩阵**：$\mathbf{x}\in\mathbb{R}^n$ 是 $n$ 个数；$A\in\mathbb{R}^{m\times n}$ 是 $m$ 行 $n$ 列。

**矩阵乘法**（最核心）：$C=AB$ 中

$$C_{ij}=\sum_{k} A_{ik}B_{kj}$$

要求 $A$ 的列数 = $B$ 的行数。一个直观理解：$A\mathbf{x}$ 是用 $\mathbf{x}$ 的分量对 $A$ 的各列做加权求和。

**转置** $A^\top$：行列互换，$(A^\top)_{ij}=A_{ji}$。**逆** $A^{-1}$ 满足 $A^{-1}A=I$（仅方阵且可逆时存在）。

**特征值与特征向量**：若 $A\mathbf{v}=\lambda\mathbf{v}$，则 $\mathbf{v}$ 是特征向量、$\lambda$ 是特征值——表示"这个方向只被缩放、不被旋转"。

**奇异值分解 SVD**：任意矩阵都能拆成 $A=U\Sigma V^\top$，其中 $\Sigma$ 的对角元（奇异值）刻画"每个方向上拉伸了多少"。PCA 正是它的应用。

!!! example "例 1：矩阵 × 向量"
    设 $A=\begin{bmatrix}1&2\\3&4\end{bmatrix},\ \mathbf{x}=\begin{bmatrix}1\\1\end{bmatrix}$，则
    $$A\mathbf{x}=\begin{bmatrix}1\cdot1+2\cdot1\\3\cdot1+4\cdot1\end{bmatrix}=\begin{bmatrix}3\\7\end{bmatrix}.$$

## 三、代码实现

```python
import numpy as np

A = np.array([[1., 2.],
              [3., 4.]])
x = np.array([1., 1.])

# 矩阵 × 向量：用 @（推荐），不要用 *（那是逐元素乘）
print(A @ x)            # [3. 7.]
print(A.T)             # 转置
print(np.linalg.inv(A))  # 逆

# 特征值/特征向量
vals, vecs = np.linalg.eig(A)
print(vals)            # 两个特征值

# SVD：U, 奇异值 s, V^T
U, s, Vt = np.linalg.svd(A)
print(s)               # 奇异值（降序）
```

## 四、常见陷阱与调试

- **`@` vs `*`**：`A @ x` 是矩阵乘法；`A * x` 是逐元素乘（且会触发广播），二者结果完全不同。
- **形状不匹配**：`(m,n) @ (n,)` 合法，`(m,n) @ (m,)` 报 `shapes not aligned`——先 `print(A.shape, x.shape)` 检查。
- **逆不要随便求**：解 $A\mathbf{x}=\mathbf{b}$ 用 `np.linalg.solve(A, b)` 比 `inv(A) @ b` 更稳更快。
- **整数数组**：`np.array([[1,2]])` 是 int，做除法会被截断；显式写成浮点 `1.`。

## 五、应用场景

- **线性回归**：正规方程 $\hat{\mathbf{w}}=(X^\top X)^{-1}X^\top\mathbf{y}$ 全是矩阵运算（见 [2.1.2 线性回归](../ch2-round1/2-1-2-linear-regression.md)）。
- **神经网络**：每一层就是 $\mathbf{y}=W\mathbf{x}+\mathbf{b}$（见 [2.2.1 感知机与 MLP](../ch2-round1/2-2-1-perceptron-mlp.md)）。
- **PCA**：对协方差矩阵做特征分解 / 对数据做 SVD 来降维（见 [2.1.8 无监督学习](../ch2-round1/2-1-8-unsupervised.md)）。

## 六、练习题

??? note "基础练习"
    1. 手算 $\begin{bmatrix}2&0\\0&3\end{bmatrix}\begin{bmatrix}1\\1\end{bmatrix}$，并说明它对向量做了什么几何操作。
    2. 用 NumPy 验证 $(AB)^\top=B^\top A^\top$（随机生成 $A,B$）。

??? note "进阶练习"
    1. 自己写一个不用 `@` 的 `matmul(A, B)`（三重循环），再与 `A @ B` 对比结果与耗时。
    2. 对一张 $50\times 50$ 的灰度图做 SVD，只保留前 $k$ 个奇异值重建，观察 $k=5,20$ 时的清晰度——这就是"低秩压缩"。

## 七、小结与延伸阅读

- 数据是向量、模型是矩阵变换；**矩阵乘法**是一切的核心运算。
- 特征值/SVD 描述"哪个方向被拉伸多少"，是 PCA 的数学基础。
- 用 `@` 做矩阵乘、用 `solve` 解方程，别滥用求逆。

延伸：[*Mathematics for Machine Learning*](https://mml-book.github.io/) 第 2–4 章；3Blue1Brown《线性代数的本质》系列。
