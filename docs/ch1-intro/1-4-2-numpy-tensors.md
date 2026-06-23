# 1.4.2 NumPy 与张量

> **难度** ⭐⭐☆☆☆ · **前置**：[1.4.1 Python 基础](1-4-1-python.md)、[1.3.1 线性代数](1-3-1-linear-algebra.md)

NumPy 是 Python 科学计算的地基：它把"对一堆数做同样的运算"变得又快又简洁（向量化）。PyTorch 的张量（tensor）几乎就是"能跑在 GPU 上、能自动求导的 NumPy 数组"，API 高度相似，学好 NumPy 等于半只脚踏进 PyTorch。

!!! abstract "学习目标"
    - 会创建/索引/切片 `ndarray`，理解 `shape` 与 `dtype`
    - 掌握广播与向量化，能用它替代显式 for 循环
    - 说清 NumPy 数组与 PyTorch 张量的关系

## 一、直觉：它解决什么问题

要给一万个数都加 1，写 for 循环又慢又啰嗦。NumPy 让你直接 `a + 1`——一行搞定，底层用 C 加速，快上几十倍。这种"对整块数据一次性运算"就是**向量化**。

## 二、核心用法速览

```python
import numpy as np

a = np.array([1, 2, 3])              # 一维
M = np.array([[1, 2], [3, 4]])       # 二维, shape=(2,2)
np.zeros((2, 3)); np.ones(3); np.arange(5); np.linspace(0, 1, 5)

print(M.shape, M.dtype)              # (2, 2) int64

# 索引与切片（支持多维、布尔）
print(M[0, 1])        # 2
print(M[:, 0])        # 第 0 列 -> [1 3]
print(a[a > 1])       # 布尔索引 -> [2 3]

# 向量化运算（逐元素，无需循环）
print(a + 1)          # [2 3 4]
print(a * 2)          # [2 4 6]

# 沿轴聚合：axis=0 按列、axis=1 按行
print(M.sum(axis=0))  # [4 6]
print(M.mean(axis=1)) # [1.5 3.5]

# 形状变换
print(np.arange(6).reshape(2, 3))
```

**广播**：形状不同的数组运算时，NumPy 自动把小的"拉伸"对齐。例如 `(3,1)` 与 `(1,4)` 相加得 `(3,4)`。

## 三、实战：向量化 vs 循环

```python
import numpy as np, time
x = np.random.rand(1_000_000)

t0 = time.time()
s = 0.0
for v in x: s += v * v          # 纯 Python 循环
print("loop:", time.time() - t0)

t0 = time.time()
s2 = (x * x).sum()              # 向量化：一行
print("vectorized:", time.time() - t0)   # 通常快几十倍
```

数据标准化也是一行（后面预处理常用）：

```python
x = np.array([[1., 2.], [3., 4.], [5., 6.]])
x_std = (x - x.mean(axis=0)) / x.std(axis=0)   # 每列零均值、单位方差
```

## 四、常见陷阱与调试

- **view vs copy**：切片返回的是**视图**，改它会改原数组；要独立副本用 `.copy()`。
- **广播规则**：从末尾维度对齐，维度要么相等、要么有一个是 1，否则报 `could not be broadcast`。先 `print(a.shape, b.shape)`。
- **整数 dtype**：`np.array([1,2,3])` 是 int，`/` 后仍可能意外；需要小数时写 `1.` 或 `astype(float)`。
- **`axis` 记反**：`axis=0` 是"压掉行、按列算"。记不住就用小数组试一下。

## 五、应用场景

- 一切数值计算的底座：特征工程、距离计算、矩阵运算。
- **PyTorch 张量**几乎同款 API：`torch.tensor`、`.shape`、广播、`@` 都通用；`torch.from_numpy()` 可互转（见 [2.2.4 PyTorch 基础](../ch2-round1/2-2-4-pytorch-basics.md)）。

## 六、练习题

??? note "基础练习"
    1. 生成 $10\times10$ 的随机矩阵，求每行最大值与每列均值。
    2. 不用循环，把向量里所有负数置零（提示：布尔索引或 `np.maximum`）。

??? note "进阶练习"
    1. 用广播一行算出两组点的两两欧氏距离矩阵（$A$ 是 $m\times d$，$B$ 是 $n\times d$，输出 $m\times n$）。
    2. 用 NumPy 实现一次 softmax（注意数值稳定：先减去每行最大值）。

## 七、小结与延伸阅读

- **向量化**用整块运算替代 for 循环，又快又简洁。
- **广播**让不同形状自动对齐，是简洁的关键，也是 bug 的来源。
- NumPy 与 PyTorch 张量同源，学一通两。

延伸：[NumPy 官方 quickstart](https://numpy.org/doc/stable/user/quickstart.html)；[100 NumPy Exercises](https://github.com/rougier/numpy-100)。
