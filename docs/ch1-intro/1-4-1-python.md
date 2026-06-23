# 1.4.1 Python 基础

> **难度** ⭐☆☆☆☆ · **前置**：任意一门编程语言基础更佳

Python 是 AI 的"普通话"：NumPy、Pandas、PyTorch 全是 Python 库，USAAIO 也在 Python（Colab）里考。这一节带你快速过一遍写 AI 代码会反复用到的语法，重点是**够用**，不求全。

!!! abstract "学习目标"
    - 熟练使用变量、控制流、函数、列表/字典与推导式
    - 能读懂并写出简单的类（后面 PyTorch 的 `nn.Module` 会用）
    - 避开新手最常踩的几个 Python 坑

## 一、直觉：它解决什么问题

你不需要成为 Python 专家，只需要能**流畅地把想法翻译成代码**：循环遍历数据、函数封装逻辑、字典做映射。Python 语法接近自然语言，正适合快速试验模型。

## 二、核心语法速览

```python
# 变量与类型（动态类型，无需声明）
n = 10                 # int
pi = 3.14              # float
name = "cat"           # str
ok = True              # bool

# 控制流（靠缩进分块，不是大括号）
for i in range(3):     # 0,1,2
    if i % 2 == 0:
        print(i, "偶")

# 函数
def square(x):
    return x * x

# 列表 / 字典 / 元组 / 集合
nums = [1, 2, 3]
d = {"cat": 1, "dog": 2}
t = (1, 2)             # 不可变
s = {1, 2, 2, 3}       # 去重 -> {1,2,3}

# 列表推导式（AI 代码里随处可见）
squares = [x * x for x in range(5)]      # [0,1,4,9,16]
even = [x for x in nums if x % 2 == 0]   # [2]
```

类用来封装"数据 + 行为"，PyTorch 的模型都是类：

```python
class Counter:
    def __init__(self, start=0):   # 构造函数
        self.count = start
    def add(self, x):
        self.count += x
        return self.count

c = Counter()
print(c.add(5))        # 5
```

## 三、实战：统计词频

```python
text = "cat dog cat bird dog cat"
freq = {}
for w in text.split():
    freq[w] = freq.get(w, 0) + 1     # get 给默认值，避免 KeyError
print(freq)            # {'cat': 3, 'dog': 2, 'bird': 1}

# 用推导式 + sorted 取最高频
top = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[0]
print(top)             # ('cat', 3)
```

## 四、常见陷阱与调试

- **可变默认参数**：`def f(x, acc=[])` 的 `[]` 只创建一次，会跨调用累积！默认用 `None`：`def f(x, acc=None): acc = acc or []`。
- **`is` vs `==`**：`==` 比值，`is` 比身份（是否同一对象）。判断相等永远用 `==`。
- **整除与浮点**：`5 / 2 == 2.5`，`5 // 2 == 2`（整除）；别混。
- **浅拷贝**：`b = a` 只是同一个列表的别名，改 `b` 会动 `a`；要副本用 `a.copy()` 或 `list(a)`。
- **缩进**：Tab 与空格混用会报 `IndentationError`，统一用 4 个空格。

## 五、应用场景

- 后续所有代码都是 Python：[1.4.2 NumPy](1-4-2-numpy-tensors.md)、[1.4.3 Pandas](1-4-3-pandas-data.md)、PyTorch 训练循环。
- `nn.Module` 子类化、`Dataset` 自定义、配置字典——都建立在本节语法上。

## 六、练习题

??? note "基础练习"
    1. 写函数 `count_vowels(s)` 返回字符串里元音字母个数（用集合判断）。
    2. 用字典推导式把 `["a","bb","ccc"]` 转成 `{"a":1,"bb":2,"ccc":3}`（键→长度）。

??? note "进阶练习"
    1. 实现一个 `Stack` 类，支持 `push/pop/peek/is_empty`。
    2. 用一行推导式生成九九乘法表的字符串列表，并打印成方阵。

## 七、小结与延伸阅读

- Python 动态类型、靠缩进分块、生态强大，是 AI 默认语言。
- **推导式 + 字典 + 类**是写模型代码的三件套。
- 记牢可变默认参数、`is`/`==`、浅拷贝三个经典坑。

延伸：[Python 官方教程](https://docs.python.org/zh-cn/3/tutorial/)；[Real Python](https://realpython.com/) 入门系列。
