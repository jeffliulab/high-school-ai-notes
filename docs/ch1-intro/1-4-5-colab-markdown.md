# 1.4.5 Colab 与 Markdown

> **难度** ⭐☆☆☆☆ · **前置**：[1.4.1 Python 基础](1-4-1-python.md)

!!! abstract "读完这一节，你会"
    - 熟悉 Colab 的单元、运行时（CPU/GPU）和几个常用操作
    - 用 Markdown 加 LaTeX 在文本单元里写清楚数学解答
    - 知道 Colab 的会话限制，避免在考场上丢掉辛苦跑出的进度

## 考试就在 Colab 里

很多人把 Colab 当成一个"周边工具"，但对 USAAIO 来说，它是**考场本身**。比赛会发给你一个 Google Colab 文件，你就在浏览器里写代码、跑模型、交答案，全程不用在自己电脑上装任何环境。

更重要的是，Round 1 有一类"非编码题"，明确要求你**在文本单元里用 Markdown 和 LaTeX 写出数学推导**。也就是说，你不仅要会让代码跑出结果，还要会用文字和公式把思路讲清楚。所以 Colab 加 Markdown 不是锦上添花的技能，而是实打实的得分点。这一节我们就把它讲透。

## 两种单元：代码和文本

Colab 的文件由一个个"单元（cell）"组成，只有两种：

- **代码单元**：写 Python，按 `Shift + Enter` 运行，结果显示在下方。
- **文本单元**：写 Markdown，渲染成排版好的文字，可以有标题、列表、表格和公式。

一份答卷往往是这两种单元交替出现：用文本单元讲思路，用代码单元做实现。

## 切换 CPU 和 GPU

Colab 免费提供 GPU，但默认是关着的。切换的入口在菜单里：

> 代码执行程序（Runtime） → 更改运行时类型 → 选择 CPU 或 GPU

要提醒的是，**Round 1 只允许用 CPU，Round 2 才可以用 GPU**。想确认 GPU 有没有就绪，跑一行代码就知道：

```python
import torch
print(torch.cuda.is_available())   # 返回 True 表示 GPU 可用
```

## 几个必会的小操作

下面这两条命令在比赛里几乎一定会用到。第一条用来临时安装某个 Python 包，注意行首要加一个 `!`，表示这是在执行 shell 命令：

```python
!pip install some-package
```

第二条用来把你的 Google Drive 挂载进来，好读取存在云盘里的数据或模型：

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 用 Markdown 加 LaTeX 写数学解答

这是 Round 1 非编码题的核心技能。在文本单元里，你可以像写普通文字一样写说明，并用美元符号嵌入数学公式——和本网站完全一样的写法：行内公式用一对 `$`，独立成行的公式用一对 `$$`。

下面是一个文本单元的内容示例：

```markdown
## 解答
设损失为 $L(w) = \frac{1}{2}(wx - y)^2$，对 $w$ 求导得：

$$\frac{\partial L}{\partial w} = (wx - y)\,x$$

令导数为 0，解得 $w^* = y / x$。
```

渲染出来，公式会变成清晰的数学排版。评分时，阅卷者既看你代码跑出的指标，也看你这段文字和公式讲得清不清楚——所以**千万别只写代码、不写说明**。

## 一份答题 notebook 长什么样

把上面这些拼起来，一道典型的 Round 1 题，作答结构大致是这样：

1. **先用一个文本单元**：用 Markdown 重述题意、写出关键的公式推导（这是非编码的得分点）。
2. **再用代码单元**：实现算法、训练模型、把结果打印出来。
3. **最后再用一个文本单元**：用一两句话解释输出说明了什么、得出什么结论。

养成"代码 + 文字"配套的习惯，你的答卷会比只堆代码的人更容易拿分。

## 容易踩的坑

- **运行时一断，变量全丢**：闲置太久（大约 90 分钟没操作）或运行超过时限（约 12 小时）后，Colab 会断开连接，**内存里的所有变量都会清空**。对策是勤保存，把训练好的模型 checkpoint 存到挂载的 Drive 里。
- **单元乱序执行会出乱子**：Colab 允许你跳着运行单元，但这样状态容易混乱。**交卷前，务必从头 `Restart and run all` 跑一遍**，确认整份 notebook 能干净地复现结果。
- **GPU 有使用配额**：免费版 GPU 用多了会被限流。Round 1 本来就只给 CPU，别养成依赖 GPU 的习惯。
- **装完包可能要重启**：少数包用 `!pip install` 装完后，要重启运行时才能正常导入。

## 它在后面会怎么用到

整个 Round 1 和 Round 2 都在 Colab 里作答（回顾 [1.1 赛事与赛程介绍](1-1-competitions.md)）。而且本网站里所有的代码，你都可以直接复制进 Colab 跑起来练手——边学边练，效果最好。

## 练习

??? note "基础练习"
    1. 新建一个 Colab 文件，用一个代码单元打印 `Hello`，再用一个文本单元写一段带行内公式 $E = mc^2$ 的说明。
    2. 把运行时切换到 GPU，运行 `torch.cuda.is_available()`，确认它返回 `True`。

??? note "进阶练习"
    1. 用 Markdown 加 LaTeX，完整写出"对 $L(w) = \tfrac{1}{2}(wx - y)^2$ 求最优 $w$"的推导过程。
    2. 把一个训练好的模型权重用 `torch.save` 存到挂载的 Drive，再在一个新会话里重新加载，模拟一次"断线恢复"。

## 小结

- USAAIO 在 **Colab** 里考试，答卷由代码单元和文本单元交替组成。
- 用 **Markdown 加 LaTeX** 写数学解答，是 Round 1 明确的得分点。
- 最要当心**运行时被重置**：勤存盘，交卷前 `Restart and run all` 验证结果可复现。

延伸：[Colab 官方欢迎页](https://colab.research.google.com/) 和一份 [Markdown 速查表](https://www.markdownguide.org/cheat-sheet/)。
