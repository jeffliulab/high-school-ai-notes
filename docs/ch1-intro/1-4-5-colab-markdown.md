# 1.4.5 Colab 与 Markdown

> **难度** ⭐☆☆☆☆ · **前置**：[1.4.1 Python 基础](1-4-1-python.md)

USAAIO 的考试就在 **Google Colab** 里进行：你在浏览器里写代码、跑模型、交答案。而且 Round 1 有"非编码题"——要求你在 **Markdown 单元里用 LaTeX 写数学推导**。所以 Colab + Markdown 不是周边技能，而是**考场基本功**。

!!! abstract "学习目标"
    - 熟悉 Colab 的单元、运行时（CPU/GPU）与常用操作
    - 会用 Markdown + LaTeX 在文本单元里写清楚数学解答
    - 知道 Colab 的会话限制，避免考场上丢进度

## 一、直觉：它解决什么问题

Colab = 云端的 Jupyter Notebook，免装环境、免费 GPU、打开即用。考试给你一个 Colab 文件，里面**代码单元**（写 Python）和**文本单元**（写说明/数学）交替——你既要会跑代码，也要会用文字把思路讲清楚。

## 二、核心用法速览

**两种单元**

- **代码单元**：写 Python，`Shift+Enter` 运行。
- **文本单元**：写 Markdown，渲染成排版文字（标题、列表、公式、表格）。

**运行时与算力**

```text
菜单 → 代码执行程序(Runtime) → 更改运行时类型 → 选 CPU / T4 GPU
```

- Round 1 仅 CPU；Round 2 可用 GPU。
- 验证 GPU 是否就绪：

```python
import torch
print(torch.cuda.is_available())   # True 表示 GPU 可用
```

**其它常用**

```python
!pip install some-package          # 行首 ! 执行 shell 命令
from google.colab import drive
drive.mount('/content/drive')      # 挂载 Google Drive 取数据
```

**Markdown + LaTeX**（写数学解答的关键）

```markdown
## 解答
设损失 $L(w)=\frac{1}{2}(wx-y)^2$，对 $w$ 求导：

$$\frac{\partial L}{\partial w}=(wx-y)\,x$$

令导数为 0 得 $w^\* = y/x$。
```

行内公式用 `$...$`，独立公式用 `$$...$$`——和本站完全一致。

## 三、实战：一份答题 notebook 的样子

一道典型 Round 1 题的作答结构：

1. **文本单元**：用 Markdown 重述思路、写出关键公式推导（非编码得分点）。
2. **代码单元**：实现算法、训练模型、打印结果。
3. **文本单元**：用一两句话解释输出、给出结论。

> 评分既看代码跑出的指标，也看你文字 + 公式表达是否清晰——别只写代码不写说明。

## 四、常见陷阱与调试

- **运行时重置丢变量**：闲置过久（约 90 分钟无操作）或超时（约 12 小时上限）会断开，**所有变量清空**。要点：及时保存、把训练 checkpoint 存到 Drive。
- **单元乱序执行**：Colab 允许跳着运行单元，状态依赖容易混乱。**交卷前从头 `Restart & Run all` 跑一遍**确认可复现。
- **GPU 配额**：免费版 GPU 有使用额度，频繁占用会被限流；Round 1 本就只给 CPU，别依赖 GPU。
- **`!pip install` 后需重启**：少数包装完要重启运行时才生效。

## 五、应用场景

- Round 1 / Round 2 全程在 Colab 作答（见 [1.1 赛事与赛程介绍](1-1-competitions.md)）。
- 本站所有代码都可直接贴进 Colab 运行练习。

## 六、练习题

??? note "基础练习"
    1. 新建一个 Colab，分别用代码单元打印 `Hello`、用文本单元写一段带行内公式 $E=mc^2$ 的说明。
    2. 切换运行时到 GPU，验证 `torch.cuda.is_available()` 为 `True`。

??? note "进阶练习"
    1. 用 Markdown + LaTeX 完整写出"对 $L(w)=\tfrac12(wx-y)^2$ 求最优 $w$"的推导过程。
    2. 把一个训练好的模型权重 `torch.save` 到挂载的 Drive，再在新会话里加载，模拟"断线恢复"。

## 七、小结与延伸阅读

- USAAIO 在 **Colab** 里考，代码单元 + 文本单元交替作答。
- Round 1 用 **Markdown + LaTeX** 写数学解答是明确得分点。
- 警惕**运行时重置**：勤存盘、交卷前 `Restart & Run all` 验证可复现。

延伸：[Colab 官方欢迎页](https://colab.research.google.com/)；[Markdown 速查表](https://www.markdownguide.org/cheat-sheet/)。
