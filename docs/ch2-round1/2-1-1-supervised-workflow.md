# 1.1 监督学习工作流

> **难度** ⭐⭐☆☆☆ · **前置**：[4.3 Pandas 与数据处理](../ch1-intro/1-4-3-pandas-data.md)

!!! abstract "读完这一节，你会"
    - 说清"监督学习"到底在学什么，以及一条完整流程长什么样：数据 → 划分 → 训练 → 评估
    - 用一句话区分过拟合和欠拟合，并知道在曲线上各对应什么样子
    - 看懂什么叫数据泄漏，知道它为什么能让你的成绩"虚高"再崩盘
    - 用 scikit-learn 把上面这条流程串成一段不到 20 行、真正能跑的最小管道

## 先搞清楚：监督学习到底在学什么

在动手之前，我们得先回答一个最朴素的问题：所谓"机器学习"，机器到底在学什么？

答案其实很接地气。**监督学习（supervised learning）就是"看着标准答案学做题"。** 你给模型一堆题目和对应的正确答案，让它自己去摸索"从题目到答案"的规律；学完之后，再丢给它一道**没见过**的新题，看它能不能答对。这里的"题目"我们叫**特征（feature）**，"答案"叫**标签（label）**。

举个例子。你手上有一千套房子的资料，每套都记着面积、楼层、地段这些数字（特征），也记着它最终的成交价（标签）。监督学习要做的，就是从这一千个"面积→房价"的例子里，归纳出一个能预测房价的函数 $f$，使得对新房子也能给出一个靠谱的价格。

按标签的类型，监督学习又分成两大类，你后面会反复见到：如果标签是连续的数值（比如房价），叫**回归（regression）**；如果标签是有限的几个类别（比如"垃圾邮件 / 正常邮件"），叫**分类（classification）**。这一节讲的流程对两者通用，差别只在最后用什么尺子衡量好坏。

## 一条完整的流程，长什么样

很多初学者一上来就急着调模型、堆算法，结果在最基础的流程上栽跟头。所以我们先把整条流水线在脑子里走一遍——它其实只有四步，下面这张图先给你一个整体印象：

<div class="diagram">
<svg viewBox="0 0 560 130" font-family="-apple-system, 'Segoe UI', sans-serif">
  <defs>
    <marker id="wf" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="var(--dia-stroke-soft)"/></marker>
  </defs>
  <rect x="10" y="45" width="92" height="40" rx="5" fill="var(--dia-bg-card)" stroke="var(--dia-stroke-soft)"/>
  <text x="56" y="69" font-size="13" text-anchor="middle" fill="var(--dia-stroke)">原始数据</text>
  <line x1="104" y1="65" x2="138" y2="65" stroke="var(--dia-stroke-soft)" stroke-width="1.5" marker-end="url(#wf)"/>
  <rect x="142" y="45" width="100" height="40" rx="5" fill="var(--dia-bg-card)" stroke="var(--dia-accent)"/>
  <text x="192" y="62" font-size="12" text-anchor="middle" fill="var(--dia-accent)">划分</text>
  <text x="192" y="78" font-size="11" text-anchor="middle" fill="var(--dia-stroke-soft)">训练/验证/测试</text>
  <line x1="244" y1="65" x2="278" y2="65" stroke="var(--dia-stroke-soft)" stroke-width="1.5" marker-end="url(#wf)"/>
  <rect x="282" y="45" width="92" height="40" rx="5" fill="var(--dia-bg-card)" stroke="var(--dia-green)"/>
  <text x="328" y="69" font-size="13" text-anchor="middle" fill="var(--dia-green)">训练模型</text>
  <line x1="376" y1="65" x2="410" y2="65" stroke="var(--dia-stroke-soft)" stroke-width="1.5" marker-end="url(#wf)"/>
  <rect x="414" y="45" width="92" height="40" rx="5" fill="var(--dia-bg-card)" stroke="var(--dia-blue)"/>
  <text x="460" y="69" font-size="13" text-anchor="middle" fill="var(--dia-blue)">评估</text>
  <path d="M460 45 C 460 15, 192 15, 192 43" fill="none" stroke="var(--dia-stroke-tertiary)" stroke-width="1" stroke-dasharray="4 3" marker-end="url(#wf)"/>
  <text x="326" y="22" font-size="11" text-anchor="middle" fill="var(--dia-stroke-tertiary)">不满意就回去调</text>
</svg>
</div>
<p class="figure-caption">图 1.1-1：监督学习的标准流水线。注意那条虚线——评估不好时我们回头调整，但真正的"考试卷"（测试集）只在最后用一次。</p>

我们来逐步拆开。**第一步是准备数据**，也就是上一节 [4.3 Pandas 与数据处理](../ch1-intro/1-4-3-pandas-data.md) 做的那些事：补缺失、标准化、把类别变成数字，最后得到一个特征矩阵 $X$ 和一列标签 $y$。

**第二步是把数据切开。** 这一步最关键，也最容易被忽视。我们绝不能拿全部数据去训练，否则就没法知道模型在"没见过的题"上表现如何。所以要预留一部分数据当"考卷"。常见的做法是切成三份：

- **训练集（training set）**：让模型从这上面学规律，通常占 60%–80%。
- **验证集（validation set）**：用来挑选模型、调超参数。你可以把它理解成"模拟考"——用它来比较"是用决策树好还是用线性模型好"。
- **测试集（test set）**：最后才动用一次的"高考卷"，用来报告模型真实水平。**它在调参阶段绝对不能碰。**

**第三步是训练**。训练的本质，是让模型不断调整自己的参数，使它在训练集上的预测尽量贴近真实标签。怎么衡量"贴近"？用一个**损失函数（loss function）**。比如回归常用均方误差，预测值 $\hat{y}_i$ 和真值 $y_i$ 差得越多，惩罚越重：

$$
L = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^2.
$$

训练就是想办法把这个 $L$ 压到尽量小（具体怎么压，留给 [1.2 线性回归](2-1-2-linear-regression.md) 和梯度下降）。

**第四步是评估**，拿模型没见过的数据来打分。如果分数不理想，我们回到第二、三步去换模型、调参数——但记住图里那条规矩：来回折腾用的是验证集，测试集要留到最后。

## 过拟合与欠拟合：两种"没学好"

训练完一看分数，你会遇到两种典型的"没学好"，搞清它们是整个机器学习的核心直觉。

第一种叫**欠拟合（underfitting）**：模型太简单，连训练集都没学明白，训练误差和验证误差都很高。这就像一个学生题目都没读懂，模拟考和真题都考砸。

第二种叫**过拟合（overfitting）**，更隐蔽也更常见：模型太复杂，把训练集"背"得滚瓜烂熟，训练误差极低，可一换新题就露馅，验证误差很高。这就像一个学生死记硬背了每道例题的答案，却没掌握解题方法，换个数字就不会了。

那"刚刚好"是什么样？下面这张图把三种情况画在了一起，你顺着两条曲线看就明白了：

<div class="diagram">
<svg viewBox="0 0 380 220" font-family="-apple-system, 'Segoe UI', sans-serif">
  <line x1="45" y1="180" x2="350" y2="180" stroke="var(--dia-stroke-tertiary)" stroke-width="1"/>
  <line x1="45" y1="180" x2="45" y2="20" stroke="var(--dia-stroke-tertiary)" stroke-width="1"/>
  <text x="197" y="205" font-size="12" text-anchor="middle" fill="var(--dia-stroke-soft)">模型复杂度 →</text>
  <text x="20" y="100" font-size="12" text-anchor="middle" fill="var(--dia-stroke-soft)" transform="rotate(-90 20 100)">误差 →</text>
  <path d="M60 165 C 120 95, 200 60, 330 40" fill="none" stroke="var(--dia-green)" stroke-width="2"/>
  <text x="300" y="35" font-size="11" fill="var(--dia-green)">训练误差</text>
  <path d="M60 150 C 130 80, 175 70, 330 150" fill="none" stroke="var(--dia-accent)" stroke-width="2"/>
  <text x="300" y="165" font-size="11" fill="var(--dia-accent)">验证误差</text>
  <line x1="168" y1="20" x2="168" y2="180" stroke="var(--dia-stroke-tertiary)" stroke-width="1" stroke-dasharray="4 3"/>
  <circle cx="168" cy="72" r="3.5" fill="var(--dia-blue)"/>
  <text x="168" y="100" font-size="11" text-anchor="middle" fill="var(--dia-blue)">最佳点</text>
  <text x="95" y="50" font-size="11" text-anchor="middle" fill="var(--dia-stroke-soft)">欠拟合</text>
  <text x="285" y="110" font-size="11" text-anchor="middle" fill="var(--dia-stroke-soft)">过拟合</text>
</svg>
</div>
<p class="figure-caption">图 1.1-2：随着模型变复杂，训练误差一路下降；但验证误差先降后升。我们要找的，是验证误差最低的那个"最佳点"。</p>

记住这条曲线，胜过记一堆定义：**训练误差总是越来越低，但验证误差先降后升——它抬头的地方，就是模型开始"背答案"的地方。** 我们调模型的全部目的，就是停在那个谷底。怎么对付过拟合（正则化、早停、加数据），后面 [2.5 优化与正则化](2-2-5-optimization-regularization.md) 会专门讲。

## 数据泄漏：最坑人、也最隐蔽的错误

讲完过拟合，必须紧接着讲一个新手几乎都会踩、而且踩了还不自知的坑——**数据泄漏（data leakage）**。

它的意思是：测试集的信息，不该出现却悄悄"漏"进了训练过程，导致模型在评估时分数虚高，可一上线就垮。打个比方，这就好比考试前你不小心瞄到了答案，模拟分自然漂亮，但那分数是假的。

最常见的一种泄漏，恰恰发生在我们上一节学过的标准化上。回想一下，标准化要用到均值和方差。如果你**先**对全部数据算均值方差再切分，那训练集就间接"知道"了测试集的统计信息——答案漏过去了。正确的顺序永远是：**先切分，再只用训练集算统计量，然后把同一套统计量套到验证集和测试集上。**

这个先后顺序很容易写反，所以 sklearn 专门用 `Pipeline` 把它兜住，我们下面就会看到。先记住这句口诀：**任何"从数据里学来的东西"——均值、方差、词表、编码映射——都只能从训练集学，绝不能让测试集参与。**

## 用 sklearn 串一条最小管道

道理讲完了，我们动手把整条流程跑通一遍。下面这段代码会做四件事：造一份数据、切分、训练、评估，正好对应图 1.1-1 的四步。我们用 sklearn 自带的鸢尾花数据集（一个经典的三分类小数据），先看不带 Pipeline 的"朴素版"，把每一步看清楚：

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 第一步：拿到特征 X 和标签 y
X, y = load_iris(return_X_y=True)

# 第二步：切分。先分出测试集（20%），它从此刻起就被"封存"
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
# stratify=y 让每个类别在训练/测试里的比例保持一致，分类任务建议都加上

# 关键：标准化的均值方差，只用训练集来 fit
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)   # fit + transform：从训练集学统计量
X_test_s  = scaler.transform(X_test)        # 只 transform：套用同一套统计量，不偷看

# 第三步：训练
model = LogisticRegression(max_iter=200)
model.fit(X_train_s, y_train)

# 第四步：评估——拿封存的测试集打分
pred = model.predict(X_test_s)
print("测试集准确率：", accuracy_score(y_test, pred))   # 约 0.97
```

跑完你会看到准确率在 0.97 上下。请特别留意 `fit_transform` 和 `transform` 的区别：训练集用 `fit_transform`（先学统计量再变换），测试集只用 `transform`（套用已学到的统计量）。**这一行之差，就是防住数据泄漏的命门。**

不过手动管理"先 fit 哪个、再 transform 哪个"很容易写错，尤其是步骤一多。所以 sklearn 给了 `Pipeline`，它能把"标准化 + 模型"打包成一个整体，自动保证统计量只从训练数据学。同样的事情，用管道写出来更短、也更不容易出错：

```python
from sklearn.pipeline import make_pipeline

# 把标准化和模型串成一条管道，当成一个整体来用
pipe = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=200),
)

pipe.fit(X_train, y_train)          # fit 时，标准化只从 X_train 学统计量
print("测试集准确率：", pipe.score(X_test, y_test))   # 约 0.97，和上面一致
```

注意这里我们直接把**没标准化**的 `X_train`、`X_test` 喂给管道——因为标准化已经被装进管道内部了。调用 `pipe.fit` 时，它只用训练数据去学均值方差；调用 `pipe.score` 时，它自动用同一套统计量去变换测试集。**泄漏的口子被从结构上堵死了**，这正是我们推荐永远用 Pipeline 的原因。

到此，从数据到一个能打分的模型，整条流水线就跑通了。后面学的每一个算法，无非是把管道里的 `LogisticRegression` 换成别的东西而已——流程本身，始终是这一套。

## 容易踩的坑

- **先标准化再切分**：这是最典型的数据泄漏。顺序永远是"先切分，统计量只从训练集学"。用 `Pipeline` 能从结构上避免。
- **拿测试集调参**：测试集只能在最最后用一次。如果你反复用它来挑模型、调参数，它就退化成了第二个训练集，报出来的分数不再可信——这时该用验证集或交叉验证。
- **分类不分层**：类别不均衡时，随机切分可能让某一类在测试集里几乎没有。分类任务记得加 `stratify=y`。
- **训练分高就高兴**：训练集上 99% 毫无意义，甚至可能是过拟合的信号。**永远盯着验证/测试分数**，那才是模型真实的水平。
- **忘了固定随机种子**：不设 `random_state`，每次切分结果都不同，实验没法复现。调试阶段务必固定它。

## 它在后面会怎么用到

这一节是整个 Round 1 的"总纲"，后面几乎每一节都在往这条流水线里填零件：

- **换"训练模型"那一步**：[1.2 线性回归](2-1-2-linear-regression.md) 和 [1.3 逻辑回归与分类](2-1-3-logistic-regression.md) 教你管道里第一个真正的模型怎么从零实现。
- **换"评估"那一步**：[1.4 分类评估指标](2-1-4-classification-metrics.md) 和 [1.9 模型评估与选择](2-1-9-model-evaluation.md) 会告诉你，除了准确率还有哪些尺子，以及怎么用交叉验证更稳地划分数据。
- **强化"准备数据"那一步**：[1.10 进阶特征工程](2-1-10-feature-engineering.md) 接着 [4.3 Pandas 与数据处理](../ch1-intro/1-4-3-pandas-data.md) 往下讲，造出更有信息量的特征。

换句话说，**这张流水线图你会用一整年**，把它记牢，后面学起来就是"对号入座"。

## 练习

??? note "基础练习"
    1. 用 `load_iris` 取数据，把测试集比例从 0.2 改成 0.4，重新跑一遍上面的管道，看准确率怎么变。想一想：测试集变大、训练集变小，对结果是好是坏？
    2. 故意制造一次数据泄漏：先对**全部** `X` 做 `StandardScaler().fit_transform`，再切分、训练、评估，和正确做法对比分数差多少。（在 iris 这种简单数据上差别可能很小，但你要养成识别这种写法的肌肉记忆。）

??? note "进阶练习"
    1. 不调用 `train_test_split`，自己用 NumPy 实现一个切分函数：先 `np.random.permutation` 打乱下标，再按比例切成两份。和官方函数对比结果是否一致。
    2. 画一条属于你自己的"图 1.1-2"：用一个多项式回归，让多项式的次数从 1 涨到 15，每个次数都记录训练误差和验证误差，把两条曲线画出来，亲眼看到验证误差"先降后升"的拐点。

## 小结

- 监督学习就是"看着标准答案学做题"：从特征到标签学一个函数，再用没见过的数据检验它。
- 整条流水线只有四步：**数据 → 划分（训练/验证/测试）→ 训练 → 评估**，把这张图记牢，后面全是往里填零件。
- 训练误差一路降、验证误差先降后升；我们要停在验证误差最低的"最佳点"，避开过拟合和欠拟合。
- **数据泄漏**是最隐蔽的错误，记住一句话：任何从数据里学来的统计量都只能从训练集学。用 `Pipeline` 从结构上堵住这个口子。

想系统地把这套流程练熟，强烈推荐 scikit-learn 官方的 [Getting Started 教程](https://scikit-learn.org/stable/getting_started.html)，它用的也正是这条"准备—切分—训练—评估"的思路。
