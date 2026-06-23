# 1.10 进阶特征工程

> **难度** ⭐⭐⭐☆☆ · **前置**：[4.3 Pandas 与数据处理](../ch1-intro/1-4-3-pandas-data.md)

!!! abstract "读完这一节，你会"
    - 说清楚特征缩放为什么重要，并会手写和用库做标准化与归一化两种缩放
    - 把文字类别变成模型能吃的数字，分清独热编码和序数编码各自适合什么场景
    - 用合理的策略补缺失值，并知道为什么"补法"也要从训练集学
    - 会造交互特征和多项式特征，给线性模型一点"非线性"的本事

## 为什么洗干净还不够，还要"造"特征

在 [4.3 Pandas 与数据处理](../ch1-intro/1-4-3-pandas-data.md) 里，我们已经学会了把一份脏数据洗干净：补上缺失、缩放数值、把文字变独热。那一节做的是"基础清洗"，目标是让数据**能喂进模型**。

可是"能喂进去"和"喂进去效果好"是两回事。同一份数据，换一种表示方式，模型的表现可能差出一大截。打个比方：你要判断一套房子贵不贵，原始数据里只有"长"和"宽"两列，模型得自己费劲地揣摩它俩的关系；可如果你顺手帮它算好一个"面积 = 长 × 宽"，模型一眼就看明白了。**特征工程（feature engineering）做的就是这件事——用你对问题的理解，把数据重新表示成模型更容易学的样子。**

在 Round 1 的表格题里，特征工程往往是性价比最高的一步：常常不用换更复杂的模型，只要把特征做对，分数就上去了。这一节我们就把四件最常用的事讲透：**缩放、编码、补缺失、造交互**。

## 缩放：把量级悬殊的特征拉到同一条起跑线

先说为什么要缩放。假设你的数据里，"年龄"在 0 到 100 之间，"年收入"却在几万到几十万之间。这两列数值量级差了上千倍。很多模型（比如逻辑回归、SVM、KNN、神经网络）在算距离或者做梯度下降时，会被数值大的那一列**主导**——年收入稍微动一点，就盖过了年龄的全部影响，于是年龄这个特征基本被无视了。结果就是模型怎么调都收敛不动，或者学得很偏。

解决办法是把每个特征都缩放到差不多的范围。最常用的是两种。

**第一种叫标准化（standardization）**，也就是把每一列变成"零均值、单位方差"。公式很简单，对某个特征 $x$，减去它的均值 $\mu$，再除以它的标准差 $\sigma$：

$$x' = \frac{x - \mu}{\sigma}.$$

这么做之后，这一列的平均值变成 0，波动幅度（标准差）变成 1。它不改变数据的分布形状，只是平移加缩放，所以对有异常值的数据也比较稳。

**第二种叫归一化（min-max normalization）**，是把每一列线性压缩到 $[0,1]$ 区间。拿这一列的最小值 $x_{\min}$ 和最大值 $x_{\max}$，做：

$$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}.$$

它的好处是范围固定、好解释；缺点是只要有一个极端的异常值，最大最小值就被它拉偏，其余数据全被挤到一小块里。所以**有异常值时优先用标准化，需要固定范围（比如图像像素、某些神经网络）时用归一化**。

下面这张图把两种缩放的效果画在一起，你能直观看到它们各自把数据搬到了哪里。

<div class="diagram">
<svg viewBox="0 0 420 200" font-family="-apple-system, 'Segoe UI', sans-serif">
  <!-- 原始数据：量级很大，偏右 -->
  <text x="10" y="35" font-size="12" fill="var(--dia-stroke-soft)">原始</text>
  <line x1="70" y1="40" x2="410" y2="40" stroke="var(--dia-stroke-tertiary)" stroke-width="1"/>
  <circle cx="300" cy="40" r="4" fill="var(--dia-stroke-soft)"/>
  <circle cx="330" cy="40" r="4" fill="var(--dia-stroke-soft)"/>
  <circle cx="360" cy="40" r="4" fill="var(--dia-stroke-soft)"/>
  <circle cx="345" cy="40" r="4" fill="var(--dia-stroke-soft)"/>
  <text x="290" y="62" font-size="11" fill="var(--dia-stroke-tertiary)">量级大、挤在一起</text>

  <!-- 标准化：以0为中心 -->
  <text x="10" y="110" font-size="12" fill="var(--dia-accent)">标准化</text>
  <line x1="70" y1="115" x2="410" y2="115" stroke="var(--dia-stroke-tertiary)" stroke-width="1"/>
  <line x1="240" y1="108" x2="240" y2="122" stroke="var(--dia-stroke-tertiary)" stroke-width="1"/>
  <text x="234" y="138" font-size="10" fill="var(--dia-stroke-tertiary)">0</text>
  <circle cx="200" cy="115" r="4" fill="var(--dia-accent)"/>
  <circle cx="230" cy="115" r="4" fill="var(--dia-accent)"/>
  <circle cx="255" cy="115" r="4" fill="var(--dia-accent)"/>
  <circle cx="280" cy="115" r="4" fill="var(--dia-accent)"/>

  <!-- 归一化：压到0~1 -->
  <text x="10" y="180" font-size="12" fill="var(--dia-blue)">归一化</text>
  <line x1="70" y1="185" x2="410" y2="185" stroke="var(--dia-stroke-tertiary)" stroke-width="1"/>
  <line x1="120" y1="178" x2="120" y2="192" stroke="var(--dia-stroke-tertiary)" stroke-width="1"/>
  <line x1="360" y1="178" x2="360" y2="192" stroke="var(--dia-stroke-tertiary)" stroke-width="1"/>
  <text x="116" y="170" font-size="10" fill="var(--dia-stroke-tertiary)">0</text>
  <text x="356" y="170" font-size="10" fill="var(--dia-stroke-tertiary)">1</text>
  <circle cx="120" cy="185" r="4" fill="var(--dia-blue)"/>
  <circle cx="200" cy="185" r="4" fill="var(--dia-blue)"/>
  <circle cx="290" cy="185" r="4" fill="var(--dia-blue)"/>
  <circle cx="360" cy="185" r="4" fill="var(--dia-blue)"/>
</svg>
</div>
<p class="figure-caption">图 1.10-1：同一组数据，标准化把它搬到以 0 为中心、波动为 1；归一化把它压进 0 到 1 之间。</p>

道理讲完了，我们动手实现。先看"从零"版本——其实就是把上面两个公式翻译成 NumPy 一行：

```python
import numpy as np

x = np.array([10., 20., 30., 40.])     # 一列原始特征

# 标准化：减均值、除标准差
x_std = (x - x.mean()) / x.std()
print(x_std)        # 均值≈0，标准差≈1

# 归一化：压到 [0, 1]
x_norm = (x - x.min()) / (x.max() - x.min())
print(x_norm)       # [0. 0.333 0.667 1.]
```

实际项目里我们更常用 scikit-learn 的现成工具，因为它会帮你**记住训练集的统计量**，方便之后原样套用到测试集上：

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = StandardScaler()          # 想要归一化就换成 MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)   # 在训练集上"学"均值方差，并变换
X_test_scaled  = scaler.transform(X_test)        # 测试集只用 transform，不能再 fit！
```

这里有个**至关重要**的细节：测试集只能 `transform`，绝不能 `fit_transform`。`fit` 会去看测试集的均值方差，那等于偷看了答案——这就是常说的数据泄漏（data leakage），我们在 [1.1 监督学习工作流](2-1-1-supervised-workflow.md) 里专门讲过它的危害。

## 类别编码：把"文字"翻译成模型听得懂的数字

模型只认数字，不认"北京""上海"这样的文字，所以类别特征必须先编码。怎么编，要看这个类别**有没有顺序**。

**如果类别之间没有大小关系**，比如城市、颜色、性别，就用**独热编码（one-hot encoding）**。它把一个类别列拆成若干个 0/1 列，每一列对应一个取值，是哪类就在哪列标 1。这样做的关键好处是：模型不会误以为"上海 = 2 比北京 = 1 大"，因为它们只是几个互不相干的开关。

```python
import pandas as pd

df = pd.DataFrame({"city": ["北京", "上海", "北京", "广州"]})
print(pd.get_dummies(df, columns=["city"]))
# 得到 city_北京 / city_上海 / city_广州 三列 0/1
```

**如果类别本身有天然顺序**，比如学历"小学 < 初中 < 高中 < 大学"、衣服尺码"S < M < L"，那就用**序数编码（ordinal encoding）**，直接按顺序映射成 0、1、2、3。因为这种顺序是真实存在的，让模型知道"大学 > 高中"反而是有用的信息：

```python
order = {"小学": 0, "初中": 1, "高中": 2, "大学": 3}
df = pd.DataFrame({"edu": ["高中", "大学", "小学"]})
df["edu_code"] = df["edu"].map(order)   # 高中→2, 大学→3, 小学→0
```

那么问题来了——无序类别能不能图省事直接用序数编码（北京 = 0、上海 = 1、广州 = 2）？**最好别。** 这样会凭空给类别造出一个"上海在北京和广州中间"的假顺序，线性模型、KNN 这类靠数值大小做判断的模型会被它误导。记一句话就够：**有序用序数，无序用独热。**

不过独热也有个坑：如果某一列有几百上千种取值（比如邮编、商品 ID），独热会一下子炸出成百上千列，又稀疏又占内存。遇到这种"高基数"类别，通常改用别的办法（比如后面会学的嵌入表示，见 [1.2 嵌入表示](../ch3-round2/3-1-2-embeddings.md)），这里先知道有这个问题即可。

## 缺失值：补，但要补得讲究

[4.3 Pandas 与数据处理](../ch1-intro/1-4-3-pandas-data.md) 里我们已经会用中位数、众数来补缺失了。这里补充两点"进阶"的讲究。

**第一，补缺失的统计量，也只能从训练集算。** 这和缩放是一个道理。如果你用全部数据（含测试集）算出的中位数去补缺失，又一次泄漏了测试集的信息。正确做法是把"补法"也当成一个要从训练集学习的步骤：

```python
from sklearn.impute import SimpleImputer

imp = SimpleImputer(strategy="median")     # 数值列用中位数；类别列改成 "most_frequent"
X_train = imp.fit_transform(X_train)       # 在训练集上学"该补成多少"
X_test  = imp.transform(X_test)            # 测试集套用同一个值
```

**第二，"缺失"本身有时就是信息。** 比如一份问卷里"月收入"没填，可能恰恰和这个人不愿透露有关，并不是随机缺的。这种情况下，除了补值，我们还可以**额外加一列标记"这里原来是不是缺的"**，让模型自己去判断这件事有没有用：

```python
df["income_missing"] = df["income"].isna().astype(int)   # 缺失=1，否则=0
df["income"] = df["income"].fillna(df["income"].median())
```

多出来的这一列 0/1，常常能给模型带来意想不到的提升——这也是一个典型的"造特征"思路。

## 交互与多项式特征：给线性模型一点非线性

最后讲最能体现"造特征"威力的一招。线性模型（比如线性回归、逻辑回归）只会画直线、平面，碰到弯曲的关系就力不从心。但我们可以**手动把特征相乘、求平方，喂给它一些非线性的'原料'**，它就能间接学出曲线来。

这正是开头那个例子：有了"长"和"宽"，再补一个**交互特征（interaction feature）**"长 × 宽 = 面积"，模型立刻能用上这个组合信息。更一般地，把所有特征的乘积、平方都造出来，就叫**多项式特征（polynomial features）**。

我们先看从零怎么造——本质就是把列两两相乘、各自平方再拼回去：

```python
import numpy as np

X = np.array([[2., 3.],
              [4., 1.]])          # 两行样本，特征是 [a, b]
a, b = X[:, 0], X[:, 1]
# 二阶多项式特征：原始的 a, b，加上 a², a·b, b²
X_poly = np.column_stack([a, b, a**2, a*b, b**2])
print(X_poly)
```

库版用 sklearn 的 `PolynomialFeatures`，一行就能生成到指定阶数，还能自动加上偏置项：

```python
from sklearn.preprocessing import PolynomialFeatures

pf = PolynomialFeatures(degree=2, include_bias=False)
X_poly = pf.fit_transform(X)          # 自动造出 a, b, a², a·b, b²
print(pf.get_feature_names_out())     # 看看每一列到底是什么
```

不过要提醒一句：阶数越高、特征越多，造出来的列数会爆炸式增长，模型也越容易**过拟合**——把训练集背得滚瓜烂熟，一到新数据就翻车。所以多项式特征是把双刃剑，一般 `degree=2` 就够用，造完之后务必配上正则化和交叉验证来把关（见 [1.9 模型评估与选择](2-1-9-model-evaluation.md)）。

## 容易踩的坑

- **缩放和补缺失的统计量从全量数据算**：这是新手最常犯、又最隐蔽的错误。均值、方差、中位数都只能 `fit` 训练集，测试集只 `transform`。一旦泄漏，线下分数虚高、线上一塌糊涂。
- **无序类别硬塞序数编码**：把"北京/上海/广州"编成 0/1/2，会凭空造出假顺序误导模型。无序一律用独热。
- **独热遇到高基数类别**：几百种取值的列做独热会炸出几百列，又稀疏又慢。这种列要换思路（目标编码、嵌入等），别硬上。
- **多项式特征开太高阶**：`degree` 一大，列数指数级膨胀，几乎必然过拟合。先从 2 阶起步，再看验证集脸色。
- **编码/缩放顺序乱套**：正确顺序通常是"先补缺失 → 再编码 → 最后缩放"，且整条流程最好用 `Pipeline` 串起来，避免在测试集上漏掉某一步。

## 它在后面会怎么用到

这一节是 [4.3 Pandas 与数据处理](../ch1-intro/1-4-3-pandas-data.md) 的直接延续：那里教"基础清洗"，这里教"高级造特征"，合起来就是表格类任务从原始数据到能打的特征的完整链条。

- 缩放对**距离和梯度敏感的模型**几乎是必做项，比如 [1.7 支持向量机 SVM](2-1-7-svm.md)、[1.5 KNN 与决策树](2-1-5-knn-decision-trees.md) 里的 KNN，以及所有神经网络。
- 造完特征后，用 [1.9 模型评估与选择](2-1-9-model-evaluation.md) 的交叉验证来判断"这个新特征到底有没有用"，避免自我感动。
- 把缩放、编码、造特征全部串进一条 `Pipeline`，是工程上避免数据泄漏的标准做法，在 [1.1 监督学习工作流](2-1-1-supervised-workflow.md) 里会再强调。

## 练习

??? note "基础练习"
    1. 给定一列特征 `[5, 10, 15, 20, 100]`，分别用标准化和归一化处理它，观察那个异常值 `100` 在两种方法下分别落在哪里，体会为什么有异常值时偏爱标准化。
    2. 造一份带"尺码"列（S/M/L/XL）和"颜色"列（红/绿/蓝）的小表，给尺码用序数编码、给颜色用独热编码，并说说为什么这样分配。

??? note "进阶练习"
    1. 用 sklearn 的 `Pipeline` 把"中位数补缺失 → 标准化 → 二阶多项式特征 → 线性回归"串成一条流水线，再用交叉验证证明：只在训练集上 `fit` 时，分数比起在全量数据上 `fit` 更接近真实的测试表现。
    2. 找一个有"长、宽"两列的数据集，对比"只用原始特征"和"额外加上长×宽交互特征"两种情况下线性回归的误差，亲眼看看一个好交互特征能带来多大提升。

## 小结

- 特征工程的一句话总结：**用你对问题的理解，把数据重新表示成模型更好学的样子**——常常比换模型更划算。
- 四件常做的事：缩放（标准化/归一化）、类别编码（无序用独热、有序用序数）、补缺失（还可加"是否缺失"标记）、造交互与多项式特征。
- 一条铁律贯穿始终：**所有统计量只从训练集学，测试集只套用**，否则就是数据泄漏。
- 多项式特征是双刃剑，阶数别贪高，造完一定用交叉验证把关。

想系统了解更多技巧，可读 scikit-learn 官方的 [Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html) 一章，里面把各种缩放器和编码器讲得很全。
