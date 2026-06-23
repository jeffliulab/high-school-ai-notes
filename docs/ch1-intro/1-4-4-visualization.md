# 1.4.4 数据可视化

> **难度** ⭐☆☆☆☆ · **前置**：[1.4.3 Pandas 与数据处理](1-4-3-pandas-data.md)

!!! abstract "读完这一节，你会"
    - 用 Matplotlib 画折线、散点、直方图
    - 用 Seaborn 画箱线图和热力图来探查数据
    - 从图里读出数据的分布形状、离群点和特征间的相关性

## 为什么动手前要先画图

有句话叫"一图胜千言"，在数据分析里尤其如此。在真正训练模型之前，我们习惯先把数据画出来看看——这一步叫**探索性数据分析（EDA，Exploratory Data Analysis）**。

为什么不直接看数字？因为光靠 `describe()` 给你的均值和方差，你很难判断数据到底长什么样。可一张直方图就能立刻告诉你：这组数据是单峰还是双峰、是对称还是偏斜、有没有几个孤零零的极端值。**人脑最擅长从图像里识别模式**，可视化做的就是把这份本事用起来。这一节我们用两个工具：Matplotlib 是底层画笔，Seaborn 在它之上让统计图更好看、更省事。

## Matplotlib：最基础的画笔

Matplotlib 几乎能画任何图，最常用的是折线、散点和直方图三种：

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)

plt.plot(x, np.sin(x))                    # 折线图：看趋势
plt.scatter(x, np.sin(x) + 0.1*np.random.randn(100))  # 散点图：看关系
plt.hist(np.random.randn(1000), bins=30)  # 直方图：看分布
plt.xlabel("x"); plt.ylabel("y"); plt.title("示例")
plt.show()
```

每画完一张图，记得加上坐标轴标签和标题——这是好习惯，能让别人（和几天后的你自己）一眼看懂图在说什么。

## Seaborn：让统计图更省事

如果说 Matplotlib 是"什么都能画但要自己调"，那 Seaborn 就是"常用统计图一行搞定"。它特别适合做 EDA：

```python
import seaborn as sns
data = np.random.randn(200)

sns.histplot(data, kde=True)    # 直方图 + 一条平滑的密度曲线
sns.boxplot(x=data)             # 箱线图：一眼看出中位数和离群点
plt.show()
```

`histplot` 加上 `kde=True` 后，会在直方图上叠一条平滑曲线，分布形状看得更清楚；`boxplot` 则专门用来揪出离群点。

到底哪种图看哪种信息？下面这张表帮你快速对号入座：

| 图类型 | 主要用来看 |
|---|---|
| 直方图 `histplot` | 单个变量的分布形状（单峰、双峰、偏斜） |
| 箱线图 `boxplot` | 中位数、四分位数，以及**离群点** |
| 散点图 `scatter` | 两个变量之间的关系或趋势 |
| 热力图 `heatmap` | 多个特征两两之间的相关性 |

## 实战：一眼看穿分布和离群点

我们把"看分布"和"找离群点"合起来做一次。先故意在一组正常数据里混进几个极端值，再用直方图和箱线图并排显示：

```python
import seaborn as sns, matplotlib.pyplot as plt, numpy as np
data = np.concatenate([np.random.randn(300), [8, 9, 10]])  # 混入 3 个离群点

fig, ax = plt.subplots(1, 2, figsize=(9, 3))
sns.histplot(data, kde=True, ax=ax[0]); ax[0].set_title("分布")
sns.boxplot(x=data, ax=ax[1]); ax[1].set_title("离群点")
plt.tight_layout(); plt.show()
```

你会发现，左边的直方图主体是个钟形，但右侧拖着几个孤立的小柱子；右边的箱线图则在右端标出几个单独的点——它们正是我们混进去的 8、9、10。这就是为什么在 [1.4.3 Pandas](1-4-3-pandas-data.md) 里我们要专门关注离群值：它们常常是录入错误，会把模型带偏。

## 容易踩的坑

- **忘了 `plt.show()`**：在普通脚本里不调用它，图就不会显示（不过在 Colab 或 Notebook 里通常会自动显示，可以省略）。
- **中文显示成方块**：matplotlib 默认不带中文字体，需要设置 `plt.rcParams["font.sans-serif"] = ["SimHei"]`，并把 `axes.unicode_minus` 设为 `False`。
- **多张图叠在一起**：连续画好几张时，要么用 `plt.figure()` 新建画布，要么用 `subplots` 分成多个子图。
- **别在配色上耗时间**：做 EDA 重在快速看清信息，不是做美工，朴素清楚就好。

## 它在后面会怎么用到

可视化会一路陪着你：训练时，你要画 **loss 和准确率曲线**来判断模型有没有过拟合（见 [2.1.9 模型评估与选择](../ch2-round1/2-1-9-model-evaluation.md)）；分析分类错误时，你要画**混淆矩阵的热力图**（见 [2.1.4 分类评估指标](../ch2-round1/2-1-4-classification-metrics.md)）；建模前，你还要看特征的分布和相关性，来决定要不要标准化、要不要去掉某些冗余特征。

## 练习

??? note "基础练习"
    1. 对 [1.3.3 概率与统计](1-3-3-probability-statistics.md) 里采样的高斯数据画一张直方图加密度曲线，亲眼验证它是钟形的。
    2. 用散点图画出 $y = x^2 + \text{噪声}$ 的数据，直观感受一下什么叫"非线性关系"。

??? note "进阶练习"
    1. 用 `sns.pairplot` 画出鸢尾花数据集里各特征两两之间的关系，并按类别上不同颜色。
    2. 画一条"学习率 vs 最终损失"的曲线，从图上找出最优学习率大致在哪个区间。

## 小结

- **先 EDA 再建模**：直方图看分布、箱线图找离群点、热力图看相关性。
- Matplotlib 是底层画笔，Seaborn 让常用统计图一行搞定。
- 训练曲线和混淆矩阵是你日后调试模型最常画的两种图。

想看更多图例，可逛 [Seaborn 官方画廊](https://seaborn.pydata.org/examples/index.html) 和 [Matplotlib 教程](https://matplotlib.org/stable/tutorials/index.html)。
