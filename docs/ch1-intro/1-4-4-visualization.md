# 1.4.4 数据可视化

> **难度** ⭐☆☆☆☆ · **前置**：[1.4.3 Pandas 与数据处理](1-4-3-pandas-data.md)

"先画图，再建模。"动手训练之前，用几张图看清数据的分布、关系和离群点（这一步叫 **EDA，探索性数据分析**），能帮你选对模型、少走弯路。Matplotlib 是底层画笔，Seaborn 在它之上让统计图更好看更省事。

!!! abstract "学习目标"
    - 会用 Matplotlib 画折线、散点、直方图
    - 会用 Seaborn 画箱线图、热力图做数据探查
    - 能从图里读出分布形状、离群点和特征相关性

## 一、直觉：它解决什么问题

一列数字 `describe()` 只给你均值方差；但一张直方图能立刻告诉你"它是钟形还是双峰、有没有极端离群点"。图把人脑最擅长的视觉模式识别用上了。

## 二、核心用法速览

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

x = np.linspace(0, 10, 100)

# Matplotlib：折线 / 散点 / 直方图
plt.plot(x, np.sin(x))                 # 折线
plt.scatter(x, np.sin(x) + 0.1*np.random.randn(100))  # 散点
plt.hist(np.random.randn(1000), bins=30)              # 直方图
plt.xlabel("x"); plt.ylabel("y"); plt.title("demo")
plt.show()
```

```python
# Seaborn：统计图更省事
data = np.random.randn(200)
sns.histplot(data, kde=True)           # 直方 + 密度曲线
sns.boxplot(x=data)                    # 箱线图：看中位数与离群点
# 相关性热力图（看哪些特征相关）
import pandas as pd
df = pd.DataFrame(np.random.randn(100, 4), columns=list("ABCD"))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()
```

| 图类型 | 看什么 |
|---|---|
| 直方图 `hist` | 单变量分布形状（单峰/双峰/偏态） |
| 箱线图 `boxplot` | 中位数、四分位、**离群点** |
| 散点图 `scatter` | 两变量关系/趋势 |
| 热力图 `heatmap` | 特征间相关性矩阵 |

## 三、实战：分布 + 离群点一眼看穿

```python
import seaborn as sns, matplotlib.pyplot as plt, numpy as np
data = np.concatenate([np.random.randn(300), [8, 9, 10]])  # 故意混入离群点

fig, ax = plt.subplots(1, 2, figsize=(9, 3))
sns.histplot(data, kde=True, ax=ax[0]); ax[0].set_title("分布")
sns.boxplot(x=data, ax=ax[1]); ax[1].set_title("离群点")
plt.tight_layout(); plt.show()
```

箱线图右侧那几个孤立点，就是混入的 8/9/10——这正是要在预处理里关注的离群值。

## 四、常见陷阱与调试

- **忘了 `plt.show()`**：脚本里不调用就不显示（Colab/Notebook 里通常自动显示，可省）。
- **中文显示成方块**：需设中文字体，如 `plt.rcParams["font.sans-serif"] = ["SimHei"]`，并 `axes.unicode_minus=False`。
- **图叠在一起**：连续画多张要 `plt.figure()` 新建，或用 `subplots` 分轴。
- **过度装饰**：competition EDA 重信息不重美观，别在配色上耗时间。

## 五、应用场景

- 训练时画 **loss/accuracy 曲线**判断过拟合（见 [2.1.9 模型评估与选择](../ch2-round1/2-1-9-model-evaluation.md)）。
- 画**混淆矩阵热力图**分析分类错误（见 [2.1.4 分类评估指标](../ch2-round1/2-1-4-classification-metrics.md)）。
- 建模前看特征分布与相关性，决定要不要标准化/去相关。

## 六、练习题

??? note "基础练习"
    1. 对 [1.3.3 概率与统计](1-3-3-probability-statistics.md) 里采样的高斯数据画直方图 + KDE，验证钟形。
    2. 用散点图画 $y=x^2+\text{噪声}$，直观感受非线性关系。

??? note "进阶练习"
    1. 用 `sns.pairplot` 画鸢尾花数据集各特征两两关系，按类别上色。
    2. 画一条"学习率 vs 最终损失"的曲线，找出最优学习率区间。

## 七、小结与延伸阅读

- **先 EDA 再建模**：直方图看分布、箱线看离群、热力图看相关。
- Matplotlib 是底层，Seaborn 让统计图更省事更好看。
- 训练曲线、混淆矩阵是日后调试的常用图。

延伸：[Matplotlib 官方教程](https://matplotlib.org/stable/tutorials/index.html)；[Seaborn 官方画廊](https://seaborn.pydata.org/examples/index.html)。
