# 1.4.3 Pandas 与数据处理

> **难度** ⭐⭐☆☆☆ · **前置**：[1.4.2 NumPy 与张量](1-4-2-numpy-tensors.md)

竞赛数据往往是一张张 CSV 表格。Pandas 就是 Python 里的"Excel++"：读表、选列、补缺失、变换特征，全都几行搞定。机器学习的第一步永远是把数据**洗干净、整理成模型能吃的形状**，这一节就讲这件事。

!!! abstract "学习目标"
    - 会用 `DataFrame` 读取、选择、过滤数据
    - 能检测并处理缺失值，理解归一化/标准化/独热编码
    - 避开 `SettingWithCopyWarning` 等常见数据处理坑

## 一、直觉：它解决什么问题

原始数据有缺失、有文字类别、量纲不一。模型只认数字、怕缺失、对尺度敏感。Pandas 负责把"脏表格"变成"干净的数值矩阵"，再交给 NumPy/PyTorch。

## 二、核心用法速览

```python
import pandas as pd

df = pd.read_csv("data.csv")     # 读 CSV
df.head()                        # 看前几行
df.info(); df.describe()         # 类型概览 / 数值统计

# 选择与过滤
df["age"]                        # 取一列（Series）
df[["age", "score"]]             # 取多列
df.loc[df["age"] > 18]           # 按条件过滤行
df.iloc[0:5]                     # 按位置取前 5 行

# 缺失值
df.isna().sum()                  # 每列缺失个数
df["age"].fillna(df["age"].median(), inplace=True)   # 中位数填补
df.dropna(subset=["label"])      # 标签缺失则丢弃该行

# 分组统计
df.groupby("class")["score"].mean()
```

## 三、实战：一条龙预处理

```python
import pandas as pd

df = pd.read_csv("data.csv")

# 1) 补缺失：数值用中位数，类别用众数
df["age"] = df["age"].fillna(df["age"].median())
df["city"] = df["city"].fillna(df["city"].mode()[0])

# 2) 标准化数值列（零均值、单位方差）——尺度敏感模型必做
num = ["age", "income"]
df[num] = (df[num] - df[num].mean()) / df[num].std()

# 3) 类别变量 -> 独热编码（文字变 0/1 列）
df = pd.get_dummies(df, columns=["city"])

X = df.drop(columns=["label"]).to_numpy()   # 交给模型的数值矩阵
y = df["label"].to_numpy()
```

## 四、常见陷阱与调试

- **`SettingWithCopyWarning`**：`df[df.a>0]["b"] = 1` 这种**链式赋值**可能改不到原表。用 `df.loc[df.a>0, "b"] = 1` 一步到位。
- **缺失值不能直接喂模型**：训练前务必 `isna().sum()` 检查，否则 sklearn/torch 会报错或得 NaN。
- **泄漏（leakage）**：标准化的均值/方差只能用**训练集**统计，再套到验证/测试集，别用全量数据算（详见 [2.1.1 监督学习工作流](../ch2-round1/2-1-1-supervised-workflow.md)）。
- **`inplace` 与链式**：`inplace=True` 原地改、返回 `None`，别再赋值给变量。

## 五、应用场景

- 表格类任务（Round 1 常见）的标准入口：读数据 → 清洗 → 特征 → 建模。
- 与 [2.1.10 进阶特征工程](../ch2-round1/2-1-10-feature-engineering.md) 衔接：这里做基础清洗，那里做高级造特征。

## 六、练习题

??? note "基础练习"
    1. 读入一份带缺失的 CSV，打印每列缺失比例，并对数值列用均值填补。
    2. 把某个类别列用 `get_dummies` 独热化，观察列数变化。

??? note "进阶练习"
    1. 实现"只用训练集统计量做标准化"的流程：先 `train_test_split`，再 fit 在训练集、transform 到测试集。
    2. 用 `groupby` + `agg` 统计每个类别的均值、最大值、计数，找出异常类别。

## 七、小结与延伸阅读

- Pandas = 表格数据处理利器：读、选、过滤、补缺失、造特征。
- 模型怕缺失、对尺度敏感 —— **填补 + 标准化 + 独热**是三件基本功。
- 统计量只用训练集算，防数据泄漏。

延伸：[Pandas 官方 10 分钟入门](https://pandas.pydata.org/docs/user_guide/10min.html)；Kaggle《Pandas》micro-course。
