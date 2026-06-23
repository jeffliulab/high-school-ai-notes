# A2 真题与数据集

> **难度** ⭐☆☆☆☆ · **前置**：[1. 赛事与赛程介绍](../ch1-intro/1-1-competitions.md)、[4.3 Pandas 与数据处理](../ch1-intro/1-4-3-pandas-data.md)

!!! abstract "本页目标"
    - 知道 IOAI / IAIO 的官方真题去哪里找、长什么样、怎么用
    - 手里攒一份"常用练手数据集"清单，每个都清楚它是干嘛的、几行代码就能下下来
    - 学会用真题和公开数据集自己搭一套模拟训练，而不是干等老师发题

## 这页是干嘛用的

备赛到后面你一定会问一句话：**"题在哪儿，数据在哪儿？"** 光看笔记、刷概念是不够的，真正能把水平顶上去的，是拿真题练手、拿真实数据集跑通一整套流程。所以这一页不讲新知识点，它是一张"资源地图"——把官方真题的来源、以及竞赛里反复出现的那几个经典数据集，集中整理在一处，方便你随时回来查。

各个备考章节（比如 [4.1 真题形式与评分](../ch2-round1/2-4-1-exam-format.md)、[1.3 历年真题库](../ch5-champion/5-1-3-ioai-past.md)）会反复指回这一页，所以你可以把它当成一个"中转站"：在那些页面里看到"去 A2 找数据集"时，就翻到这里来。

## 官方真题去哪里找

先说最权威的来源——官方真题。这些是历届比赛真正考过的题，价值最高，建议优先刷。下面这张表帮你对号入座：

| 来源 | 是什么 | 怎么获取 |
| --- | --- | --- |
| IOAI 官方网站 | 国际人工智能奥林匹克（International Olympiad in Artificial Intelligence）历届赛题、Syllabus、样题 | 官网 results / past problems 页面，链接占位：`https://ioai-official.org`（以官方为准） |
| IAIO 官方网站 | 国际人工智能奥赛（International AI Olympiad）赛制与历年题 | 官网 archive 页面，链接占位：`<待补官方链接>` |
| USAAIO / 国家选拔赛 | 北美选拔赛历年题，难度梯度更适合入门 | 选拔赛官方页 + 公开的 GitHub 题库，链接占位：`<待补官方链接>` |
| 官方 GitHub 仓库 | 很多届会把题目、数据、baseline notebook 一起开源 | 在 GitHub 搜 `IOAI` / `IAIO` + 年份，认准官方组织账号 |

!!! warning "链接以官方为准"
    上面带"占位"的网址只是给你一个找的方向，**真正的官方域名请以当年组委会公告为准**——竞赛官网换域名是常事。具体的赛制和真题分析，分别在 [1.1 赛制详解](../ch5-champion/5-1-1-ioai-format.md) 和 [1.3 历年真题库](../ch5-champion/5-1-3-ioai-past.md) 里展开。

除了官方题，**Kaggle** 是练手的金矿。它上面有海量公开竞赛和数据集，题型、评分方式都和 AI 奥赛高度相似（给你数据、让你训模型、按指标排名）。注册一个账号，搜 "image classification" 或 "NLP" 之类的关键词，就能找到大量带标准答案、带 baseline 的练习题。

## 常用数据集清单：练手就从这几个开始

光有题还不够，你得有数据反复跑实验。下面这几个是 AI 圈里"人人都用过"的经典数据集——它们体积小、下载方便、被无数教程当例子，特别适合验证你的代码和模型能不能跑通。按"图像 / 文本 / 表格"三类整理：

| 数据集 | 类型 | 用途（练什么） | 规模 |
| --- | --- | --- | --- |
| MNIST | 图像 | 手写数字 0–9 识别，入门图像分类的"Hello World" | 7 万张 28×28 灰度图 |
| Fashion-MNIST | 图像 | 衣物分类，比 MNIST 难一点、接口完全兼容 | 7 万张 28×28 灰度图 |
| CIFAR-10 / CIFAR-100 | 图像 | 彩色小图分类，练 CNN 的标准台阶 | 6 万张 32×32 彩图，10 / 100 类 |
| ImageNet（子集） | 图像 | 大规模分类、迁移学习的预训练来源 | 上百万张，通常用子集 |
| IMDB Reviews | 文本 | 电影评论情感分类（正面/负面），入门 NLP | 5 万条带标签影评 |
| AG News | 文本 | 新闻多分类，练文本分类与词嵌入 | 12 万条新闻 |
| Iris | 表格 | 鸢尾花三分类，最经典的小型 ML 数据集 | 150 行 4 特征 |
| Titanic | 表格 | 生存预测，练特征工程与二分类 | 约 900 行 |

这张表你不用背，用到哪个查哪个就行。一个小建议：**入门期优先用 MNIST 和 CIFAR-10**，因为几乎所有教程、所有报错的 StackOverflow 答案都拿它们举例，你遇到问题时最容易搜到帮助。

## 几行代码就能把数据下下来

清单看着多，但真正取数据其实很省事——主流框架都内置了下载接口，不用你手动去找网址、解压文件。下面分两个生态各给一个例子。

先看 **PyTorch**，它的 `torchvision` 自带一堆图像数据集，第一次运行会自动下载、之后从本地缓存读：

```python
import torchvision

# download=True：本地没有就自动下载；root 是缓存目录
train = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True
)
print(len(train))          # 训练集样本数：50000
img, label = train[0]      # 取第一张图和它的标签
print(img.size, label)     # (32, 32) 和类别编号
```

跑完你会看到它打印出 `50000`——这说明数据已经准备好，可以直接送进后面的 `DataLoader` 训练了。整个过程你没碰过一次浏览器，这正是用内置接口的好处。

再看 **scikit-learn**，它更偏向小型表格数据集，适合练经典机器学习：

```python
from sklearn.datasets import load_iris

data = load_iris()
X, y = data.data, data.target   # X 是特征矩阵，y 是标签
print(X.shape)                  # (150, 4)：150 个样本，每个 4 个特征
print(data.target_names)        # ['setosa' 'versicolor' 'virginica']
```

`X.shape` 是 `(150, 4)` 告诉你：这是一张 150 行、4 列的表，每一行是一朵花，每一列是一个测量值（花瓣长宽之类）。拿到这个 `X` 和 `y`，你就能直接接上 [1.2 线性回归](../ch2-round1/2-1-2-linear-regression.md)、[1.5 KNN 与决策树](../ch2-round1/2-1-5-knn-decision-trees.md) 里学的那些模型练手了。

文本数据集（IMDB、AG News）则可以用 Hugging Face 的 `datasets` 库一行 `load_dataset("imdb")` 取到，用法和上面类似。无论哪个生态，套路都一样：调一个内置函数，数据就自动下到本地缓存，你拿到的是已经整理好的样本和标签。

## 容易踩的坑

- **数据集别提交进 Git 仓库**：MNIST、CIFAR 动辄几十上百 MB，提交进版本库会把仓库撑爆。把缓存目录（如 `./data`）写进 `.gitignore`，让代码每次自动下载即可。
- **首次下载要联网，比赛环境可能断网**：IOAI 正式赛常常是**离线**环境，临场 `download=True` 会直接失败。务必提前把要用的数据集下到本地、确认能离线读取。
- **官方链接会变**：上面的网址是"找的方向"，不是永久地址。每年开赛前以组委会公告为准，别拿去年的链接硬试。
- **数据许可要看一眼**：有些数据集（尤其人脸、医疗类）有使用限制，正式提交作品前确认它允许你这么用，免得踩版权红线。

## 它在后面会怎么用到

这一页是一张"随用随查"的资源表，几乎贯穿整个备赛过程：[3.2 图像分类实战](../ch2-round1/2-3-2-image-classification.md) 要图像就回来挑 CIFAR，[4.1 文本分类与词嵌入](../ch3-round2/3-4-1-text-embeddings.md) 要文本就回来挑 IMDB；到了模拟阶段，更要拿官方真题加这些公开数据搭出"自测套餐"。配套的学习资料和工具，分别在 [A1 资料中心](a1-resources.md) 和 [A4 环境与工具](a4-tools.md) 里。

## 小结

- 真题优先刷官方的（IOAI / IAIO / 选拔赛），Kaggle 是练手与找类似题的金矿；官方链接每年以公告为准。
- 经典数据集按图像（MNIST / CIFAR）、文本（IMDB / AG News）、表格（Iris / Titanic）三类记，入门先用 MNIST 和 CIFAR-10。
- 主流框架内置下载接口，`torchvision`、`sklearn.datasets`、Hugging Face `datasets` 几行就能取数据——但要提前下好，因为正式赛常常离线。

想系统看数据集目录，可逛 [Hugging Face Datasets](https://huggingface.co/datasets) 和 [Kaggle Datasets](https://www.kaggle.com/datasets)，按任务类型筛选非常方便。
