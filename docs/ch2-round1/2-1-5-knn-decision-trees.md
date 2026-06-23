# 1.5 KNN 与决策树

> **难度** ⭐⭐☆☆☆ · **前置**：[1.4 分类评估指标](2-1-4-classification-metrics.md)

!!! abstract "读完这一节，你会"
    - 理解 KNN"看邻居投票"的思路，会选距离度量和 $k$，也知道它为什么慢
    - 看懂决策树是怎么靠"提问"一步步把数据分开的，会算信息增益和基尼不纯度
    - 能分别从零（NumPy）实现 KNN，再用 scikit-learn 把 KNN 和决策树都跑一遍

## 这一节换个思路：不学参数，直接看数据

前面几节的模型，不管是线性回归还是逻辑回归，套路都一样：先假设一条直线（或一个 sigmoid），再去训练里调出最合适的那组参数 $\mathbf{w}$。但这一节要讲的两个模型，思路完全不同——它们几乎不"训练"参数，而是**直接拿数据本身来做判断**。

为什么要专门讲这两个？因为它们各代表了一种极朴素、也极重要的直觉。**K 近邻（K-Nearest Neighbors，简称 KNN）** 的想法是"近朱者赤"：要判断一个新样本属于哪一类，就看离它最近的几个老样本是什么类，少数服从多数。**决策树（Decision Tree）** 的想法则像我们玩"二十个问题"猜东西：通过一连串"是/否"的提问，把数据一层层切开，最后切到每一小块里都是同一类。

这两种直觉简单到几乎不像"算法"，但它们恰恰是后面 [1.6 集成方法](2-1-6-ensembles.md)里随机森林、梯度提升这些强力模型的地基。所以别小看它们，先把这两块砖砌牢。

## KNN：让最近的邻居替你投票

我们先把 KNN 的完整流程说清楚。给定一个待分类的新点，它做三件事：**第一，算出这个新点到训练集里每一个点的距离；第二，挑出距离最近的 $k$ 个邻居；第三，看这 $k$ 个邻居里哪一类最多，就把新点判给哪一类。** 就这么简单，没有任何需要"训练"出来的参数。

下面这张图把这个过程画了出来。中间那个问号是待分类的新点，我们取 $k=5$，看看它周围最近的 5 个邻居都是谁：

<div class="diagram">
<svg viewBox="0 0 360 240" font-family="-apple-system, 'Segoe UI', sans-serif">
  <!-- 决策圈：把最近 5 个邻居圈进来 -->
  <circle cx="180" cy="120" r="74" fill="var(--dia-accent-soft)" stroke="var(--dia-accent)" stroke-width="1.5" stroke-dasharray="5 4"/>
  <!-- A 类点（蓝，圆）：圈内 2 个 -->
  <circle cx="150" cy="95" r="6" fill="var(--dia-blue)"/>
  <circle cx="215" cy="100" r="6" fill="var(--dia-blue)"/>
  <circle cx="70" cy="55" r="6" fill="var(--dia-blue)"/>
  <circle cx="100" cy="190" r="6" fill="var(--dia-blue)"/>
  <!-- B 类点（绿，方）：圈内 3 个 -->
  <rect x="200" y="150" width="11" height="11" fill="var(--dia-green)"/>
  <rect x="142" y="158" width="11" height="11" fill="var(--dia-green)"/>
  <rect x="225" y="135" width="11" height="11" fill="var(--dia-green)"/>
  <rect x="300" y="60" width="11" height="11" fill="var(--dia-green)"/>
  <rect x="290" y="200" width="11" height="11" fill="var(--dia-green)"/>
  <!-- 待分类的新点（问号） -->
  <circle cx="180" cy="120" r="9" fill="none" stroke="var(--dia-stroke)" stroke-width="2"/>
  <text x="180" y="125" text-anchor="middle" font-size="13" fill="var(--dia-stroke)">?</text>
  <!-- 文字标注全部放在空白处 -->
  <text x="180" y="222" text-anchor="middle" font-size="12" fill="var(--dia-accent-deep)">k = 5 的范围：圈内 3 个绿方 vs 2 个蓝圆 → 判为绿</text>
  <text x="34" y="30" font-size="11" fill="var(--dia-blue)">● A 类</text>
  <text x="100" y="30" font-size="11" fill="var(--dia-green)">■ B 类</text>
</svg>
</div>
<p class="figure-caption">图 1.5-1：KNN 取最近的 5 个邻居投票——绿方 3 票胜过蓝圆 2 票，于是新点被判为绿类。</p>

你可能已经注意到一个有意思的地方：KNN 的"训练"几乎什么都没做，它只是把训练数据**原封不动地存起来**。真正的计算全压在了预测的那一刻——这种"用时才算"的模型，我们叫它**惰性学习（lazy learning）**。这也埋下了它最大的短板，待会儿讲计算代价时再说。

### 距离怎么度量

"最近"这两个字，得先说清楚到底怎么量"近"。最常用的是**欧氏距离（Euclidean distance）**，也就是我们最熟悉的直线距离，对两个点 $\mathbf{x}$ 和 $\mathbf{z}$：

$$d(\mathbf{x},\mathbf{z})=\sqrt{\sum_{j=1}^{n}(x_j-z_j)^2}.$$

它就是中学学的"两点间距离"在 $n$ 维空间的推广。另一个常见的是**曼哈顿距离（Manhattan distance）**，把平方和开方换成绝对值相加 $\sum_j|x_j-z_j|$，相当于在棋盘格街道上只能横平竖直地走。一般表格数据用欧氏距离就好，等学到[进阶特征工程](2-1-10-feature-engineering.md)再讲怎么按数据特点挑度量。

这里有个新手最容易踩、又最致命的坑：**算距离前一定要把特征标准化**。假设一个特征是"年龄"（范围 0–100），另一个是"年收入"（范围 0–1000000），那么算欧氏距离时收入项会以绝对优势压倒年龄项——不是因为收入更重要，纯粹是因为它的数值更大。结果就是 KNN 几乎只按收入找邻居，年龄白给了。所以用 KNN 之前，请务必先做标准化（怎么做见 [4.3 Pandas 与数据处理](../ch1-intro/1-4-3-pandas-data.md)）。

### k 该选多大

那么问题来了——邻居取几个（也就是 $k$ 取多少）才合适？这是 KNN 唯一要你操心的"旋钮"，它直接决定模型的脾气。

如果 $k$ 取得很小，比如 $k=1$，模型就只听最近那一个邻居的。这样它对噪声极其敏感：万一最近的那个点恰好是个标错的样本，新点就跟着错。这种"太把训练数据当回事"的状态，就是我们说的**过拟合（overfitting）**。

反过来，如果 $k$ 取得很大，比如接近整个训练集，那么不管新点在哪，投票结果几乎总是"哪类样本多就归哪类"，模型变得迟钝、抓不住局部差异，这叫**欠拟合（underfitting）**。

所以 $k$ 要折中。实践里有两条经验：一是 $k$ 通常取**奇数**，这样二分类投票不会出现平票；二是不要凭感觉拍脑袋，而是用[交叉验证](2-1-9-model-evaluation.md)在一组候选值里挑出表现最好的那个。

## 决策树：用一连串问题把数据切开

讲完 KNN，我们看第二种直觉。决策树的工作方式，特别像医生问诊或者你玩猜谜：**每次问一个最能区分数据的问题，根据答案把数据分成两拨，再对每一拨继续问，直到每一拨都足够"纯"（基本只剩一类）为止。**

举个最直白的例子。要判断一封邮件是不是垃圾邮件，树可能先问"是否包含'中奖'这个词？"——是的话再问"发件人是否在通讯录里？"，不是的话走另一条分支。每一个问题就是树上的一个**节点**，每一个最终的判断结果就是一片**叶子**。下面这张图画的就是这样一棵小树：

<div class="diagram">
<svg viewBox="0 0 380 240" font-family="-apple-system, 'Segoe UI', sans-serif">
  <!-- 根节点 -->
  <rect x="135" y="14" width="110" height="34" rx="4" fill="var(--dia-bg-card)" stroke="var(--dia-accent)" stroke-width="1.8"/>
  <text x="190" y="35" text-anchor="middle" font-size="12" fill="var(--dia-stroke)">含"中奖"? </text>
  <!-- 第二层左：内部节点 -->
  <rect x="40" y="100" width="118" height="34" rx="4" fill="var(--dia-bg-card)" stroke="var(--dia-accent)" stroke-width="1.8"/>
  <text x="99" y="121" text-anchor="middle" font-size="12" fill="var(--dia-stroke)">在通讯录?</text>
  <!-- 第二层右：叶子 -->
  <rect x="240" y="100" width="100" height="34" rx="4" fill="var(--dia-green-soft)" stroke="var(--dia-green)" stroke-width="1.8"/>
  <text x="290" y="121" text-anchor="middle" font-size="12" fill="var(--dia-stroke)">正常</text>
  <!-- 第三层两片叶子 -->
  <rect x="14" y="186" width="86" height="34" rx="4" fill="var(--dia-green-soft)" stroke="var(--dia-green)" stroke-width="1.8"/>
  <text x="57" y="207" text-anchor="middle" font-size="12" fill="var(--dia-stroke)">正常</text>
  <rect x="116" y="186" width="86" height="34" rx="4" fill="var(--dia-accent-soft)" stroke="var(--dia-accent)" stroke-width="1.8"/>
  <text x="159" y="207" text-anchor="middle" font-size="12" fill="var(--dia-stroke)">垃圾</text>
  <!-- 连线 + 是/否标注（放线旁空白处） -->
  <line x1="160" y1="48" x2="99" y2="100" stroke="var(--dia-stroke-soft)" stroke-width="1.4"/>
  <text x="116" y="78" font-size="11" fill="var(--dia-stroke-soft)">是</text>
  <line x1="220" y1="48" x2="290" y2="100" stroke="var(--dia-stroke-soft)" stroke-width="1.4"/>
  <text x="262" y="78" font-size="11" fill="var(--dia-stroke-soft)">否</text>
  <line x1="80" y1="134" x2="57" y2="186" stroke="var(--dia-stroke-soft)" stroke-width="1.4"/>
  <text x="56" y="166" font-size="11" fill="var(--dia-stroke-soft)">是</text>
  <line x1="118" y1="134" x2="159" y2="186" stroke="var(--dia-stroke-soft)" stroke-width="1.4"/>
  <text x="150" y="166" font-size="11" fill="var(--dia-stroke-soft)">否</text>
</svg>
</div>
<p class="figure-caption">图 1.5-2：一棵判别垃圾邮件的小决策树——从根节点开始顺着"是/否"往下走，最终落到某一片叶子上得到结论。</p>

### 怎么决定"先问哪个问题"

整棵树的关键，全在一句话上：**每一步，都选那个最能把数据分清楚的问题。** 可"分得清不清楚"得有个量化的尺子，这就引出了两个衡量"不纯度"的指标。

我们先建立一个直觉：如果一堆数据里两类各占一半，那它最"混乱"、最不纯；如果全是同一类，那它最"纯净"。我们想要的，是每问一个问题，就让分出来的子集尽量变纯。

第一把尺子是**信息熵（entropy）**，它衡量一堆数据有多"混乱"。设某节点里第 $c$ 类样本占比为 $p_c$，熵定义为

$$H=-\sum_{c} p_c\log_2 p_c.$$

当所有样本同属一类时 $H=0$（最纯），当两类各半时 $H=1$（最乱）。基于熵，我们定义**信息增益（information gain）**：用某个问题划分后，熵下降了多少。下降越多，说明这个问题越有用，于是树就选信息增益最大的那个问题来分。

第二把尺子是**基尼不纯度（Gini impurity）**，它和熵想表达的是同一件事，只是算法略简单、不用算对数：

$$G=1-\sum_{c} p_c^2.$$

它同样在"全是一类"时取 0、在"各类均匀"时取最大。scikit-learn 的决策树**默认就用基尼**，因为算起来更快，效果和用熵几乎没差别。两者你理解一个，另一个自然就通了。

!!! example "例 1：手算一次基尼不纯度"
    一个节点里有 10 个样本，6 个 A 类、4 个 B 类。那么 $p_A=0.6$、$p_B=0.4$，基尼不纯度就是
    $$G=1-(0.6^2+0.4^2)=1-(0.36+0.16)=0.48.$$
    如果某个问题能把它们干净地切成"全 A"和"全 B"两块，那两块的基尼都是 0——下降幅度最大，这正是树最想要的划分。

### 树长太深会过拟合，所以要剪枝

决策树有个天然的毛病：**只要不拦着它，它会一直往下分，直到每片叶子只剩一个样本。** 这样它在训练集上能做到 100% 正确，但其实是把每个训练样本（包括噪声）都死记硬背了下来，换一批新数据就原形毕露——这又是过拟合。

对付的办法叫**剪枝（pruning）**，直觉就是"别让树长得太疯"。最常用、也最该先掌握的是**预剪枝**：在长树的时候就加几条限制，比如树最多 5 层深（`max_depth=5`）、一个节点至少要有 10 个样本才允许继续分（`min_samples_split=10`）。这几个参数是你调决策树时最常拧的旋钮，本质上都是在"拟合训练数据"和"保持简单、能泛化"之间找平衡。

## 动手：从零写 KNN，再用 sklearn 跑一遍

道理讲完了，我们动手验证。KNN 的逻辑足够简单，特别适合先亲手实现一遍、再对照库版，这样你会对"它到底在算什么"心里有底。

先用 NumPy 从零实现。核心就是前面说的三步：算距离、取最近 $k$ 个、投票。注意看注释里每一步对应的是哪一步：

```python
import numpy as np
from collections import Counter

def knn_predict(X_train, y_train, x_new, k=5):
    """对单个新样本 x_new 做 KNN 分类。"""
    # 第一步：算 x_new 到每个训练点的欧氏距离
    dists = np.sqrt(np.sum((X_train - x_new) ** 2, axis=1))
    # 第二步：取距离最小的 k 个的下标
    nearest = np.argsort(dists)[:k]
    # 第三步：看这 k 个邻居里哪一类最多，少数服从多数
    votes = Counter(y_train[nearest])
    return votes.most_common(1)[0][0]
```

这段代码完整复刻了图 1.5-1 的逻辑：`np.argsort` 帮我们把距离从小到大排好序，取前 $k$ 个就是最近的邻居，再用 `Counter` 数票。你会发现整个"训练"过程压根不存在——我们只是在预测时直接用了全部训练数据。

亲手写过之后，真正用的时候我们当然不会自己造轮子，而是用 scikit-learn。同样一件事，它两行就能搞定，而且底层做了大量优化：

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# 务必先标准化！否则数值大的特征会主导距离
X_train_s = StandardScaler().fit_transform(X_train)

clf = KNeighborsClassifier(n_neighbors=5)   # k=5
clf.fit(X_train_s, y_train)                  # "训练"只是把数据存起来
preds = clf.predict(X_test_s)                # 真正的计算在这里发生
```

注意我特意在 `fit` 前面加了一句标准化——这正是前面反复强调的那个坑，库不会替你自动做，得你自己记得。

## 再用 sklearn 跑一棵决策树

决策树的从零实现要递归地选最优划分，代码偏长，对现在的我们性价比不高（理解上面的信息增益/基尼就够了）。所以这里直接看库版，重点放在那几个控制过拟合的参数上：

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

clf = DecisionTreeClassifier(
    criterion="gini",      # 用基尼不纯度（默认）；想用信息增益就填 "entropy"
    max_depth=4,           # 预剪枝：最多 4 层，防止树长太深而过拟合
    min_samples_leaf=5,    # 每片叶子至少 5 个样本，进一步抑制过拟合
    random_state=42,
)
clf.fit(X_train, y_train)        # 决策树不需要标准化，它只看"大于/小于某阈值"

# 把训练好的树画出来，直观看它问了哪些问题
plt.figure(figsize=(10, 6))
plot_tree(clf, filled=True, feature_names=feature_names)
plt.show()
```

这里有个和 KNN 截然不同、值得记住的好处：**决策树不需要标准化**。原因很简单——它每次只问"某个特征是否大于某个阈值"，比的是单个特征自己跟阈值的大小，根本不跨特征算距离，所以特征的量纲差异完全不影响它。这也是决策树在实际工程里很受欢迎的一个原因：省去了一道预处理，还自带可解释性（整棵树画出来，每个判断都看得见）。

## 容易踩的坑

- **KNN 忘了标准化**：这是最致命的坑。量纲大的特征会霸占整个距离，模型实际上只看了一个特征。用 KNN 前请务必 `StandardScaler`。
- **决策树却不要标准化**：反过来，给决策树做标准化纯属多此一举，不会让它变好（虽然也不会变坏）。两个模型在这点上正相反，别记混了。
- **k 取了偶数**：二分类时偶数 $k$ 可能出现平票，结果由实现的"打破平局"规则随机决定，不稳定。优先取奇数。
- **决策树不限深度**：不设 `max_depth` 或 `min_samples_leaf`，树会一路长到把训练集背得滚瓜烂熟，测试集上却惨不忍睹。一上来就该给它加预剪枝。
- **拿 KNN 跑大数据集**：KNN 每预测一个点都要和全部训练样本算距离，训练集一大就慢到没法用——这正是下面要展开的计算代价问题。

## 它在后面会怎么用到

- **KNN 的距离思想**会在[无监督学习](2-1-8-unsupervised.md)的 K-Means 聚类里再次出现，那里同样靠"算距离、找最近"来分组。
- **决策树是集成方法的基本积木**。单棵树容易过拟合，但把很多棵树组合起来（随机森林、梯度提升）就成了 Round 1 表格题里最能打的模型之一，详见 [1.6 集成方法](2-1-6-ensembles.md)。
- 选 $k$、选 `max_depth` 这些都属于"调超参数"，统一的做法是交叉验证，见 [1.9 模型评估与选择](2-1-9-model-evaluation.md)。
- 评判这两个模型分得好不好，回到 [1.4 分类评估指标](2-1-4-classification-metrics.md)里的准确率、F1 等指标。

## 练习

??? note "基础练习"
    1. 用上面的 `knn_predict`，在鸢尾花（iris）数据集上分别试 $k=1$、$k=5$、$k=15$，对比准确率，看看 $k$ 太小和太大各会发生什么。
    2. 手算：一个节点有 8 个 A 类、2 个 B 类样本，求它的基尼不纯度和信息熵。哪个数更接近 0？为什么这说明它已经比较"纯"了？
    3. 把同一份数据分别喂给 KNN（标准化后）和决策树（不标准化），用[分类评估指标](2-1-4-classification-metrics.md)里的方法比较两者的准确率。

??? note "进阶练习"
    1. 给从零版 KNN 加上"加权投票"：让越近的邻居票数权重越大（比如权重取 $1/d$）。对比一下它和普通多数投票在边界附近的差异。
    2. 用 sklearn 的决策树，把 `max_depth` 从 1 一直加到 20，分别记录训练集和测试集的准确率并画成两条曲线。你会清楚看到：训练准确率一路涨，测试准确率先升后降——那个拐点就是过拟合的开始。

## 小结

- KNN 是"看邻居投票"：算距离、取最近 $k$ 个、少数服从多数；它几乎不训练，代价全压在预测时（惰性学习）。**用它一定先标准化，$k$ 优先取奇数。**
- 决策树是"连环提问"：每步选信息增益最大（或基尼下降最多）的问题来切数据，**靠剪枝（限深度、限叶子样本数）防止过拟合**；它不需要标准化，还自带可解释性。
- 这两块砖看着朴素，却是 [1.6 集成方法](2-1-6-ensembles.md)里随机森林、梯度提升的地基，值得吃透。

想直观感受决策树是怎么一刀刀切平面的，推荐玩一玩 [scikit-learn 官方的决策树可视化文档](https://scikit-learn.org/stable/modules/tree.html)；想看 KNN 与决策边界的交互动画，可以去 [R2D3 的图解决策树](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/)看一遍，它用动画把这一节讲得格外清楚。
