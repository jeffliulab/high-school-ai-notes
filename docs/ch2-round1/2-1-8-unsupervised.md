# 1.8 无监督学习

> **难度** ⭐⭐⭐☆☆ · **前置**：[3.1 线性代数](../ch1-intro/1-3-1-linear-algebra.md)、[1.1 监督学习工作流](2-1-1-supervised-workflow.md)

!!! abstract "读完这一节，你会"
    - 说清"无监督"和"有监督"差在哪，并知道聚类、降维各自要解决什么问题
    - 从零写出 K-Means 的 Lloyd 迭代，理解它在最小化什么损失、为什么一定会收敛
    - 高层理解 PCA：它如何借特征分解/SVD 找到"信息最多的方向"来降维
    - 会用 sklearn 跑 K-Means、PCA，并知道 t-SNE/UMAP 该在什么时候拿来可视化

## 没有标签的数据，我们还能学到什么

到这里为止，我们学的全是**有监督学习**——每条数据都配着一个"标准答案"（标签），模型的任务是学会从输入预测这个答案。但现实里，更多的数据其实是**没有答案**的：你手上有一百万张用户的浏览记录，没人告诉你谁是谁的同类；你有一万个手写数字的像素图，却没有标注它们是几。

那么问题来了——没有标签，我们还能学到东西吗？答案是能。这类"数据自己没带答案，靠数据内部的结构去发现规律"的方法，就叫**无监督学习（unsupervised learning）**。它主要干两件事：

1. **聚类（clustering）**：把相似的数据自动归成一堆一堆，比如把用户分成"夜猫子""周末党""通勤族"。
2. **降维（dimensionality reduction）**：把又高又冗余的特征压缩成几个"最有信息量"的维度，既方便可视化，也能给后续模型减负。

这一节我们就抓两个最经典、竞赛里也最常考的代表：聚类里的 **K-Means**，降维里的 **PCA**。把这两个吃透，无监督学习的半壁江山就在你手里了。

## K-Means：把点分成 k 堆的最朴素办法

先说最直观的聚类算法 K-Means。它的目标一句话就能讲清楚：**给定一堆点，把它们分成 $k$ 堆，让每堆内部的点尽量挨得近。**

怎么衡量"挨得近"？我们用最熟悉的**欧氏距离（Euclidean distance）**——也就是两点之间的直线距离。两个点 $\mathbf{x}$、$\mathbf{y}$ 的距离是

$$
\lVert \mathbf{x}-\mathbf{y}\rVert = \sqrt{\sum_{j}(x_j - y_j)^2}.
$$

每一堆点，我们都用它的"重心"来代表，这个重心叫**簇中心（centroid）**，其实就是这堆点各个坐标的平均值。于是 K-Means 想最小化的，是所有点到自己那堆中心的距离平方之和，这个量有个名字叫**簇内平方和（inertia / within-cluster sum of squares）**：

$$
J = \sum_{i=1}^{n} \lVert \mathbf{x}_i - \boldsymbol{\mu}_{c_i} \rVert^2,
$$

这里 $\mathbf{x}_i$ 是第 $i$ 个点，$c_i$ 是它被分到的簇编号，$\boldsymbol{\mu}_{c_i}$ 是那个簇的中心。$J$ 越小，说明每堆都抱得越紧、分得越好。我们的全部目标，就是找一组分配方式和一组中心，把 $J$ 压到最低。

### Lloyd 迭代：两步轮流走，越走越好

直接一口气求出让 $J$ 最小的方案是很难的（这其实是个 NP 难问题）。但有一个聪明又朴素的近似办法，叫 **Lloyd 算法**，它把问题拆成两步轮流做：

- **第一步（分配 / assignment）**：固定当前的 $k$ 个中心，把每个点划给离它最近的那个中心。
- **第二步（更新 / update）**：固定刚才的划分，把每个中心挪到它那堆点的平均位置上。

然后回到第一步，再分配、再更新……如此反复，直到中心不再移动为止。为什么这样能行？因为**每一步都只会让 $J$ 变小或不变**：分配步让每个点投靠最近的中心，距离当然不会变大；更新步把中心移到重心，而"到一堆点距离平方和最小的位置"恰好就是它们的平均值。两步都在下降，$J$ 又不会小于零，所以它**一定会收敛**。下面这张图把这两步画了出来。

<div class="diagram">
<svg viewBox="0 0 420 200" font-family="-apple-system, 'Segoe UI', sans-serif">
  <!-- 左：分配步 -->
  <text x="95" y="22" text-anchor="middle" font-size="12" fill="var(--dia-stroke-soft)">分配：点投靠最近的中心</text>
  <!-- 簇 A 的点（蓝） -->
  <circle cx="45" cy="70" r="4" fill="var(--dia-blue)"/>
  <circle cx="62" cy="58" r="4" fill="var(--dia-blue)"/>
  <circle cx="55" cy="88" r="4" fill="var(--dia-blue)"/>
  <!-- 簇 B 的点（绿） -->
  <circle cx="130" cy="120" r="4" fill="var(--dia-green)"/>
  <circle cx="148" cy="135" r="4" fill="var(--dia-green)"/>
  <circle cx="120" cy="148" r="4" fill="var(--dia-green)"/>
  <!-- 两个中心（叉） -->
  <path d="M48 68 l10 10 M58 68 l-10 10" stroke="var(--dia-accent-deep)" stroke-width="2"/>
  <path d="M128 130 l10 10 M138 130 l-10 10" stroke="var(--dia-accent-deep)" stroke-width="2"/>
  <!-- 分隔线 -->
  <line x1="210" y1="40" x2="210" y2="175" stroke="var(--dia-rule)" stroke-width="1"/>
  <!-- 右：更新步 -->
  <text x="320" y="22" text-anchor="middle" font-size="12" fill="var(--dia-stroke-soft)">更新：中心移到各堆重心</text>
  <circle cx="270" cy="70" r="4" fill="var(--dia-blue)"/>
  <circle cx="287" cy="58" r="4" fill="var(--dia-blue)"/>
  <circle cx="280" cy="88" r="4" fill="var(--dia-blue)"/>
  <circle cx="355" cy="120" r="4" fill="var(--dia-green)"/>
  <circle cx="373" cy="135" r="4" fill="var(--dia-green)"/>
  <circle cx="345" cy="148" r="4" fill="var(--dia-green)"/>
  <!-- 中心移到重心位置 -->
  <path d="M274 67 l10 10 M284 67 l-10 10" stroke="var(--dia-accent-deep)" stroke-width="2"/>
  <path d="M353 130 l10 10 M363 130 l-10 10" stroke="var(--dia-accent-deep)" stroke-width="2"/>
  <!-- 移动小箭头标注 -->
  <text x="320" y="190" text-anchor="middle" font-size="11" fill="var(--dia-stroke-tertiary)">✕ = 簇中心</text>
</svg>
</div>
<p class="figure-caption">图 1.8-1：Lloyd 迭代的一轮——先把每个点分给最近的中心（左），再把中心挪到各自那堆点的平均位置（右）；两步交替，直到中心不再动。</p>

### 从零实现一遍，你才真懂

道理讲完了，我们动手把 Lloyd 算法写出来。下面这段代码先随机挑 $k$ 个点当初始中心，然后老老实实地"分配—更新"循环。注意看每一步是怎么对应上面那两步的：

```python
import numpy as np

def kmeans(X, k, n_iter=100, seed=0):
    """X: (n, d) 数据；返回每个点的簇标签和最终的 k 个中心。"""
    rng = np.random.default_rng(seed)
    # 初始化：从数据里随机抽 k 个点当中心
    centers = X[rng.choice(len(X), k, replace=False)]

    for _ in range(n_iter):
        # 第一步·分配：算每个点到每个中心的距离，取最近的那个
        # dist 形状 (n, k)，dist[i, j] = 点 i 到中心 j 的距离平方
        dist = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = dist.argmin(axis=1)        # 每个点归到最近的中心

        # 第二步·更新：把每个中心移到它那堆点的平均位置
        new_centers = np.array([
            X[labels == j].mean(axis=0) if np.any(labels == j) else centers[j]
            for j in range(k)
        ])

        # 收敛判断：中心几乎不动了就停
        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    return labels, centers
```

这段代码里最值得品的是分配那一步用了**广播（broadcasting）**：`X[:, None, :]` 把数据撑成 $(n,1,d)$，`centers[None, :, :]` 撑成 $(1,k,d)$，相减后自动得到 $(n,k,d)$，再沿最后一维求平方和，一次性算出了所有"点到中心"的距离平方，完全没用 `for` 循环。这正是 [3.1 线性代数](../ch1-intro/1-3-1-linear-algebra.md) 里向量化思维的实战。

### 库版只要两行

理解了内部机理，实战里我们当然不会自己手写——sklearn 早就封装好了，而且做了更好的初始化（k-means++）和多次重启。同样的活，库版只要两行：

```python
from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, n_init=10, random_state=0)
labels = km.fit_predict(X)      # 直接拿到每个点的簇标签
print(km.inertia_)              # 这就是上面那个损失 J，越小越紧
```

`n_init=10` 表示用 10 组不同的初始中心各跑一遍、留最好的那次——这是为了对付 K-Means 一个老毛病，我们马上在"坑"里细说。`inertia_` 打印出来的，正是我们前面推导的损失 $J$。

## 怎么定 k：肘部法则

K-Means 有个绕不开的问题：**$k$ 得你自己指定**，可数据本来该分几堆，往往并不知道。一个常用的经验办法叫**肘部法则（elbow method）**：把 $k$ 从小到大试一遍，画出每个 $k$ 对应的损失 $J$。$k$ 越大 $J$ 当然越小（极端情况下每个点自成一堆，$J=0$），但你会发现曲线在某个 $k$ 之后下降突然变缓——那个"胳膊肘"般的拐点，通常就是比较合适的簇数。

```python
inertias = [KMeans(n_clusters=k, n_init=10, random_state=0).fit(X).inertia_
            for k in range(1, 8)]
# 把 inertias 对 k 画成折线，找下降明显变缓的那个拐点
```

肘部法则不是铁律，有时拐点并不明显，这时还可以参考**轮廓系数（silhouette score）**等指标，我们在 [1.9 模型评估与选择](2-1-9-model-evaluation.md) 里会再碰到评估这件事。

## PCA：把高维数据"压扁"到最有信息的方向

聊完聚类，我们转向降维的当家算法 **PCA（主成分分析，Principal Component Analysis）**。先讲它要解决的痛点：很多数据的维度高得吓人（一张 $28\times28$ 的图就是 784 维），但这些维度里有大量冗余和噪声。PCA 想做的，是**找几个新的方向，让数据投影上去之后还尽量保留原来的"差异"，从而用更少的维度概括大部分信息**。

那什么叫"保留差异"？这里的关键词是**方差（variance）**：一个方向上数据铺得越开、方差越大，说明这个方向携带的信息越多。所以 PCA 的直觉就是——**优先保留方差最大的那些方向，丢掉方差很小的方向。**

怎么找这些方向？这就用上了 [3.1 线性代数](../ch1-intro/1-3-1-linear-algebra.md) 里的特征分解。具体三步：

1. 先把数据**中心化**：每个特征减去它的均值，让数据围着原点。
2. 算数据的**协方差矩阵（covariance matrix）** $\Sigma=\frac{1}{n}X^\top X$，它刻画了各维度之间一起变化的程度。
3. 对 $\Sigma$ 做**特征分解**：得到的特征向量就是那些"主方向"（主成分），对应的特征值则告诉你每个方向上方差有多大。

把特征值从大到小排好，**取前 $k$ 个特征向量当新坐标轴，把数据投影上去，就完成了降到 $k$ 维。** 下面这张图画的是最常见的情形：一团斜着铺开的二维点，PCA 找出的第一主成分正好顺着它"最长"的方向。

<div class="diagram">
<svg viewBox="0 0 360 210" font-family="-apple-system, 'Segoe UI', sans-serif">
  <defs>
    <marker id="pc1" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="var(--dia-accent)"/></marker>
    <marker id="pc2" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="var(--dia-blue)"/></marker>
  </defs>
  <!-- 一团斜向铺开的点，整体趋势沿对角线 -->
  <circle cx="90" cy="150" r="3.5" fill="var(--dia-stroke-soft)"/>
  <circle cx="115" cy="138" r="3.5" fill="var(--dia-stroke-soft)"/>
  <circle cx="140" cy="120" r="3.5" fill="var(--dia-stroke-soft)"/>
  <circle cx="160" cy="118" r="3.5" fill="var(--dia-stroke-soft)"/>
  <circle cx="180" cy="100" r="3.5" fill="var(--dia-stroke-soft)"/>
  <circle cx="205" cy="92" r="3.5" fill="var(--dia-stroke-soft)"/>
  <circle cx="225" cy="78" r="3.5" fill="var(--dia-stroke-soft)"/>
  <circle cx="150" cy="135" r="3.5" fill="var(--dia-stroke-soft)"/>
  <circle cx="195" cy="108" r="3.5" fill="var(--dia-stroke-soft)"/>
  <circle cx="130" cy="108" r="3.5" fill="var(--dia-stroke-soft)"/>
  <!-- 第一主成分：顺着点云最长方向 -->
  <line x1="80" y1="158" x2="240" y2="68" stroke="var(--dia-accent)" stroke-width="2.5" marker-end="url(#pc1)"/>
  <!-- 第二主成分：垂直于第一主成分，方差小 -->
  <line x1="160" y1="113" x2="128" y2="78" stroke="var(--dia-blue)" stroke-width="2" marker-end="url(#pc2)"/>
  <!-- 标注放在空白处，避开点云与主成分线 -->
  <text x="252" y="64" font-size="12" fill="var(--dia-accent)">第一主成分（方差大）</text>
  <text x="20" y="186" font-size="12" fill="var(--dia-blue)">第二主成分（方差小，可丢）</text>
</svg>
</div>
<p class="figure-caption">图 1.8-2：PCA 找出数据铺得最开的方向当第一主成分；垂直方向方差小、信息少，降维时第一个被舍弃。</p>

实战里更稳的算法其实不是直接对协方差矩阵做特征分解，而是对中心化后的数据矩阵做 **SVD（奇异值分解）**，结果等价但数值更稳——这也是 sklearn 内部的做法。我们仍然只要几行：

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)        # 想降到 2 维（常用于可视化）
X2 = pca.fit_transform(X)       # X2 形状 (n, 2)
print(pca.explained_variance_ratio_)   # 每个主成分保留了百分之多少的方差
```

那个 `explained_variance_ratio_` 特别有用：它告诉你前 2 个主成分一共留住了原数据多大比例的信息。如果加起来有 0.9，说明你用 2 维就概括了九成信息，降维降得很值。

## t-SNE 和 UMAP：只为"看一眼"高维数据

最后简单提两个你会在论文图里经常看到的名字：**t-SNE** 和 **UMAP**。它们也是降维，但目标和 PCA 不太一样——PCA 追求"保留全局的方差结构"，而这两个追求"把高维里挨得近的点，在二维图上也画得近"，专门用来**把高维数据可视化成一张好看、聚类清晰的散点图**。

对竞赛来说，你**只需要会用、会看，不必懂内部数学**。一句话记住它们的定位：

```python
from sklearn.manifold import TSNE

X2 = TSNE(n_components=2, random_state=0).fit_transform(X)
# 把 X2 画成散点图、按真实类别上色，常能看到漂亮的"成团"效果
```

但要留个心眼：t-SNE/UMAP 画出来的**簇之间的距离、簇的大小都不能当真**，它们只适合"定性地看看数据有没有结构"，绝不能拿坐标值去做后续计算。这一点考试爱设陷阱，务必记牢。

## 容易踩的坑

- **K-Means 会陷进局部最优**：初始中心选得不好，结果可能很糟，而且每次跑还不一样。所以一定要设 `n_init`（多跑几组初始值取最好），sklearn 默认用 k-means++ 初始化也是为此。
- **不做标准化就聚类**：欧氏距离对量纲极其敏感。如果一个特征是"年龄（0~100）"、另一个是"收入（0~100000）"，收入会完全主导距离，年龄形同虚设。聚类前几乎总要先标准化（见 [4.3 Pandas 与数据处理](../ch1-intro/1-4-3-pandas-data.md)）。
- **K-Means 只爱"圆球状"的簇**：它假设每堆是各向同性的团。遇到月牙形、环形这种数据，K-Means 会切得乱七八糟，这时该换 DBSCAN 一类基于密度的方法。
- **PCA 前忘了中心化**：不减均值就做分解，找出的方向会被数据整体的偏移带偏，主成分就不对了。sklearn 的 `PCA` 会自动帮你中心化，但自己从零写时千万别漏。
- **拿 t-SNE 的坐标做计算**：前面强调过——它的坐标只能看、不能算，更不能喂给下游模型当特征。

## 它在后面会怎么用到

- **降维是特征工程的常规武器**：PCA 既能给模型减负、去冗余，也能压噪声，[1.10 进阶特征工程](2-1-10-feature-engineering.md) 会把它当成标准工具来用。
- **聚类常用来"无标签时先探路"**：在 [1.9 模型评估与选择](2-1-9-model-evaluation.md) 里，我们会聊怎么评估这种没有标准答案的结果。
- **自编码器是降维的"神经网络版"**：到了 Round 2，[2.3 自编码器](../ch3-round2/3-2-3-autoencoders.md) 会用神经网络学到比 PCA 更强的非线性压缩，思路一脉相承。
- **嵌入表示也是降维的近亲**：把高维稀疏的东西压成稠密低维向量，[1.2 嵌入表示](../ch3-round2/3-1-2-embeddings.md) 用的就是这个大思路。

## 练习

??? note "基础练习"
    1. 用 sklearn 造一份三团的二维数据（`make_blobs`），分别用上面手写的 `kmeans` 和 `KMeans` 跑一遍，画出散点图按簇上色，确认两者分出来的堆基本一致。
    2. 对同一份数据，把 $k$ 从 1 试到 7，画出 `inertia_` 随 $k$ 的折线，用肘部法则判断它真实该分几堆，看看和你造数据时的设定对不对得上。
    3. 取 sklearn 自带的鸢尾花数据（4 维），用 PCA 降到 2 维并画散点图，再打印 `explained_variance_ratio_`，说说前两个主成分一共留住了多少信息。

??? note "进阶练习"
    1. 给手写的 K-Means 加一个"多次重启取最优"的外壳：跑 10 组不同初始中心，每次记下最终的损失 $J$，返回 $J$ 最小的那次结果。亲手体会 `n_init` 在防局部最优上的作用。
    2. 故意把鸢尾花数据的某一列乘以 1000（制造量纲悬殊），分别在"标准化"和"不标准化"两种情况下跑 K-Means，对比聚类结果差多少，亲眼验证"坑"里说的量纲问题。
    3. 对一张 $50\times50$ 的灰度图，把每一行当成一个样本做 PCA，只保留前 $k$ 个主成分再重建，观察 $k=5$ 和 $k=20$ 时清晰度的差别——这正是 [3.1 线性代数](../ch1-intro/1-3-1-linear-algebra.md) 里 SVD 低秩压缩的同一件事。

## 小结

- 无监督学习不靠标签，靠数据自身的结构说话，两大主力是**聚类**和**降维**。
- **K-Means** 用欧氏距离把点分成 $k$ 堆，靠 Lloyd 迭代"分配—更新"交替下降损失 $J$，一定收敛、但可能陷局部最优，且对量纲敏感。
- **PCA** 借协方差的特征分解（实战用 SVD）找方差最大的方向，优先保留信息多的维度来降维。
- **t-SNE/UMAP** 只用来可视化，会用、会看就行，坐标不能当真。

想把 PCA 的几何直觉建牢，推荐看 StatQuest 的 PCA 讲解视频，它用动画把"主成分"讲得很清楚；更系统的推导见 [*Mathematics for Machine Learning*](https://mml-book.github.io/) 第 10 章。sklearn 官方的 [Clustering](https://scikit-learn.org/stable/modules/clustering.html) 文档则是一份很好的"各种聚类算法适用场景"对照表。
