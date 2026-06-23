# 1.6 集成方法

> **难度** ⭐⭐⭐☆☆ · **前置**：[1.5 KNN 与决策树](2-1-5-knn-decision-trees.md)、[1.4 分类评估指标](2-1-4-classification-metrics.md)

!!! abstract "读完这一节，你会"
    - 用"三个臭皮匠顶个诸葛亮"的直觉，说清为什么把很多弱模型组合起来反而更强
    - 理解 Bagging 和随机森林是怎么靠"制造差异"来降低方差的，并能从零写一个袋装分类器
    - 在高层上看懂梯度提升（GBDT/XGBoost）是怎么"一棵接一棵地补错"的
    - 会用 scikit-learn 的 `RandomForestClassifier` 和 `GradientBoostingClassifier` 几行代码上手，并知道各自的关键超参数

## 一棵树不够，那就种一片森林

上一节我们学了决策树。它有个很讨喜的优点：直观、好解释、不用做特征缩放。可它也有个很要命的毛病——**一棵树太"敏感"了**。你只要把训练数据稍微改动几条，长出来的树可能就完全变样，预测也跟着翻脸。这种"对数据抖动特别敏感、稍微换批数据结果就大变"的现象，专业上叫做**高方差（high variance）**。

那怎么办呢？这里有一个特别朴素、却出奇有效的想法：**既然一棵树不靠谱，那我就种很多棵，让它们投票。** 这就好比一道难题，单独问一个人容易答错，但让一百个见识各不相同的人各自答一遍、再取多数意见，整体往往就靠谱多了。把多个模型组合起来一起做决定，这件事就叫**集成学习（ensemble learning）**，里面每一个单独的模型叫**弱学习器（weak learner）**——它单打独斗时可能只比瞎猜强一点点。

为什么"凑一堆弱的"能变强？关键在一个词：**多样性**。如果一百棵树长得一模一样，那它们错也错在一块儿，投票毫无意义。只有当每棵树各有各的脾气、各犯各的错时，它们的错误才会在投票里互相抵消，而它们共同认对的部分会被保留下来。这一节的两大主角——随机森林和梯度提升——本质上就是两种"制造多样性"的不同思路。

## Bagging：靠"重新抽数据"造出一片不一样的树

第一种思路叫 **Bagging**，全称是 Bootstrap Aggregating，可以拆成"自助采样 + 聚合"两步来理解。

先说**自助采样（bootstrap）**。假设你手里有 $N$ 条训练数据。我们不直接拿这一整份去训练，而是**有放回地随机抽 $N$ 次**——抽出一条记下来再放回去，所以同一条可能被抽中好几次，也有些条一次都没被抽到。这样抽一轮，就得到一份和原数据"像但不完全一样"的新数据集。重复这个过程 $T$ 次，就得到 $T$ 份各不相同的数据，在每一份上各训练一棵树。因为喂给它们的数据本来就有差异，长出来的树自然也就各有不同——多样性就这么造出来了。

再说**聚合（aggregating）**。$T$ 棵树都训练好之后，预测时让它们各报一票：分类问题就取**多数表决**，回归问题就取**平均值**。

为什么这招能降方差？我们可以从数学上看一眼。假设有 $T$ 个预测，每个的方差都是 $\sigma^2$，并且它们两两之间的相关系数是 $\rho$。那么取平均之后，整体的方差是：

$$
\mathrm{Var}\!\left(\frac{1}{T}\sum_{i=1}^{T} f_i\right) = \rho\,\sigma^2 + \frac{1-\rho}{T}\,\sigma^2.
$$

这个式子很说明问题。右边第二项里有个 $\frac{1}{T}$，意味着**只要树足够多，这一项就趋近于 0**——这就是"投票/平均能降方差"的来源。但第一项 $\rho\sigma^2$ 不含 $T$，它告诉我们：**如果各棵树高度相关（$\rho$ 接近 1），那再怎么加树也压不下方差。** 换句话说，光靠多还不够，还得让树彼此"不一样"。这一点正是随机森林要进一步解决的。

## 随机森林：在 Bagging 上再加一把"随机"

随机森林（Random Forest）就是把 Bagging 用在决策树上，然后再加一个聪明的小动作，专门去对付上面那个 $\rho$。

光靠自助采样造出来的树，其实还是挺像的——因为只要某个特征特别强势，每棵树的根节点八成都会去分裂这个特征，结果大家长得大同小异，$\rho$ 降不下来。随机森林的解法是：**每次分裂节点时，不让树看全部特征，而是随机只给它一个特征子集去挑。** 一棵树有的节点看不到那个最强势的特征，就被迫去用别的特征分裂，于是各棵树的"长相"被强行拉开了距离。一个常用的默认值是：每次分裂只考虑 $\sqrt{d}$ 个特征（$d$ 是特征总数）。

下面这张图把随机森林的整个流程画了出来，你可以从左到右顺着走一遍。

<div class="diagram">
<svg viewBox="0 0 440 230" font-family="-apple-system, 'Segoe UI', sans-serif">
  <defs>
    <marker id="rf-a" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="var(--dia-stroke-soft)"/></marker>
  </defs>
  <!-- 原始数据 -->
  <rect x="10" y="95" width="70" height="40" rx="4" fill="var(--dia-bg-card)" stroke="var(--dia-stroke)" stroke-width="1.2"/>
  <text x="45" y="112" font-size="11" fill="var(--dia-stroke)" text-anchor="middle">原始</text>
  <text x="45" y="126" font-size="11" fill="var(--dia-stroke)" text-anchor="middle">数据</text>
  <!-- 三份自助样本到三棵树 -->
  <line x1="80" y1="115" x2="150" y2="55" stroke="var(--dia-stroke-soft)" stroke-width="1" marker-end="url(#rf-a)"/>
  <line x1="80" y1="115" x2="150" y2="115" stroke="var(--dia-stroke-soft)" stroke-width="1" marker-end="url(#rf-a)"/>
  <line x1="80" y1="115" x2="150" y2="175" stroke="var(--dia-stroke-soft)" stroke-width="1" marker-end="url(#rf-a)"/>
  <circle cx="170" cy="50" r="18" fill="var(--dia-accent-soft)" stroke="var(--dia-accent)" stroke-width="1.4"/>
  <text x="170" y="54" font-size="11" fill="var(--dia-accent-deep)" text-anchor="middle">树1</text>
  <circle cx="170" cy="115" r="18" fill="var(--dia-accent-soft)" stroke="var(--dia-accent)" stroke-width="1.4"/>
  <text x="170" y="119" font-size="11" fill="var(--dia-accent-deep)" text-anchor="middle">树2</text>
  <circle cx="170" cy="180" r="18" fill="var(--dia-accent-soft)" stroke="var(--dia-accent)" stroke-width="1.4"/>
  <text x="170" y="184" font-size="11" fill="var(--dia-accent-deep)" text-anchor="middle">树3</text>
  <text x="110" y="40" font-size="9" fill="var(--dia-stroke-tertiary)" text-anchor="middle">自助采样</text>
  <!-- 各自投票 -->
  <line x1="188" y1="50" x2="290" y2="100" stroke="var(--dia-stroke-soft)" stroke-width="1" marker-end="url(#rf-a)"/>
  <line x1="188" y1="115" x2="290" y2="115" stroke="var(--dia-stroke-soft)" stroke-width="1" marker-end="url(#rf-a)"/>
  <line x1="188" y1="180" x2="290" y2="130" stroke="var(--dia-stroke-soft)" stroke-width="1" marker-end="url(#rf-a)"/>
  <!-- 投票框 -->
  <rect x="295" y="90" width="65" height="50" rx="4" fill="var(--dia-bg-deep)" stroke="var(--dia-stroke)" stroke-width="1.2"/>
  <text x="327" y="110" font-size="11" fill="var(--dia-stroke)" text-anchor="middle">多数</text>
  <text x="327" y="124" font-size="11" fill="var(--dia-stroke)" text-anchor="middle">表决</text>
  <!-- 最终输出 -->
  <line x1="360" y1="115" x2="400" y2="115" stroke="var(--dia-stroke-soft)" stroke-width="1" marker-end="url(#rf-a)"/>
  <text x="420" y="119" font-size="12" fill="var(--dia-green)" text-anchor="middle" font-weight="600">预测</text>
</svg>
</div>
<p class="figure-caption">图 1.6：随机森林——先用自助采样造出多份带差异的数据各训一棵树（每次分裂还随机限制候选特征），预测时让所有树投票表决。</p>

注意一个关键点：**Bagging 里所有树是彼此独立、可以并行训练的**，谁也不用等谁。这让随机森林又快又稳，几乎不用怎么调参就能给出一个相当强的基线，是 Round 1 表格题里最值得先试的模型之一。

## 动手实现：从零写一个袋装分类器，再对照 sklearn

道理讲完了，我们动手把 Bagging 最核心的"自助采样 + 投票"亲手实现一遍。为了聚焦在集成本身，单棵树我们直接借用 sklearn 的 `DecisionTreeClassifier` 当弱学习器：

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import mode

class MyBagging:
    def __init__(self, n_estimators=20, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        n = len(X)
        self.trees = []
        for _ in range(self.n_estimators):
            # 自助采样：有放回地抽 n 个下标，于是有的样本重复、有的缺席
            idx = np.random.choice(n, size=n, replace=True)
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X[idx], y[idx])      # 在这份"抖动过"的数据上训一棵树
            self.trees.append(tree)
        return self

    def predict(self, X):
        # 收集每棵树的预测，形状是 (树数, 样本数)
        preds = np.array([t.predict(X) for t in self.trees])
        # 沿"树"这个维度取众数 = 多数表决
        return mode(preds, axis=0, keepdims=False).mode

# 跑一个真实的小数据集看看
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)

single = DecisionTreeClassifier(random_state=0).fit(Xtr, ytr)
bag = MyBagging(n_estimators=30).fit(Xtr, ytr)
print("单棵树   :", (single.predict(Xte) == yte).mean())
print("袋装30棵 :", (bag.predict(Xte) == yte).mean())
```

跑下来你通常会看到，袋装版的准确率比单棵树明显高出一截。**这正是前面那条方差公式在现实里的体现**：三十棵在不同数据上长出来的树各犯各的错，投票一过滤，整体就稳了。

理解了原理之后，实战里我们当然不会自己手写——直接用随机森林就好，它在我们上面的基础上还多加了"随机特征子集"那一招：

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=300,     # 树越多越稳，但收益递减、也更慢
    max_features="sqrt",  # 每次分裂只看 √d 个特征——这就是"随机"的来源
    n_jobs=-1,            # 用上所有 CPU 核心并行训练
    random_state=0,
)
rf.fit(Xtr, ytr)
print("随机森林 :", rf.score(Xte, yte))

# 顺手看看哪些特征最重要——随机森林免费送的可解释性
import numpy as np
top = np.argsort(rf.feature_importances_)[::-1][:5]
print("最重要的 5 个特征下标:", top)
```

最后那个 `feature_importances_` 很实用：随机森林在训练中统计了"每个特征带来了多少纯度提升"，于是顺带就告诉你哪些特征最有用——这在做特征筛选时是个很方便的参考。

## 梯度提升：一棵接一棵，专门补上一轮的错

随机森林是"大家各自独立、最后一起投票"，属于**并行**的思路。还有一条完全不同的路子，叫**提升（Boosting）**，它是**串行**的：树一棵接一棵地训练，**每一棵新树的任务，是去修正前面所有树合起来还没做好的地方**。

我们重点理解其中最主流的一种——**梯度提升（Gradient Boosting）**。它的核心直觉可以这样讲：先让第一棵很浅的树做个粗糙的预测，它肯定会有误差；于是我们算出"还差多少"（这个差距叫**残差**），让第二棵树专门去拟合这个残差；第二棵补完还剩一点误差，再让第三棵去补……如此一轮轮地"亡羊补牢"，预测就越来越准。把它写成式子，就是模型一点点累加起来：

$$
F_{m}(x) = F_{m-1}(x) + \eta\, h_m(x),
$$

这里 $F_{m-1}$ 是前 $m-1$ 棵树的合力，$h_m$ 是新加的这棵专门补错的树，而 $\eta$ 是**学习率（learning rate）**——它控制每棵新树"补"多少。为什么叫"梯度"提升？因为可以证明，这里让新树去拟合的"残差"，本质上就是损失函数对当前预测的**负梯度**。换句话说，**它是在用一棵棵决策树，做一种特殊的梯度下降**——你在 [3.2 微积分与梯度](../ch1-intro/1-3-2-calculus-gradients.md) 里建立的下山直觉，在这里又出现了一次，只不过这次"每一步"是加一棵树。

Bagging 和 Boosting 的差别值得记牢，因为它俩擅长的事不一样：

| 对比项 | Bagging / 随机森林 | Boosting / 梯度提升 |
| --- | --- | --- |
| 树怎么训 | 互相独立，**可并行** | 一棵接一棵，**必须串行** |
| 主要降低 | 方差（让模型更稳） | 偏差（让模型更准） |
| 单棵树 | 通常长得很深 | 通常很浅（弱学习器） |
| 调参难度 | 低，开箱即用 | 高，对学习率、树数敏感 |
| 过拟合风险 | 树多了基本不增 | 树太多 / 学习率太大会过拟合 |

实战里，梯度提升（以及它的高效实现 XGBoost、LightGBM）几乎是表格类竞赛的"夺冠常客"，调好了往往比随机森林再高一档。我们先用 sklearn 自带的版本体会一下：

```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=200,     # 串行的树数；和学习率要配合着调
    learning_rate=0.1,    # η：每棵新树补多少，小一点更稳但需要更多树
    max_depth=3,          # 每棵都是浅树（弱学习器），这是 Boosting 的惯例
    random_state=0,
)
gb.fit(Xtr, ytr)
print("梯度提升 :", gb.score(Xte, yte))
```

这里 `learning_rate` 和 `n_estimators` 是一对要一起调的"冤家"：**学习率调小会让每棵树更谨慎、最终更不容易过拟合，但作为代价你得种更多的树才能学到位。** 一个常见的稳妥组合是"小学习率 + 多树 + 早停"。

## 容易踩的坑

- **以为随机森林的树越多越会过拟合**：恰恰相反，Bagging 加树只会让结果更稳，几乎不会因为树多而过拟合；多到一定程度只是收益递减、变慢而已。真正会因为"太多"而过拟合的是 Boosting。
- **把 Boosting 也想成能并行**：梯度提升的每棵树都依赖前一棵的结果，**天生是串行的**，没法像随机森林那样靠 `n_jobs=-1` 加速。想要快，得换 LightGBM 这类专门优化过的实现。
- **学习率和树数分开调**：这两个参数耦合得很紧。把学习率从 0.1 改到 0.01 却不相应地增加 `n_estimators`，模型很可能"没学够"就停了，表现反而变差。
- **盲目相信特征重要性**：随机森林的 `feature_importances_` 对取值很多的特征（比如连续值、ID 类）有系统性偏好，会高估它们。要更可靠的结论，可以改用基于打乱的排列重要性（permutation importance）。
- **忘了 Boosting 对噪声敏感**：因为每棵新树都在努力拟合上一轮的残差，数据里的离群点和标签噪声会被它反复盯着补，容易过拟合。上 Boosting 之前，先把数据清干净。

## 它在后面会怎么用到

集成方法不是一个孤立的知识点，它在后面的关卡里会反复成为你的"涨分利器"：

- **限时拿分的首选武器**：在 [4.1 真题形式与评分](2-4-1-exam-format.md) 这类限时表格题里，随机森林几乎是开箱即用的强基线，而梯度提升常常是冲分上限的关键——所以这一节务必练熟到"闭着眼睛也能写出来"。
- **建立在决策树之上**：本节的一切都以 [1.5 KNN 与决策树](2-1-5-knn-decision-trees.md) 的单棵树为积木，回头把树的分裂准则再确认一遍会理解得更深。
- **要靠评估指标来判好坏**：集成模型调得好不好，得用 [1.4 分类评估指标](2-1-4-classification-metrics.md) 里的准确率、F1、AUC 来量；而怎么公平地比较不同模型、避免被随机性骗到，见 [1.9 模型评估与选择](2-1-9-model-evaluation.md)。
- **和特征工程相辅相成**：梯度提升对好特征极其敏感，[1.10 进阶特征工程](2-1-10-feature-engineering.md) 里的技巧能直接转化成分数。

## 练习

??? note "基础练习"
    1. 在 `load_breast_cancer` 上，把随机森林的 `n_estimators` 依次设成 1、5、20、100、500，画出测试准确率随树数变化的曲线。你会观察到：准确率先快速上升，然后趋于平稳——亲眼验证一下"加树降方差但收益递减"。
    2. 拿上面手写的 `MyBagging`，把单棵树的预测和袋装后的预测各跑 10 次（每次换 `random_state`），比较两者准确率的**波动幅度**。哪个更稳？这正是"方差"的直观含义。

??? note "进阶练习"
    1. 在同一个数据集上，固定梯度提升的总"学习量"，试三组配置：`(learning_rate=0.3, n_estimators=50)`、`(0.1, 150)`、`(0.03, 500)`，比较它们的测试表现和训练时间。总结一下"小学习率 + 多树"为什么往往更稳。
    2. 自己动手写一个最简版的**回归**梯度提升：用一棵浅树拟合 $y$，算出残差，再用第二棵树拟合残差，把两棵树的预测按学习率加起来。只做 3 轮，验证整体均方误差是否在一轮轮地下降——你就亲手实现了一遍"亡羊补牢"。

## 小结

一句话记住这一节：**集成的精髓是"造出一群会犯不同错误的弱模型，再让它们合作"**——Bagging/随机森林靠数据和特征的随机来降方差、求稳，Boosting/梯度提升靠一棵接一棵地补错来降偏差、求准。表格题里先上随机森林拿基线，再用梯度提升冲上限，是一套极其顺手的组合拳。

想把直觉建得更牢，强烈推荐读一读 scikit-learn 官方的 [Ensemble methods 文档](https://scikit-learn.org/stable/modules/ensemble.html)，它把每种集成的用法和参数都讲得很清楚；如果想往竞赛级再进一步，[XGBoost 官方教程](https://xgboost.readthedocs.io/en/stable/tutorials/model.html) 对梯度提升的数学推导有更细致的展开。
