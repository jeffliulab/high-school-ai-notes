# 1.3 逻辑回归与分类

> **难度** ⭐⭐⭐☆☆ · **前置**：[1.2 线性回归](2-1-2-linear-regression.md)、[3.3 概率与统计](../ch1-intro/1-3-3-probability-statistics.md)

!!! abstract "读完这一节，你会"
    - 明白为什么"线性回归 + sigmoid"就能做分类，并理解 sigmoid 的几何含义
    - 会写出二元交叉熵损失，并能手推它对参数的梯度
    - 能用 NumPy 从零实现逻辑回归，再用 sklearn 一行对照
    - 知道决策阈值怎么调，以及调它会带来什么后果

## 先搞清楚：分类和回归到底差在哪

上一节的线性回归，干的是预测一个**连续数值**的活：房价是 312 万、气温是 26.5 度。可现实里有一大类问题，答案不是一个数，而是一个**类别**：这封邮件是不是垃圾邮件？这张细胞图是良性还是恶性？这两个答案非黑即白，只有"是（1）"和"否（0）"两种，中间没有 0.7 封垃圾邮件这种说法。这种"判断属于哪一类"的任务，就叫**分类（classification）**。

那能不能偷个懒，直接拿线性回归来做分类？我们让模型输出一个数，大于 0.5 就判 1、小于就判 0——听起来好像行得通。但你很快会撞墙：线性回归的输出可以是任意实数，可能蹦到 8.3，也可能是 -5，把这种数字硬当成"概率"既不合理也不稳定。

我们真正想要的，是让模型输出一个**落在 0 到 1 之间的概率**：比如"这封邮件有 92% 的可能是垃圾邮件"。这样既能表达确信程度，又能自然地拿 0.5 当分界线。问题就变成了：怎么把线性回归那个可大可小的输出，**温柔地压进 (0, 1) 这个区间**？答案就是这一节的主角——sigmoid 函数。

## sigmoid：把任意实数压成一个概率

逻辑回归的做法其实很巧妙：它**先照搬线性回归**，算出一个我们熟悉的线性组合

$$z = \mathbf{w}^\top\mathbf{x} + b,$$

这里 $\mathbf{w}$ 是权重、$b$ 是偏置、$\mathbf{x}$ 是输入特征，跟线性回归一模一样。$z$ 可以是任意实数。接着，它把 $z$ 喂给一个叫 **sigmoid**（也叫 logistic 函数）的"压缩器"：

$$\sigma(z) = \frac{1}{1 + e^{-z}}.$$

别被公式吓到，看一眼它的图像就全明白了：

<div class="diagram">
<svg viewBox="0 0 380 220" font-family="-apple-system, 'Segoe UI', sans-serif">
  <!-- 坐标轴 -->
  <line x1="40" y1="185" x2="350" y2="185" stroke="var(--dia-stroke-tertiary)" stroke-width="1"/>
  <line x1="195" y1="30" x2="195" y2="195" stroke="var(--dia-stroke-tertiary)" stroke-width="1"/>
  <!-- y=1 与 y=0.5 的参考虚线 -->
  <line x1="40" y1="50" x2="350" y2="50" stroke="var(--dia-stroke-tertiary)" stroke-width="0.8" stroke-dasharray="3 3"/>
  <line x1="40" y1="117" x2="350" y2="117" stroke="var(--dia-stroke-tertiary)" stroke-width="0.8" stroke-dasharray="3 3"/>
  <!-- sigmoid 曲线 -->
  <path d="M44 183 C120 181 150 175 195 117 C240 59 270 53 346 51" fill="none" stroke="var(--dia-accent)" stroke-width="2.2"/>
  <!-- 中心点 (0, 0.5) -->
  <circle cx="195" cy="117" r="4" fill="var(--dia-accent-deep)"/>
  <!-- 文字标注：全部放在曲线不经过的空白处 -->
  <text x="46" y="44" font-size="12" fill="var(--dia-stroke-soft)">σ(z) → 1</text>
  <text x="300" y="178" font-size="12" fill="var(--dia-stroke-soft)">σ(z) → 0</text>
  <text x="202" y="111" font-size="12" fill="var(--dia-accent-deep)">(0, 0.5)</text>
  <text x="330" y="200" font-size="12" fill="var(--dia-stroke-soft)">z</text>
  <text x="58" y="135" font-size="11" fill="var(--dia-stroke-tertiary)">z 很负</text>
  <text x="300" y="70" font-size="11" fill="var(--dia-stroke-tertiary)">z 很正</text>
</svg>
</div>
<p class="figure-caption">图 1.3-1：sigmoid 把 (-∞, +∞) 的任意实数 z 平滑地压进 (0, 1)。z=0 时正好输出 0.5，z 越大越接近 1，越小越接近 0。</p>

你会发现这条曲线有几个讨喜的性质：它永远夹在 0 和 1 之间，正好可以当概率读；它在 $z=0$ 处穿过 0.5，左右对称；而且它是光滑的，处处可导——这一点对后面要用梯度下降来训练至关重要。

所以逻辑回归对一个样本的完整预测就是：先算 $z=\mathbf{w}^\top\mathbf{x}+b$，再过一遍 sigmoid，得到"它属于类别 1 的概率"

$$\hat{y} = \sigma(\mathbf{w}^\top\mathbf{x}+b) = P(y=1\mid\mathbf{x}).$$

顺带说一句：虽然名字里带"回归"，逻辑回归其实是个**分类**模型。这个别扭的名字是历史遗留，记住它做的是分类就行。

## 损失函数：为什么不用均方误差，而用交叉熵

模型有了，下一个问题是：怎么衡量它预测得好不好，好让我们去优化它？线性回归用的是均方误差，这里能照搬吗？理论上能，但实践中很糟——sigmoid 套上平方误差后，损失曲面会变得坑坑洼洼（非凸），梯度下降很容易卡住。

更合适的选择，是从"概率"的角度来设计损失。我们手上的真实标签 $y$ 要么是 1 要么是 0，而模型给出"是 1 的概率"$\hat{y}$。一个好模型应该做到：当真实是 1 时，$\hat{y}$ 尽量接近 1；当真实是 0 时，$\hat{y}$ 尽量接近 0。把这个愿望写成数学，对单个样本就是

$$\ell = -\big[\,y\log\hat{y} + (1-y)\log(1-\hat{y})\,\big].$$

这个式子叫**二元交叉熵（binary cross-entropy）**。它看着复杂，其实是个很聪明的"二选一开关"：

- 当真实标签 $y=1$ 时，后半项 $(1-y)\log(1-\hat y)$ 因为 $1-y=0$ 直接消失，只剩 $-\log\hat y$。$\hat y$ 越接近 1，$-\log\hat y$ 越接近 0（惩罚小）；$\hat y$ 越接近 0，它就冲向 $+\infty$（惩罚极重）。
- 当 $y=0$ 时反过来，只剩 $-\log(1-\hat y)$，逼着 $\hat y$ 往 0 靠。

换句话说，**模型越是自信地答错，被罚得越狠**。把所有 $n$ 个样本的损失平均起来，就是我们要最小化的总损失：

$$L(\mathbf{w}, b) = -\frac{1}{n}\sum_{i=1}^{n}\Big[y_i\log\hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\Big].$$

这个损失还有一个更深的来头：它其实就是**负对数似然（negative log-likelihood）**。我们在 [3.3 概率与统计](../ch1-intro/1-3-3-probability-statistics.md) 里讲过"最大似然估计"——找一组参数，让"观测到这批数据"的概率最大。把那个似然取对数、再取负号，得到的正好就是上面这个交叉熵。所以最小化交叉熵 = 最大化似然，两件事是一回事。

## 手推梯度：训练靠的就是它

要用梯度下降训练，我们就得知道损失 $L$ 对每个参数的梯度长什么样。这一步初看吓人，但推完你会发现结果出奇地干净。我们先看单个样本，再加和。

先准备一块关键拼图——sigmoid 的导数。它有一个非常漂亮的性质（你在 [3.2 微积分与梯度](../ch1-intro/1-3-2-calculus-gradients.md) 的进阶练习里推过）：

$$\sigma'(z) = \sigma(z)\,\big(1 - \sigma(z)\big) = \hat{y}(1-\hat{y}).$$

接下来用**链式法则**，把"损失对 $\mathbf{w}$ 的导数"拆成三段相乘：损失对 $\hat y$、$\hat y$ 对 $z$、$z$ 对 $\mathbf{w}$。

第一段，损失对 $\hat y$ 求导：

$$\frac{\partial \ell}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}} = \frac{\hat{y}-y}{\hat{y}(1-\hat{y})}.$$

第二段，$\hat y$ 对 $z$ 求导，正是上面那块拼图 $\hat{y}(1-\hat{y})$。把这两段一乘，分母里的 $\hat{y}(1-\hat{y})$ 被神奇地约掉了：

$$\frac{\partial \ell}{\partial z} = \frac{\hat{y}-y}{\hat{y}(1-\hat{y})}\cdot\hat{y}(1-\hat{y}) = \hat{y} - y.$$

这个结果干净得让人惊讶——**梯度对 $z$ 的部分，就是"预测减真实"这个误差**。第三段最简单，$z=\mathbf{w}^\top\mathbf{x}+b$ 对 $\mathbf w$ 的导数就是 $\mathbf x$、对 $b$ 的导数就是 1。三段合起来，再对所有样本取平均，得到：

$$\frac{\partial L}{\partial \mathbf{w}} = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)\,\mathbf{x}_i,\qquad
\frac{\partial L}{\partial b} = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i).$$

如果你做过上一节的线性回归，会觉得这两个式子眼熟得很——它们和线性回归的梯度**长得几乎一模一样**，唯一的区别只是 $\hat y$ 这里多套了一层 sigmoid。这不是巧合，而是交叉熵这个损失被"精心设计"出来的好处。记住这个形式，写代码时就有底了。

## 从零实现：NumPy 版逻辑回归

道理讲透了，我们动手把它写出来。下面这段代码先造一批二分类的假数据，再用上面推出的梯度做梯度下降。每一块前面都有一句话告诉你它在干嘛。

先准备数据和两个基础函数——sigmoid 和损失：

```python
import numpy as np

rng = np.random.default_rng(0)
# 造两簇点：类别 0 围着 (-2,-2)，类别 1 围着 (2,2)，理论上线性可分
X0 = rng.normal([-2, -2], 1.0, size=(100, 2))
X1 = rng.normal([ 2,  2], 1.0, size=(100, 2))
X = np.vstack([X0, X1])                       # (200, 2)
y = np.concatenate([np.zeros(100), np.ones(100)])  # 前100个是0，后100个是1

def sigmoid(z):
    # 用 np.clip 防止 exp 溢出（z 太负时 e^-z 会爆）
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def bce_loss(y, y_hat):
    eps = 1e-9                                  # 防止 log(0) 得到 -inf
    return -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))
```

注意那两个小细节：`clip` 和 `eps` 都是为了挡住数值溢出，这是新手最容易栽跟头的地方，后面"容易踩的坑"还会专门提。接着是训练主循环，核心就是反复"算预测 → 算梯度 → 更新参数"：

```python
w = np.zeros(2)        # 权重，初始全 0
b = 0.0                # 偏置
lr = 0.1               # 学习率
for step in range(2000):
    z = X @ w + b
    y_hat = sigmoid(z)                          # 预测概率
    error = y_hat - y                           # 关键！梯度的核心就是这个误差
    grad_w = X.T @ error / len(y)               # ∂L/∂w
    grad_b = error.mean()                       # ∂L/∂b
    w -= lr * grad_w                            # 沿负梯度走一步
    b -= lr * grad_b
    if step % 500 == 0:
        print(f"step {step:4d}  loss={bce_loss(y, y_hat):.4f}")

# 预测：概率 ≥ 0.5 判为类别 1
pred = (sigmoid(X @ w + b) >= 0.5).astype(int)
print("准确率:", (pred == y).mean())
```

跑起来你会看到 loss 一路下降，准确率接近 1.0。最该留意的是 `error = y_hat - y` 这一行——它就是我们刚才费劲推出来的那个"预测减真实"，整个训练的引擎全在这里。

## 库版对照：sklearn 一行搞定

从零写一遍是为了**真懂**，但真到了竞赛或项目里，我们当然不会每次都手搓。scikit-learn 把上面这一整套封装进了 `LogisticRegression`，调用起来就是经典的"建模 → fit → predict"三连：

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X, y)                          # 训练

print(clf.predict(X[:5]))              # 直接给出 0/1 标签
print(clf.predict_proba(X[:5])[:, 1])  # 给出"属于类别1"的概率
print("准确率:", clf.score(X, y))
```

注意 `predict_proba` 返回的就是 sigmoid 算出的概率，这在下一节调阈值时会用到。两个版本结果应该高度一致——这正是"从零实现 + 库版对照"的意义：你既看清了引擎盖底下的机械，又掌握了实战里真正会用的那一行。

## 决策边界：模型在数据上画的那条线

训练完，模型到底学到了什么？它学到的是一条**决策边界（decision boundary）**——一条把两类样本分开的分界线。在逻辑回归里，这条线就出现在"概率正好等于 0.5"的地方，而 $\sigma(z)=0.5$ 当且仅当 $z=0$，所以决策边界就是那个干净的线性方程

$$\mathbf{w}^\top\mathbf{x} + b = 0.$$

这正是为什么逻辑回归被归为**线性分类器**：它的分界线（在二维里是直线、高维里是超平面）永远是直的。下面这张图把它画了出来：

<div class="diagram">
<svg viewBox="0 0 360 240" font-family="-apple-system, 'Segoe UI', sans-serif">
  <!-- 坐标框 -->
  <rect x="40" y="20" width="280" height="190" fill="none" stroke="var(--dia-stroke-tertiary)" stroke-width="1"/>
  <!-- 决策边界：一条斜线 wx+b=0 -->
  <line x1="60" y1="200" x2="300" y2="40" stroke="var(--dia-stroke)" stroke-width="2"/>
  <!-- 类别 0 的点（左上一簇） -->
  <circle cx="85" cy="55" r="5" fill="var(--dia-blue)"/>
  <circle cx="110" cy="80" r="5" fill="var(--dia-blue)"/>
  <circle cx="75" cy="95" r="5" fill="var(--dia-blue)"/>
  <circle cx="130" cy="58" r="5" fill="var(--dia-blue)"/>
  <circle cx="100" cy="115" r="5" fill="var(--dia-blue)"/>
  <!-- 类别 1 的点（右下一簇） -->
  <circle cx="240" cy="160" r="5" fill="var(--dia-accent)"/>
  <circle cx="265" cy="135" r="5" fill="var(--dia-accent)"/>
  <circle cx="225" cy="185" r="5" fill="var(--dia-accent)"/>
  <circle cx="280" cy="170" r="5" fill="var(--dia-accent)"/>
  <circle cx="255" cy="190" r="5" fill="var(--dia-accent)"/>
  <!-- 标注：放在各自空白角落，不压线、不压点 -->
  <text x="52" y="180" font-size="12" fill="var(--dia-blue)">类别 0</text>
  <text x="250" y="115" font-size="12" fill="var(--dia-accent)">类别 1</text>
  <text x="150" y="225" font-size="12" fill="var(--dia-stroke-soft)">决策边界：wᵀx + b = 0（σ=0.5）</text>
</svg>
</div>
<p class="figure-caption">图 1.3-2：逻辑回归学到的是一条线性的决策边界。边界一侧概率大于 0.5（判 1），另一侧小于 0.5（判 0）。</p>

理解了这一点，你也就知道了逻辑回归的**软肋**：如果两类数据天生没法用一条直线分开（比如一类被另一类团团围住），它就无能为力。这种情况要靠后面更灵活的模型，比如 [1.5 KNN 与决策树](2-1-5-knn-decision-trees.md) 或加了核技巧的 [1.7 支持向量机 SVM](2-1-7-svm.md)。

## 阈值不是非 0.5 不可：怎么调、为什么调

前面我们一直默认"概率 ≥ 0.5 就判 1"。但 0.5 这个分界线（叫**决策阈值，threshold**）其实是可以、有时也必须挪动的。为什么？

设想一个癌症筛查模型。把一个真病人误判成"健康"（漏诊），后果可能是致命的；而把一个健康人误判成"疑似"（虚惊一场）再去复查，代价小得多。这两类错误的轻重显然不一样。这时我们就该把阈值**调低**，比如概率只要超过 0.3 就拉去复查——宁可多抓几个虚惊，也别放过真病人。

反过来，如果是"自动给用户发促销短信"这种场景，误打扰用户的成本不低，我们可能把阈值**调高**到 0.7，只对很有把握的对象出手。代码上，调阈值不需要重新训练，只要在概率上换个比较值：

```python
proba = clf.predict_proba(X)[:, 1]   # 拿到"属于类别1"的概率
pred_low  = (proba >= 0.3).astype(int)   # 阈值调低：更敢判 1，召回↑、误报↑
pred_high = (proba >= 0.7).astype(int)   # 阈值调高：更谨慎，精确↑、漏判↑
```

这里藏着一个永恒的**权衡（trade-off）**：阈值往下调，你会抓到更多真阳性，但也会冤枉更多人；往上调则相反。到底定在哪，取决于"这两类错误各有多痛"。怎么系统地量化、怎么画出 ROC 曲线一次看遍所有阈值，正是下一节 [1.4 分类评估指标](2-1-4-classification-metrics.md) 要讲的内容。

## 容易踩的坑

- **`log(0)` 和 `exp` 溢出**：当预测概率被算到正好 0 或 1 时，$\log$ 会得到 `-inf`，损失变 `nan`。务必像代码里那样给 $\log$ 加一个极小的 `eps`，并对 sigmoid 的输入做 `clip`。这是逻辑回归从零实现里最高频的报错。
- **忘了标准化特征**：逻辑回归对特征尺度敏感，如果各特征量级差很多（年龄 0–100、收入 0–100000），梯度下降会收敛得极慢甚至跑偏。训练前先做标准化（见 [4.3 Pandas 与数据处理](../ch1-intro/1-4-3-pandas-data.md)）。
- **把概率当成"真实把握"**：模型输出 0.9 不代表现实里真有 90% 把握，它只是 sigmoid 的数值。类别不平衡或模型没校准时，这个概率可能很不靠谱。
- **类别极度不平衡**：如果 99% 的样本是类别 0，模型可能学会"无脑全判 0"也有 99% 准确率。这时准确率会骗人，要么调阈值、要么给损失加权，评估也要换看精确率/召回率。
- **以为它能解决非线性问题**：逻辑回归的边界永远是直的。数据弯弯绕绕时它会一直欠拟合，别指望靠调参救回来——该换模型就换模型。

## 它在后面会怎么用到

逻辑回归是承上启下的一节，它把前面的线性模型和后面的深度学习串了起来：

- 它的两个核心零件——**sigmoid 激活**和**交叉熵损失**——会原封不动地出现在神经网络里。一个不带隐藏层的神经网络，本质上就是逻辑回归（见 [2.1 感知机与 MLP](2-2-1-perceptron-mlp.md)、[2.3 激活函数与损失函数](2-2-3-activations-losses.md)）。
- 这一节末尾提到的"阈值权衡"，会在 [1.4 分类评估指标](2-1-4-classification-metrics.md) 里展开成精确率、召回率、ROC/AUC 一整套评估工具。
- 多分类时，sigmoid 会推广成 **softmax**，交叉熵也随之推广，这是图像分类的标配（见 [3.2 图像分类实战](2-3-2-image-classification.md)）。

## 练习

??? note "基础练习"
    1. 把"从零实现"那段代码跑起来，画出训练过程中 loss 随步数下降的曲线。再把学习率改成 `1.0` 和 `0.001`，观察收敛速度和稳定性各有什么变化。
    2. 用 sklearn 在乳腺癌数据集（`from sklearn.datasets import load_breast_cancer`）上训练逻辑回归，对比阈值取 0.3、0.5、0.7 时，被判为"恶性"的样本数各是多少。

??? note "进阶练习"
    1. 不查资料，独立把损失对 $\mathbf{w}$ 和 $b$ 的梯度从头推一遍，并解释为什么分母里的 $\hat y(1-\hat y)$ 会被约掉。
    2. 给损失加上一个 L2 正则项 $\frac{\lambda}{2}\|\mathbf w\|^2$，重新推出梯度的变化，并在 NumPy 版里实现它。观察 $\lambda$ 变大时决策边界会怎么变（提示：sklearn 里这个强度由参数 `C` 控制，`C` 越小正则越强）。

## 小结

- 逻辑回归 = **线性组合 + sigmoid**，把任意实数压成 (0,1) 的概率，再拿阈值切成类别——名字带"回归"，干的却是分类。
- 它配的损失是**二元交叉熵**，本质是负对数似然；正因为这个搭配，梯度才化简成干净的"$(\hat y-y)\mathbf x$"，和线性回归如出一辙。
- 它的决策边界永远是**直线/超平面**，处理非线性问题会力不从心。
- **决策阈值可调**，调它是在两类错误之间做权衡，具体怎么衡量留到评估指标那一节。

想看更直观的可视化，推荐 sklearn 官方的 [Logistic Regression 文档](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)；想把概率视角彻底吃透，可读 [*Pattern Recognition and Machine Learning*](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/) 第 4.3 节。
