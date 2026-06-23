# 1.4 分类评估指标

> **难度** ⭐⭐⭐☆☆ · **前置**：[1.3 逻辑回归与分类](2-1-3-logistic-regression.md)

!!! abstract "读完这一节，你会"
    - 看懂混淆矩阵，并能从它手算准确率、精确率、召回率、F1 这四个指标
    - 说清楚为什么"准确率高"在类别不平衡时会骗人，并知道该换哪个指标
    - 理解 ROC 曲线和 AUC 在衡量什么，会用 `sklearn` 一行算出来
    - 拿到一个分类任务，能根据"漏检贵还是误报贵"选对评估指标

## 先想清楚：为什么"算对了多少"还不够

训练完一个分类器，我们第一反应都是问："它准不准？"于是很自然地去算一个数：预测对的样本占了多少比例。这个数叫**准确率（accuracy）**，它确实是最直观的指标。

可问题是，准确率有时会把人骗得很惨。举个例子：假设你做一个癌症筛查模型，一千个人里真正得病的只有 5 个。这时只要模型偷懒地把**所有人**都判成"健康"，它的准确率就高达 99.5%——可它一个病人都没查出来，作为筛查工具完全是废的。

你看，问题出在哪？出在我们把"判对"笼统地算成了一个数，却没有区分**两种完全不同的错误**：把病人误判成健康（漏诊），和把健康人误判成病人（虚惊一场）。这两种错误的代价天差地别，但准确率把它们一锅烩了。所以这一节真正要教你的，是一套能把这些错误**分门别类看清楚**的工具。

## 一切的起点：混淆矩阵

要把错误看清楚，先得把"对和错"拆成四格。我们以二分类为例，约定其中一类叫**正类（positive）**（比如"得病"），另一类叫**负类（negative）**（"健康"）。把"真实是什么"和"模型预测是什么"两两组合，正好有四种情况，画成一张 2×2 的表，这张表就叫**混淆矩阵（confusion matrix）**：

<div class="diagram">
<svg viewBox="0 0 380 230" font-family="-apple-system, 'Segoe UI', sans-serif">
  <text x="190" y="20" font-size="13" fill="var(--dia-stroke-soft)" text-anchor="middle">模型预测</text>
  <text x="120" y="44" font-size="12" fill="var(--dia-stroke)" text-anchor="middle">预测正类</text>
  <text x="270" y="44" font-size="12" fill="var(--dia-stroke)" text-anchor="middle">预测负类</text>
  <text x="22" y="125" font-size="13" fill="var(--dia-stroke-soft)" text-anchor="middle" transform="rotate(-90 22 125)">真实标签</text>
  <text x="52" y="95" font-size="12" fill="var(--dia-stroke)" text-anchor="middle">真为正</text>
  <text x="52" y="175" font-size="12" fill="var(--dia-stroke)" text-anchor="middle">真为负</text>
  <rect x="80" y="60" width="120" height="70" fill="var(--dia-green-soft)" stroke="var(--dia-stroke)" stroke-width="1.2"/>
  <rect x="200" y="60" width="120" height="70" fill="var(--dia-accent-soft)" stroke="var(--dia-stroke)" stroke-width="1.2"/>
  <rect x="80" y="130" width="120" height="70" fill="var(--dia-accent-soft)" stroke="var(--dia-stroke)" stroke-width="1.2"/>
  <rect x="200" y="130" width="120" height="70" fill="var(--dia-green-soft)" stroke="var(--dia-stroke)" stroke-width="1.2"/>
  <text x="140" y="90" font-size="13" fill="var(--dia-stroke)" text-anchor="middle" font-weight="600">TP</text>
  <text x="140" y="110" font-size="11" fill="var(--dia-stroke-soft)" text-anchor="middle">真正例</text>
  <text x="260" y="90" font-size="13" fill="var(--dia-stroke)" text-anchor="middle" font-weight="600">FN</text>
  <text x="260" y="110" font-size="11" fill="var(--dia-stroke-soft)" text-anchor="middle">漏检</text>
  <text x="140" y="160" font-size="13" fill="var(--dia-stroke)" text-anchor="middle" font-weight="600">FP</text>
  <text x="140" y="180" font-size="11" fill="var(--dia-stroke-soft)" text-anchor="middle">误报</text>
  <text x="260" y="160" font-size="13" fill="var(--dia-stroke)" text-anchor="middle" font-weight="600">TN</text>
  <text x="260" y="180" font-size="11" fill="var(--dia-stroke-soft)" text-anchor="middle">真负例</text>
</svg>
</div>
<p class="figure-caption">图 1.4-1：混淆矩阵的四格。绿色对角线是判对了（TP、TN），橙色两格是两种不同的错误（FN 漏检、FP 误报）。</p>

这四个格子的名字看着唬人，其实记法很简单——**第一个字母看"判对没"（T 对、F 错），第二个字母看"模型说的是正还是负"（P 正、N 负）**：

- **TP（真正例）**：真的是正类，模型也说正类，判对了。
- **TN（真负例）**：真的是负类，模型也说负类，判对了。
- **FP（假正例 / 误报）**：本来是负类，模型却喊正类——虚惊一场。
- **FN（假负例 / 漏检）**：本来是正类，模型却说负类——漏掉了。

记住这张表是关键，因为**接下来所有的指标，都只是用这四个数算出来的不同比值**。

## 四个核心指标：准确率、精确率、召回率、F1

有了四个格子，我们就能定义指标了。我们一个个来，每个都先说它"回答什么问题"，再给公式。

**准确率（accuracy）**回答的是"总共判对了几成"：

$$\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}.$$

它把对角线上判对的，除以全部样本。简单，但前面说过，类别不平衡时它会骗人，所以我们需要下面两个更细的指标。

**精确率（precision）**只盯着模型喊"正类"的那些预测，问："你喊了这么多正类，里头有几个是真的？"

$$\text{Precision}=\frac{TP}{TP+FP}.$$

分母是"所有被判成正类的"。精确率低，意味着误报（FP）多——模型一惊一乍，喊狼来了喊太多次。

**召回率（recall）**换个角度，盯着所有真正的正类，问："真正的正类一共这么多，你揪出来了几个？"

$$\text{Recall}=\frac{TP}{TP+FN}.$$

分母是"所有真实的正类"。召回率低，意味着漏检（FN）多——病人从你眼皮底下溜走了。

你大概已经发现了，**精确率和召回率往往是一对冤家**。模型判得越激进（动不动就喊正类），召回率上去了，但误报也多了，精确率就掉；判得越保守，精确率上去了，却容易漏检，召回率就掉。怎么办？我们想要一个数能同时照顾两边，于是有了 **F1 分数**——它取精确率和召回率的**调和平均**：

$$F_1=2\cdot\frac{\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}.$$

为什么用调和平均而不是普通平均？因为调和平均"偏袒小的那个"：只要精确率和召回率有一个很低，F1 就会被拉得很低。换句话说，**F1 高，必须两者都不差**，这正是我们想要的"平衡"。

## 一个挪动门槛就翻盘的例子

光看公式不踏实，我们来动手验证一遍。还记得 [1.3 逻辑回归](2-1-3-logistic-regression.md) 里，分类器其实先输出一个 0 到 1 的概率，再拿它跟一个**门槛（threshold）**比，超过就判正类。这个门槛默认是 0.5，但它是可以调的——而调它，会直接改变上面四个指标。先用 NumPy 从零把指标算出来，看清每个数怎么来：

```python
import numpy as np

# y_true: 真实标签（1=正类）；y_prob: 模型输出的"是正类"的概率
y_true = np.array([1, 1, 1, 0, 0, 0, 0, 1])
y_prob = np.array([0.9, 0.8, 0.4, 0.7, 0.3, 0.2, 0.1, 0.6])

def metrics(y_true, y_pred):
    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    acc  = (TP + TN) / len(y_true)
    prec = TP / (TP + FP) if TP + FP else 0      # 防止分母为 0
    rec  = TP / (TP + FN) if TP + FN else 0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0
    return acc, prec, rec, f1

y_pred = (y_prob >= 0.5).astype(int)             # 默认门槛 0.5
print("门槛 0.5:", [round(m, 2) for m in metrics(y_true, y_pred)])
```

你会看到输出 `门槛 0.5: [0.75, 0.75, 0.75, 0.75]`——四个指标在这份小数据上正好都等于 0.75。现在把门槛提高到 0.65，再算一次：

```python
y_pred = (y_prob >= 0.65).astype(int)            # 门槛调高，判正类更保守
print("门槛 0.65:", [round(m, 2) for m in metrics(y_true, y_pred)])
```

这次你会看到 `门槛 0.65: [0.62, 0.67, 0.5, 0.57]`：门槛一抬高，召回率立刻从 0.75 掉到了 0.5——因为有一个概率只有 0.6 的真正类被卡在了门外，漏检了。**这说明门槛不是固定不动的旋钮，而是你在"宁可错杀还是宁可放过"之间的权衡杆**——筛查癌症你会把门槛调低保住召回率，过滤垃圾邮件你会把门槛调高保住精确率，免得误删正常邮件。（顺带一提，这份数据里有个 0.7 的误报一直没被滤掉，所以这次精确率没怎么涨——真实数据上提高门槛通常会让精确率上升，但能不能涨终究取决于那些被你挡在门外的，到底是真正类还是误报。）

实际比赛里没人这么手写。`sklearn` 把这一切打包好了，结果应当和我们手算的对得上：

```python
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score,
                             classification_report)

y_pred = (y_prob >= 0.5).astype(int)
print(confusion_matrix(y_true, y_pred))          # 直接打出 2x2 混淆矩阵
print(accuracy_score(y_true, y_pred),
      precision_score(y_true, y_pred),
      recall_score(y_true, y_pred),
      f1_score(y_true, y_pred))
print(classification_report(y_true, y_pred))     # 一张表汇总所有指标
```

`classification_report` 是考场上的好帮手——它一次性把每个类别的精确率、召回率、F1 都列出来，省得你一个个调函数。

## 当类别不平衡时，到底该信哪个指标

现在回到开头那个 99.5% 准确率的癌症模型。我们用上面的工具重新审它：它把所有人判成健康，于是 TP=0，召回率 $0/(0+5)=0$。**召回率立刻揭穿了它**——准确率说它很棒，召回率说它一个病人都没找到。这就是为什么，**在正类稀少又重要的场景里，准确率基本没参考价值，要看精确率、召回率和 F1。**

那精确率和召回率之间又怎么选？记住一个判断标准——**看哪种错误更"贵"**：

- **漏检（FN）更贵**就优先保**召回率**。比如疾病筛查、欺诈检测：漏掉一个真病人/真欺诈，后果严重，宁可多几个误报也要把正类尽量捞全。
- **误报（FP）更贵**就优先保**精确率**。比如把正常邮件误判成垃圾邮件、给用户推一个明显不相关的内容：误报会直接伤害体验，宁可漏掉几个也要保证报出来的都靠谱。
- **两者都重要、又说不清谁更贵**，就看 **F1**，让模型在两边都别太差。

除此之外还有个实用习惯：类别不平衡时，**别用准确率挑模型，用 F1 或下面要讲的 AUC**。很多同学在 Round 1 的表格题里栽过跟头，就是因为盯着准确率调参，结果模型其实什么都没学会。

## ROC 曲线与 AUC：不挑门槛地评一个模型

前面我们发现，换个门槛，精确率召回率就全变了。这就带来一个尴尬：**如果两个模型用的门槛不一样，怎么公平地比它们谁更强？**

办法是：干脆把所有可能的门槛都试一遍，把模型在每个门槛下的表现连成一条曲线。这条曲线就是 **ROC 曲线（受试者工作特征曲线）**。它的横轴是**假正率**（误报占所有负类的比例，$FP/(FP+TN)$），纵轴是**真正率**（也就是召回率，$TP/(TP+FN)$）。

怎么读这条线？想象门槛从高往低慢慢放松：一开始很严，几乎不喊正类，曲线在左下角；逐渐放松，真正例和假正例都开始增多，曲线往右上爬。**一个好模型，应该在抓到很多真正例的同时尽量少误报**，所以它的曲线会尽量贴着左上角鼓起来。

我们用曲线**下方的面积**来给它打一个分，这个面积叫 **AUC（Area Under the Curve）**：

- AUC = 1.0：完美分类器，存在某个门槛能把两类完全分开。
- AUC = 0.5：跟瞎猜（抛硬币）一样，曲线就是那条对角线。
- AUC 越接近 1，说明**不管门槛怎么定**，模型把正类排在负类前面的能力越强。

AUC 最大的好处，就是它**不依赖某个具体门槛，也不受类别比例影响**，所以特别适合在不平衡数据上横向比较模型。算它同样是一行：

```python
from sklearn.metrics import roc_auc_score, roc_curve

auc = roc_auc_score(y_true, y_prob)              # 注意：传的是概率，不是 0/1 预测！
print("AUC =", round(auc, 3))

# 想画出整条曲线，roc_curve 会返回各门槛下的 (假正率, 真正率)
fpr, tpr, thresholds = roc_curve(y_true, y_prob)
```

这里有个**最容易错的点**：`roc_auc_score` 要喂**概率** `y_prob`，不是已经卡过门槛的 0/1 标签。喂错了，算出来的数就没意义。至于怎么把 `fpr, tpr` 画成一条好看的曲线，留到 [4.4 数据可视化](../ch1-intro/1-4-4-visualization.md) 里专门讲。

## 容易踩的坑

- **类别不平衡还盯着准确率**：这是头号陷阱。99% 的准确率可能只是模型把所有样本判成了多数类。一旦数据不平衡，第一时间换 F1 或 AUC。
- **算 AUC 时传错了输入**：`roc_auc_score(y_true, y_prob)` 要传**概率/打分**，传成 0/1 预测会得到一个错误的小数字。
- **精确率召回率分不清谁的分母是谁**：记口诀——精确率分母是"模型喊的正类"，召回率分母是"真实的正类"。想不起来就回去看混淆矩阵那张图。
- **分母为零没处理**：如果模型一个正类都没判（TP+FP=0），精确率会除以零。从零实现时一定要像上面代码那样加个判断。
- **多分类直接套二分类公式**：类别超过两个时，要指定 `average='macro'`（各类平均）还是 `'weighted'`（按样本数加权），两者结论可能不同，别用默认值蒙混过去。

## 它在后面会怎么用到

这套评估工具不是学完就放下的，几乎贯穿后面所有内容：

- 训练任何分类器——[1.5 KNN 与决策树](2-1-5-knn-decision-trees.md)、[1.6 集成方法](2-1-6-ensembles.md)、[1.7 支持向量机 SVM](2-1-7-svm.md)——都要靠这些指标来判断好坏。
- [1.9 模型评估与选择](2-1-9-model-evaluation.md) 会把这些指标和交叉验证结合起来，教你**靠谱地**比较模型，而不是被一次划分的运气骗到。
- ROC 曲线、混淆矩阵这些图怎么画得清楚，见 [4.4 数据可视化](../ch1-intro/1-4-4-visualization.md)。
- 到了 Round 2 的 [3.1 目标检测](../ch3-round2/3-3-1-object-detection.md)，你还会见到这套思想的升级版（mAP），核心仍是精确率和召回率的权衡。

## 练习

??? note "基础练习"
    1. 一个垃圾邮件分类器的混淆矩阵是 TP=40、FP=10、FN=20、TN=930。请手算它的准确率、精确率、召回率和 F1，并判断：这个数据是不是类别不平衡的？此时准确率可信吗？
    2. 还是上面这个例子，如果产品经理说"误删正常邮件最让用户恼火"，那你应该优先提高精确率还是召回率？为什么？
    3. 用 `sklearn` 的 `classification_report` 跑一遍上一题的预测，核对你手算的结果。

??? note "进阶练习"
    1. 不调用任何库，自己写一个函数：输入 `y_true` 和 `y_prob`，遍历一组门槛，画出（或打印出）每个门槛对应的 (假正率, 真正率)。再用 `sklearn.metrics.roc_curve` 验证你算的点是否一致。
    2. 造一份正负比例 1:99 的极度不平衡数据，训练一个"永远预测负类"的傻瓜模型，分别打印它的准确率和 AUC。亲眼看看为什么这种情况下准确率会骗人、而 AUC 不会。

## 小结

一句话记住这一节：**别只看准确率，先画混淆矩阵，再根据"漏检贵还是误报贵"，在精确率、召回率、F1 之间做选择；要不挑门槛地比模型，就看 AUC。**

- 混淆矩阵的四个格（TP/TN/FP/FN）是一切指标的源头。
- 精确率管"报出来的准不准"，召回率管"该报的漏没漏"，F1 让两者都别太差。
- 类别不平衡时准确率会骗人，改用 F1 或 AUC。
- AUC 衡量的是模型不依赖门槛的整体排序能力，1.0 完美、0.5 等于瞎猜。

想把这些指标的直觉建得更牢，推荐看 Google 的 [Machine Learning Crash Course — 分类指标章节](https://developers.google.com/machine-learning/crash-course/classification)，它用交互式的小例子把门槛、ROC 和 AUC 讲得很清楚。`sklearn` 官方的 [Metrics 文档](https://scikit-learn.org/stable/modules/model_evaluation.html) 则是你比赛时最该收藏的速查页。
