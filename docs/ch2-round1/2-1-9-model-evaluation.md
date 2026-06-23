# 1.9 模型评估与选择

> **难度** ⭐⭐⭐☆☆ · **前置**：[1.1 监督学习工作流](2-1-1-supervised-workflow.md)、[1.4 分类评估指标](2-1-4-classification-metrics.md)

!!! abstract "读完这一节，你会"
    - 说清"调超参数"到底在调什么，并用网格搜索和随机搜索两种方式找到一组好参数
    - 理解为什么普通交叉验证选出来的成绩会"偏乐观"，并用嵌套交叉验证给出诚实的估计
    - 会看学习曲线，判断模型现在是欠拟合还是过拟合，从而决定"加数据"还是"调正则"
    - 用 scikit-learn 把上面这套流程跑通，并避开"用测试集调参"这个致命错误

## 先分清两件事：训练参数和超参数

在动手之前，我们得先把一个常被混淆的概念掰开。模型里其实有**两种**"数字"。

第一种是模型**自己从数据里学出来的**，比如线性回归的权重 $w$、决策树每个节点上的分裂阈值。这些叫**参数（parameters）**，你不用管，训练过程会自动把它们调好。

第二种是**你在训练开始之前就得替模型定好的**，训练过程不会去碰它们。比如 KNN 里要看几个邻居（$k$）、岭回归里正则项的强度（$\lambda$）、决策树最大允许长多深。这些叫**超参数（hyperparameters）**。它们就像做菜前你定的火候和盐量——模型再聪明，也没法自己决定该放多少盐。

所以"模型选择"这件事，很大程度上就是在问：**这一堆超参数，到底取什么值，模型在没见过的数据上表现最好？** 这一节讲的全部方法，都是在回答这个问题，并且要回答得**诚实**——不能自己骗自己。

## 网格搜索和随机搜索：怎么把超参数试一遍

既然超参数得人来定，最朴素的想法就是：**多试几组，挑最好的那组**。问题只是"怎么试"。

最直接的办法叫**网格搜索（grid search）**。你给每个超参数列出几个候选值，然后把所有组合**穷举**一遍。比如 SVM 有两个超参数 $C$ 和 $\gamma$，你各给 5 个候选，那就是 $5\times 5=25$ 种组合，挨个训练、挨个打分，谁高用谁。它的好处是规整、不漏；坏处也很明显——**组合数会爆炸**。三个超参数各试 10 个值，就是 $10^3=1000$ 次训练，再多就跑不动了。

于是有了第二种办法：**随机搜索（random search）**。不再老老实实铺满整个网格，而是在每个超参数的取值范围里**随机采样**若干组，比如随机抽 50 组来试。听起来很草率，但有一个反直觉却被反复验证的事实：**当超参数里只有少数几个真正重要时，随机搜索往往比网格搜索更划算。**

为什么？看下面这张图就懂了。

<div class="diagram">
<svg viewBox="0 0 420 200" font-family="-apple-system, 'Segoe UI', sans-serif">
  <!-- 左：网格搜索 -->
  <text x="100" y="22" text-anchor="middle" font-size="13" fill="var(--dia-stroke-soft)">网格搜索</text>
  <rect x="30" y="34" width="140" height="140" fill="none" stroke="var(--dia-stroke-tertiary)" stroke-width="1"/>
  <!-- 9个网格点 3x3 -->
  <g fill="var(--dia-blue)">
    <circle cx="58" cy="62" r="4"/><circle cx="100" cy="62" r="4"/><circle cx="142" cy="62" r="4"/>
    <circle cx="58" cy="104" r="4"/><circle cx="100" cy="104" r="4"/><circle cx="142" cy="104" r="4"/>
    <circle cx="58" cy="146" r="4"/><circle cx="100" cy="146" r="4"/><circle cx="142" cy="146" r="4"/>
  </g>
  <!-- 顶部"重要超参数"投影：只有3个不同的x -->
  <text x="100" y="192" text-anchor="middle" font-size="11" fill="var(--dia-accent-deep)">重要参数只试到 3 个值</text>
  <!-- 右：随机搜索 -->
  <text x="320" y="22" text-anchor="middle" font-size="13" fill="var(--dia-stroke-soft)">随机搜索</text>
  <rect x="250" y="34" width="140" height="140" fill="none" stroke="var(--dia-stroke-tertiary)" stroke-width="1"/>
  <g fill="var(--dia-accent)">
    <circle cx="266" cy="58" r="4"/><circle cx="312" cy="72" r="4"/><circle cx="370" cy="50" r="4"/>
    <circle cx="288" cy="100" r="4"/><circle cx="346" cy="118" r="4"/><circle cx="262" cy="132" r="4"/>
    <circle cx="324" cy="150" r="4"/><circle cx="376" cy="138" r="4"/><circle cx="300" cy="160" r="4"/>
  </g>
  <text x="320" y="192" text-anchor="middle" font-size="11" fill="var(--dia-accent-deep)">同样 9 次，重要参数试到 9 个值</text>
</svg>
</div>
<p class="figure-caption">图 1.9-1：同样花 9 次训练，网格搜索在"真正重要"的那个参数（横轴）上只覆盖了 3 个不同取值，随机搜索却覆盖了 9 个——所以更可能撞上好值。</p>

横轴假设是那个真正重要的超参数，纵轴是个无关紧要的。网格搜索把预算浪费在了"纵轴的不同取值"上，横轴只试到 3 个点；随机搜索因为是乱撒的，横轴上几乎每个点都不一样，反而把重要的那一维探得更细。**所以预算紧张时，优先考虑随机搜索。**

## 一个致命错误：千万别用测试集调参

讲方法之前，必须先敲响这个警钟，因为它是竞赛和实际工作中最常见、也最隐蔽的失误。

你可能会想：把数据分成训练集和测试集，我在训练集上训练，然后**拿测试集去比较哪组超参数最好**，不就行了吗？听起来很合理，但这里藏着一个陷阱——**你一旦用测试集来选参数，测试集就"泄露"进了你的决策过程，它就不再"干净"了。** 此时它报出来的成绩会偏高，你以为模型有 95 分，真上线一跑可能只有 88 分。

正确的做法是引入第三方：把数据切成**训练集 / 验证集 / 测试集**三份。用训练集学参数，用**验证集**比较超参数、做模型选择，最后只在全部尘埃落定后，用**测试集**报一次、且只报一次最终成绩。测试集就像高考，你平时刷的是模拟卷（验证集），高考卷只能在最后亲手拆封一次——提前偷看就失去意义了。

## 交叉验证为什么还不够诚实

只切一刀分出验证集，有个现实问题：如果数据本来就不多，分给验证集的那几百条样本好不好，**很看运气**。换一种分法，选出来的"最优超参数"可能就变了。

为了让验证更稳，我们用**交叉验证（cross-validation）**：把训练数据均分成 $k$ 份（常见 $k=5$），轮流拿其中一份当验证、其余四份当训练，跑 $k$ 轮，把 $k$ 个分数平均。这样每条样本都当过验证，结果不再依赖某一次"手气"。

但这里有个更深的坑，很多人没意识到。当你用交叉验证去**挑选超参数**时，你已经"偷看"了这些数据来做决策——于是这个被挑出来的最优分数本身，又变得偏乐观了。**换句话说，"用来调参的那套交叉验证分数"不能直接当成模型的最终战绩。** 它和"用测试集调参"是同一个病根，只是更隐蔽。

这就引出了**嵌套交叉验证（nested cross-validation）**。它的思路是套两层循环：

- **内层循环**：在当前这份训练数据上做交叉验证，专门用来**挑超参数**。
- **外层循环**：把数据再切成几折，每一折轮流当"从未参与调参的测试折"，用来**诚实地评估**内层选出来的模型。

内层负责"选"，外层负责"评"，两者用的数据互不重叠，这样报出来的成绩才没被污染。它的代价是计算量翻倍（外层 5 折 × 内层 5 折 = 25 套搜索），所以一般只在数据少、又特别在意"成绩到底有多可信"时才动用它。

## 学习曲线：模型现在缺的是数据还是模型容量

调完参，我们常常还想知道一个更根本的问题：**模型现在表现不好，到底是因为它太笨（欠拟合），还是因为它把训练集背得太死（过拟合）？** 答案藏在**学习曲线（learning curve）**里。

学习曲线的做法很简单：不断增加用于训练的样本数量，每次都记下**训练集得分**和**验证集得分**，画成两条随样本量变化的曲线。看这两条线的走势和它们之间的缝隙，就能诊断病情。

<div class="diagram">
<svg viewBox="0 0 420 200" font-family="-apple-system, 'Segoe UI', sans-serif">
  <!-- 坐标轴 -->
  <line x1="50" y1="170" x2="390" y2="170" stroke="var(--dia-stroke-tertiary)" stroke-width="1"/>
  <line x1="50" y1="30" x2="50" y2="170" stroke="var(--dia-stroke-tertiary)" stroke-width="1"/>
  <text x="220" y="192" text-anchor="middle" font-size="11" fill="var(--dia-stroke-soft)">训练样本数 →</text>
  <text x="30" y="100" text-anchor="middle" font-size="11" fill="var(--dia-stroke-soft)" transform="rotate(-90 30 100)">得分 ↑</text>
  <!-- 训练分（高，缓降）蓝 -->
  <path d="M70 48 Q200 56 380 70" fill="none" stroke="var(--dia-blue)" stroke-width="2"/>
  <text x="384" y="64" font-size="11" fill="var(--dia-blue)">训练分</text>
  <!-- 验证分（低，上升）橙 -->
  <path d="M70 150 Q200 120 380 92" fill="none" stroke="var(--dia-accent)" stroke-width="2"/>
  <text x="384" y="98" font-size="11" fill="var(--dia-accent)">验证分</text>
  <!-- 缝隙标注 -->
  <line x1="300" y1="66" x2="300" y2="100" stroke="var(--dia-stroke-soft)" stroke-width="1" stroke-dasharray="3 2"/>
  <text x="306" y="86" font-size="10" fill="var(--dia-stroke-soft)">缝隙=过拟合程度</text>
</svg>
</div>
<p class="figure-caption">图 1.9-2：学习曲线。两线之间的缝隙大、且加数据还在缩小，说明过拟合，再喂数据有用；若两线早早贴在一起且都低，说明欠拟合，加数据没用，得换更强的模型。</p>

怎么读它？记两个典型情况就够：

- **两条线之间缝隙很大**（训练分高高在上、验证分低低在下），这是典型的**过拟合**——模型把训练集背下来了，但没学到通用规律。这时缝隙如果随样本增多还在缩小，说明**多搞点数据**就能改善。
- **两条线早早贴在一起、却都停在很低的位置**，这是**欠拟合**——模型本身太简单，连训练集都学不好。这时加数据没用，得换个更强的模型或加更多特征。

学习曲线最大的价值，就是帮你**别瞎忙**：缺数据的时候你跑去换模型，或者欠拟合的时候你拼命标数据，都是白费力气。

## 用代码把这套流程跑通

道理讲完，我们用 scikit-learn 把它串起来。下面以岭回归（Ridge）为例，它的超参数就是正则强度 $\alpha$（也就是前面说的 $\lambda$）——$\alpha$ 越大，模型被"压"得越平、越不容易过拟合，但太大又会欠拟合。我们要做的，正是替它选一个恰到好处的 $\alpha$。

先看网格搜索 + 交叉验证：

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=300, n_features=20, noise=15, random_state=0)

# 候选的正则强度，从弱到强跨几个数量级
param_grid = {"alpha": [0.01, 0.1, 1, 10, 100]}

grid = GridSearchCV(
    Ridge(), param_grid,
    cv=5,                 # 5 折交叉验证，结果更稳
    scoring="r2",
)
grid.fit(X, y)
print("最优 alpha：", grid.best_params_)         # 例如 {'alpha': 1}
print("交叉验证最优分：", grid.best_score_)
```

`GridSearchCV` 内部做的事，正是我们前面讲的：对每个候选 $\alpha$ 跑一遍 5 折交叉验证、取平均分，最后留下平均分最高的那个。注意 `best_score_` 是**调参用的分数**，它偏乐观，**不能拿去当最终战绩对外汇报**。

如果候选很多、想省预算，把它换成随机搜索即可，接口几乎一样：

```python
from scipy.stats import loguniform

param_dist = {"alpha": loguniform(1e-3, 1e3)}    # 在对数尺度上连续采样
rand = RandomizedSearchCV(
    Ridge(), param_dist,
    n_iter=20,            # 只随机试 20 组，不穷举
    cv=5, scoring="r2", random_state=0,
)
rand.fit(X, y)
print("随机搜索最优 alpha：", rand.best_params_)
```

注意正则强度这种"跨好几个数量级"的超参数，采样要用**对数尺度**（`loguniform`），这样 0.001 和 0.01 之间、与 100 和 1000 之间被照顾得一样多，而不是把样本全浪费在大数那一头。

最后，如果你想要一个**诚实的**最终成绩，就把上面的搜索整个塞进外层交叉验证里——这就是嵌套交叉验证：

```python
from sklearn.model_selection import cross_val_score

inner = GridSearchCV(Ridge(), param_grid, cv=5, scoring="r2")  # 内层：选 alpha
# 外层：拿没参与选 alpha 的折来评分
nested_scores = cross_val_score(inner, X, y, cv=5, scoring="r2")
print("嵌套 CV 诚实估计：%.3f ± %.3f" % (nested_scores.mean(), nested_scores.std()))
```

你会发现这个"嵌套"分数通常比前面的 `best_score_` 略低——**这才是模型真实水平的诚实估计**，那点差距正是"调参偷看"带来的乐观偏差。

想亲手画学习曲线也只要一行核心调用：

```python
from sklearn.model_selection import learning_curve

sizes, train_sc, val_sc = learning_curve(
    Ridge(alpha=grid.best_params_["alpha"]),
    X, y, cv=5, scoring="r2",
    train_sizes=np.linspace(0.1, 1.0, 8),   # 用 10%~100% 的数据各跑一次
)
# train_sc.mean(1) 与 val_sc.mean(1) 就是图 1.9-2 里那两条线
```

把 `train_sc` 和 `val_sc` 沿样本量画出来，对照图 1.9-2 一读，就知道下一步该补数据还是换模型了。

## 容易踩的坑

- **用测试集调参**：这是头号大忌。测试集只能在最后揭晓一次，凡是参与了"选择"的数据都不再干净，报出的分数会虚高。
- **拿调参分当最终成绩**：`GridSearchCV.best_score_` 是用来挑参数的，它偏乐观。要诚实估计就用嵌套交叉验证的外层分数。
- **预处理泄露进验证集**：标准化、特征选择这些步骤，必须**只用训练折去 `fit`**，再套用到验证折上。正确做法是把它们和模型一起装进 `Pipeline`，让交叉验证每折单独处理，否则验证集的信息会偷偷漏进来。
- **正则强度用线性尺度搜索**：$\alpha$ 这类跨数量级的超参数要在对数尺度上采样，否则候选点全挤在大数那一端，小值区根本没探到。
- **看学习曲线只看一条线**：单看验证分低没用，得看**两条线的缝隙**才能区分欠拟合还是过拟合，进而决定"加数据"还是"换模型"。

## 它在后面会怎么用到

- 这一节挑超参数的纪律——尤其是"别用测试集调参"——是 [4.2 模拟与查漏](2-4-2-mock-review.md) 里反复要守住的底线。
- 进入深度学习后，超参数从一两个变成一大把（学习率、批大小、层数……），随机搜索的优势会更明显，相关取舍见 [2.5 优化与正则化](2-2-5-optimization-regularization.md)。
- 当你需要系统地记录"哪组超参数配哪个分数、用了哪份数据"时，就该上实验管理工具了，这是 Camp 阶段 [4.1.2 调参与实验管理](../ch4-camp/4-1-2-experiment-management.md) 的主题。
- 学习曲线之外的另一套诊断手段（按错误类型拆解模型到底错在哪）见 [4.1.1 误差分析与调试](../ch4-camp/4-1-1-error-analysis.md)。

## 练习

??? note "基础练习"
    1. 你有一个 SVM，超参数 $C$ 想试 `[0.1, 1, 10]`、$\gamma$ 想试 `[0.01, 0.1, 1]`。如果做网格搜索 + 5 折交叉验证，一共要训练多少次模型？换成只随机试 5 组呢？
    2. 用 `make_classification` 造一份二分类数据，对决策树的 `max_depth`（候选 `[2, 4, 6, 8, None]`）做一次 `GridSearchCV`，打印最优深度和对应的交叉验证分数。
    3. 一句话回答：为什么 `best_score_` 不能直接当成"这个模型上线后的预期成绩"？

??? note "进阶练习"
    1. 对同一份数据，分别用普通 5 折交叉验证的 `best_score_` 和嵌套交叉验证的外层分数报一个成绩，比较两者大小，并解释差距来自哪里。
    2. 故意构造一个欠拟合的场景（比如用 `max_depth=1` 的决策树去拟合一个复杂数据集），画出它的学习曲线，确认两条线"早早贴在一起且都很低"，并说明此时加数据为什么没用。
    3. 把标准化（`StandardScaler`）一次写进 `Pipeline`、一次手动在交叉验证外面先做，对比两种写法报出的分数，体会"预处理泄露"会让成绩虚高多少。

## 小结

一句话：**模型选择就是诚实地挑超参数——用网格或随机搜索去试，用交叉验证去比，用嵌套交叉验证或独立测试集去给出不掺水的成绩，再用学习曲线判断下一步该补数据还是换模型。** 守住"测试集只揭晓一次"这条纪律，你的成绩才靠得住。

想把这套流程的官方用法吃透，scikit-learn 的 [Model selection 用户指南](https://scikit-learn.org/stable/modules/cross_validation.html) 讲得非常细；随机搜索为何常胜网格搜索，可读 Bergstra & Bengio 的经典论文 [*Random Search for Hyper-Parameter Optimization*](https://www.jmlr.org/papers/v13/bergstra12a.html)。
