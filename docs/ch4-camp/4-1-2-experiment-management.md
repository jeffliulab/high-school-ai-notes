# 1.2 调参与实验管理

> **难度** ⭐⭐⭐☆☆ · **前置**：[1.9 模型评估与选择](../ch2-round1/2-1-9-model-evaluation.md)、[2.5 优化与正则化](../ch2-round1/2-2-5-optimization-regularization.md)

!!! abstract "读完这一节，你会"
    - 用日志和实验追踪工具（TensorBoard / Weights & Biases）把每次跑的结果记下来，再也不靠脑子记
    - 做消融实验、用学习率 finder 找到合适的学习率，并在有限时间里选对超参数搜索策略（网格 vs 随机 vs 贝叶斯）
    - 把一次实验做到"可复现"：固定随机种子、打开确定性开关，让结果能被重新跑出来

## 比赛拼到最后，拼的是"管得住自己的实验"

到了集训营这个阶段，模型该怎么搭、损失该怎么选，你其实已经懂得七七八八了。可一旦真坐进赛场，你会发现真正拖垮人的往往不是"不会做"，而是"乱"——你一个下午跑了二十多次实验，改了学习率、又改了 batch size、顺手还动了数据增强，结果分数忽高忽低，等你想回头复盘"到底是哪一下让分数涨上去的"，却怎么也想不起来了。

这就是为什么我们要单独花一节讲**实验管理（experiment management）**。它听起来不像个"技术"，更像个"习惯"，但它恰恰是把你和那些临场手忙脚乱的对手区分开的关键。一句话先记住：**没有记录的实验，等于没做过。** 你跑得再多，记不住、复现不出来，就提炼不出有用的结论，下一步也就无从下手。

这一节我们就把这套习惯拆开来讲：怎么把每次实验记下来，怎么设计实验去回答"到底是哪个改动起了作用"，怎么在有限的时间和算力里把超参数调到够好，以及怎么保证你今天跑出来的好成绩，明天还能一模一样地重现。

## 先把每一次实验都记下来：日志与追踪

最朴素的记录方式，就是在训练循环里把关键数字打印出来，或者写进一个文件。这一步无论如何都不能省——哪怕你什么花哨工具都不用，至少也要把每个 epoch 的训练损失、验证损失、验证指标记下来：

```python
import logging

# 配置一个最简单的日志：同时打到屏幕和文件
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[logging.FileHandler("run.log"), logging.StreamHandler()],
)

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader)
    val_loss, val_acc = evaluate(model, val_loader)
    # 把这一轮的结果落到日志里，事后能逐行翻看
    logging.info(f"epoch={epoch} train_loss={train_loss:.4f} "
                 f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
```

但纯文本日志有个毛病：数字一多就看不出趋势，你很难一眼判断"损失是在稳稳下降，还是已经开始过拟合反弹了"。所以更进一步，我们会用**可视化的追踪工具**，把这些数字实时画成曲线。最常用的两个，一个是 PyTorch 自带就能配合的 **TensorBoard**，另一个是云端的 **Weights & Biases（简称 wandb）**。

先看 TensorBoard，它的核心就是一个 `SummaryWriter`，你把标量喂给它，它替你画曲线：

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="runs/exp_lr3e-4")   # 每个实验一个独立文件夹

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader)
    val_loss, val_acc = evaluate(model, val_loader)
    writer.add_scalar("loss/train", train_loss, epoch)   # 训练损失曲线
    writer.add_scalar("loss/val",   val_loss,   epoch)   # 验证损失曲线
    writer.add_scalar("acc/val",    val_acc,    epoch)   # 验证准确率曲线

writer.close()
# 跑完后在终端执行 tensorboard --logdir runs，浏览器里就能看交互式曲线
```

wandb 的用法几乎一样，只是它把数据存到云端，还会自动把超参数、不同实验的曲线**叠在同一张图上对比**——这在你一口气跑了十几个配置、想横向比较时特别省心：

```python
import wandb

wandb.init(project="ioai-camp", config={"lr": 3e-4, "batch_size": 64})
for epoch in range(num_epochs):
    ...
    wandb.log({"loss/train": train_loss, "loss/val": val_loss, "acc/val": val_acc})
```

提醒一句：IOAI 这类闭卷上机的赛场，多半是**没有外网**的，wandb 这种依赖联网的工具不一定能用。所以平时练习可以用 wandb 享受方便，但你必须保证 TensorBoard 或纯文本日志这条"离线退路"也用得顺手，到了赛场才不至于抓瞎。

## 消融实验：到底是哪个改动起了作用

把实验记下来只是第一步，记下来是为了**比较**。这里要介绍一个很重要的实验思想，叫**消融实验（ablation study）**。名字听着唬人，其实道理朴素得很：你想知道某个零件到底有没有用，就把它"消"掉，看看少了它成绩会不会变差。

举个例子，你的模型现在用了三个技巧：数据增强、Dropout、学习率预热。整体验证准确率是 88%。这时千万别得意地以为"三个都很关键"——也许其中某个根本没用，甚至在帮倒忙。消融实验的做法是：**每次只去掉一个，其余原封不动**，分别跑一遍：

| 配置 | 验证准确率 |
| --- | --- |
| 完整模型（三个技巧都在） | 88.0% |
| 去掉数据增强 | 84.1% |
| 去掉 Dropout | 87.6% |
| 去掉学习率预热 | 87.9% |

这张表一摆出来，结论就很清楚了：数据增强贡献最大（去掉掉了近 4 个点），而 Dropout 和预热的作用其实很小。于是你就知道，时间该花在打磨数据增强上，而不是在那两个几乎没影响的开关上反复纠结。

这里有个最容易被忽视的纪律：**一次只动一个变量**。如果你同时改了两样东西，分数变了，你根本说不清是哪样的功劳——这和初中物理里的"控制变量法"是一模一样的道理。赛场上时间紧，越是着急，越要忍住"一把全改了"的冲动。

## 学习率怎么定：用 LR finder 替你试

在所有超参数里，**学习率（learning rate）**几乎是影响最大、也最玄学的一个。设大了，损失会在谷底两边来回横跳甚至直接发散；设小了，训练慢得像蜗牛，半天不收敛。那到底设多少？与其凭感觉瞎猜，不如用一个叫**学习率扫描（LR finder）**的小技巧让它自己"现形"。

它的思路很巧妙：从一个极小的学习率开始，每跑一个 batch 就把学习率乘大一点点，一路指数式地涨上去，同时把"当前学习率"和"当前损失"记下来。画出来你会看到一条很有规律的曲线——损失先是稳稳下降，到某个点开始触底，再往后就突然炸起来。**那个损失下降最快、还没开始上翘的位置，就是合适的学习率。**

<div class="diagram">
<svg viewBox="0 0 380 210" font-family="-apple-system, 'Segoe UI', sans-serif">
  <!-- 坐标轴 -->
  <line x1="44" y1="170" x2="350" y2="170" stroke="var(--dia-stroke-soft)" stroke-width="1.2"/>
  <line x1="44" y1="20" x2="44" y2="170" stroke="var(--dia-stroke-soft)" stroke-width="1.2"/>
  <!-- 损失曲线：先平、再陡降、触底、炸起 -->
  <path d="M60 56 Q120 60 165 78 Q215 100 250 138 Q272 158 290 96 Q305 50 330 28"
        fill="none" stroke="var(--dia-blue)" stroke-width="2.2"/>
  <!-- 最佳点标记 -->
  <circle cx="250" cy="138" r="5.5" fill="var(--dia-accent)"/>
  <line x1="250" y1="138" x2="250" y2="170" stroke="var(--dia-accent)" stroke-width="1" stroke-dasharray="3 3"/>
  <!-- 文字标注放左下空白处，用引线指回最佳点，避免被曲线遮挡 -->
  <line x1="205" y1="151" x2="245" y2="139" stroke="var(--dia-accent)" stroke-width="1" stroke-dasharray="3 3"/>
  <text x="100" y="155" font-size="11.5" fill="var(--dia-accent-deep)">下降最快处 → 选它</text>
  <text x="300" y="46" font-size="11.5" fill="var(--dia-stroke-soft)">损失炸起</text>
  <text x="355" y="174" text-anchor="end" font-size="11" fill="var(--dia-stroke-soft)">学习率（对数）→</text>
  <text x="36" y="30" text-anchor="end" font-size="11" fill="var(--dia-stroke-soft)">损失</text>
</svg>
</div>
<p class="figure-caption">图 1.2：LR finder 的典型曲线——挑损失下降最陡、还没上翘的那一点作为学习率。</p>

下面这段代码就是这个扫描过程的骨架，你可以直接套到自己的训练循环外面：

```python
import torch

def lr_finder(model, loader, optimizer, loss_fn,
              lr_start=1e-7, lr_end=1.0, num_iter=100):
    """指数式抬高学习率，记录每一步的损失，找出下降最快的那个 lr。"""
    gamma = (lr_end / lr_start) ** (1 / num_iter)   # 每步学习率乘的倍数
    lr = lr_start
    lrs, losses = [], []
    it = iter(loader)
    for i in range(num_iter):
        x, y = next(it)
        for g in optimizer.param_groups:            # 手动设置本步学习率
            g["lr"] = lr
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        lrs.append(lr)
        losses.append(loss.item())
        lr *= gamma                                  # 指数抬高
        if loss.item() > 4 * min(losses):            # 损失炸了就提前停
            break
    return lrs, losses
# 实践中 fastai、pytorch-lightning 都内置了 lr_find，竞赛里直接调库即可
```

跑完之后，把 `lrs` 和 `losses` 画成曲线，按图 1.2 的方法挑点就行。最后强调一句：**学习率不是孤立的，它和 batch size、优化器深度绑定**——batch size 翻倍，学习率通常也要相应调大。所以换了 batch size，别忘了重新跑一遍 finder。

## 算力和时间有限，搜索策略怎么选

学习率定好了，可你还有一堆别的超参数要调：权重衰减、Dropout 比例、网络层数……它们组合起来是个巨大的空间，而你在赛场上的算力和时间都是有限的。怎么搜，是个需要动脑子的策略问题。常见有三种打法，适用场景各不相同：

**网格搜索（grid search）**：把每个超参数的候选值列出来，所有组合挨个跑一遍。它的好处是彻底、不漏，坏处是组合数会爆炸——3 个参数各 5 个候选，就是 $5^3=125$ 次实验。**参数一多就跑不动了**，适合只调两三个、且每次训练很快的小任务。

**随机搜索（random search）**：不再规规矩矩遍历，而是在参数空间里随机撒点。听上去比网格"草率"，但有个反直觉的事实：当大多数参数其实没那么重要时，随机搜索能用同样的实验次数，把真正重要的那个参数试到更多不同取值，因此往往**比网格更高效**。预算紧张时，它常常是更聪明的默认选择。

**贝叶斯优化（Bayesian optimization）**：这是三者里最"聪明"的。它会根据已经跑过的实验结果，建一个"哪片区域可能更好"的概率模型，然后**有的放矢地挑下一个最值得试的点**，而不是瞎撞。所以在每次训练都很贵（比如要跑好几个小时）、总预算又很有限时，它能用最少的实验次数逼近最优。代价是它实现起来稍复杂，且实验之间是串行的（得等上一个跑完才知道下一个试哪），并行不如随机搜索方便。

一句话帮你抉择：**便宜又少，就网格；预算有限、参数偏多，就随机；每次都很贵、试一次心疼，就贝叶斯。** 下面用 `optuna` 这个库演示一下随机/贝叶斯搜索有多省事——你只需描述"参数在什么范围里取、目标是什么"，搜索策略交给它：

```python
import optuna

def objective(trial):
    # 告诉 optuna 每个超参数的搜索范围
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    model = build_model(dropout=dropout)
    val_acc = train_and_eval(model, lr=lr, weight_decay=wd)
    return val_acc            # 返回我们想最大化的指标

study = optuna.create_study(direction="maximize")   # 默认就是贝叶斯式（TPE）采样
study.optimize(objective, n_trials=30)              # 在预算内跑 30 次
print(study.best_params, study.best_value)          # 打印出最优配置
```

## 可复现：让今天的好成绩明天还能再现

最后这块特别容易被忽视，却在比赛里反复坑人。设想你某次跑出了 91% 的好成绩，兴奋地把代码原样再跑一遍想确认一下——结果这次只有 88%。问题出在哪？出在**随机性**。模型权重的初始化、数据加载的打乱顺序、Dropout 丢哪些神经元，背后都是随机数在做主。如果不把这些随机性"钉死"，你每次跑的其实都是一个略有不同的实验，自然复现不出来。

解决办法叫**固定随机种子（random seed）**：随机数生成器其实是"伪随机"的，给它一个固定的起点（种子），它每次吐出的随机序列就完全一样。所以我们在程序最开头，把所有用到随机的库的种子都设成同一个值：

```python
import random, numpy as np, torch

def set_seed(seed=42):
    random.seed(seed)                    # Python 自带的 random
    np.random.seed(seed)                 # NumPy 的随机
    torch.manual_seed(seed)              # PyTorch CPU 端
    torch.cuda.manual_seed_all(seed)     # PyTorch 所有 GPU
    # 让 cuDNN 走确定性算法：结果可复现，但可能略慢一点
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

这里有个要想清楚的取舍：打开 `cudnn.deterministic` 后，GPU 会改用结果固定的算法，速度可能稍慢；而 `cudnn.benchmark = True` 会让 cuDNN 自动挑最快的算法，跑得快但每次结果可能有细微差别。**调试和提交复现时求"稳"，就打开确定性；只图跑得快的探索阶段，可以容忍一点不确定。**

还要补一句容易踩的细节：如果你用了多进程数据加载（`DataLoader(num_workers>0)`），每个子进程也有自己的随机状态，需要额外通过 `worker_init_fn` 给它们各自播种，否则数据增强的随机性仍然会泄漏进来，让你白忙活。

## 容易踩的坑

- **一次改一堆东西**：同时动了学习率和数据增强，分数变了也说不清是谁的功劳。消融实验的铁律是一次只动一个变量。
- **只看训练损失**：训练损失一直降不代表模型在变好，它可能正在过拟合。永远盯着**验证集**指标来做判断和早停。
- **实验不命名、互相覆盖**：所有结果都写进同一个 `runs/` 文件夹，或模型都存成 `model.pth`，跑第二次就把第一次盖掉了。给每个实验起个带超参数的唯一名字（如 `exp_lr3e-4_bs64`）。
- **种子设了一半**：只 `torch.manual_seed` 却忘了 NumPy 和 `cudnn`，结果照样复现不出来。要么用上面那个 `set_seed` 一键全设。
- **数据泄漏混进调参**：拿验证集反复调超参数调到飞起，等于把验证集也"训练"进去了。最终评估一定要留一份从没碰过的测试集（见 [1.9 模型评估与选择](../ch2-round1/2-1-9-model-evaluation.md)）。
- **赛场没网才发现 wandb 用不了**：平时只练联网工具，闭卷赛场抓瞎。务必把 TensorBoard / 纯文本日志这条离线退路练熟。

## 它在后面会怎么用到

- 这一节的方法贯穿整个集训营：当你做完[1.1 误差分析与调试](4-1-1-error-analysis.md)、找到了模型的弱点，就要靠这里的消融和调参手段去逐一验证、对症下药。
- 在[1.3 限时解题策略](4-1-3-time-strategy.md)里，"有限预算下选什么搜索策略"会直接变成你的时间分配决策——是把宝贵的算力砸在网格细搜，还是先用随机搜索快速摸底。
- 调参绕不开学习率、权重衰减这些旋钮，它们的来龙去脉在[2.5 优化与正则化](../ch2-round1/2-2-5-optimization-regularization.md)里，建议对照着回看。
- 真正下场做[2.1 CV 类真题](4-2-1-cv-problems.md)、[2.2 NLP 类真题](4-2-2-nlp-problems.md)时，把每次提交都当成一次有记录、可复现的实验，是稳定拿分的底层习惯。

## 练习

??? note "基础练习"
    1. 给你常用的训练脚本加上 `SummaryWriter`，把训练损失、验证损失、验证准确率三条曲线都记进 TensorBoard，跑一遍并在浏览器里看出"从第几个 epoch 开始过拟合"。
    2. 写一个 `set_seed` 函数并在脚本开头调用，把同一份训练连跑两次，确认两次的验证准确率完全一致；再把 `cudnn.deterministic` 关掉，观察两次结果是否开始有细微差别。

??? note "进阶练习"
    1. 针对"数据增强、Dropout、学习率预热"这三个技巧设计一组消融实验：每次只去掉一个，列成一张表，得出"哪个贡献最大"的结论，并解释你的实验为什么是公平的（控制了哪些变量）。
    2. 用 `optuna` 给一个小模型搜索 `lr`、`weight_decay`、`dropout` 三个超参数，分别限定预算为 10 次和 30 次实验，对比两种预算下找到的最优验证分数，思考"再多给 20 次实验，值不值"。

## 小结

这一节的核心就一句话：**把实验当成可记录、可比较、可复现的科学过程来做，而不是凭感觉乱试一通。** 具体落到四件事——用 TensorBoard / wandb 把每次结果记下来；用消融实验一次只动一个变量、看清谁在起作用；用 LR finder 定学习率、按预算在网格 / 随机 / 贝叶斯之间选搜索策略；用固定种子和确定性开关把随机性钉死，保证结果能重现。

想深入，可以读读那篇经典论文 [*Random Search for Hyper-Parameter Optimization*](https://www.jmlr.org/papers/v13/bergstra12a.html)（讲清了随机为何常胜过网格），以及 [Weights & Biases 官方教程](https://docs.wandb.ai/)和 [Optuna 文档](https://optuna.org/)，都很适合边做边学。
