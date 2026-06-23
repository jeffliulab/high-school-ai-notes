# 1.1 GPU 训练

> **难度** ⭐⭐⭐☆☆ · **前置**：[2.4 PyTorch 基础](../ch2-round1/2-2-4-pytorch-basics.md)

!!! abstract "读完这一节，你会"
    - 说清楚 GPU 为什么比 CPU 适合训练神经网络，以及显存的几层结构各管什么
    - 用 `.to("cuda")` 把模型和数据搬上 GPU，配好 `DataLoader`，并开启混合精度训练
    - 学会定位训练慢的瓶颈到底卡在哪，并据此正确地缩放 batch size 和学习率

## 从 Round 1 到 Round 2：算力为什么突然重要了

在 Round 1，你跑的大多是表格数据上的经典机器学习，一台普通笔记本的 CPU 就够用了。可一进 Round 2，画风就变了：你要训练的是更深的网络、Transformer、做图像分割、甚至微调一个预训练大模型。这些模型动辄上百万、上千万个参数，要在成千上万张图片上反复迭代——**如果还用 CPU，一个实验可能要跑上一整天，比赛根本耗不起。**

所以从这一关开始，我们必须学会用 GPU（图形处理器，Graphics Processing Unit）来训练。好消息是，PyTorch 把搬上 GPU 这件事做得非常省心，往往只是几行代码的事；难的地方在于，**你得理解它为什么快、慢的时候卡在哪儿**，否则一旦显存爆了、或者 GPU 明明在却跑得不快，你会完全摸不着头脑。这一节就是带你把这层窗户纸捅破。

## GPU 凭什么比 CPU 快这么多

先回答"为什么"。CPU 和 GPU 的根本区别，不在于谁的主频高，而在于**它们是为完全不同的任务设计的**。

你可以这样打个比方：CPU 像几个博士生，每个人都很聪明、能处理复杂又多变的逻辑，但人数很少（通常几个到几十个核心）；GPU 则像几千个小学生，每个人只会做简单的加减乘，但人数极多。如果任务是"把一百万道同样的乘法题分下去做"，那显然是几千个小学生一起上更快——而神经网络的核心运算，正好就是这种"对海量数据做同样的矩阵乘法和加法"。

回忆一下我们在 [3.1 线性代数](../ch1-intro/1-3-1-linear-algebra.md) 里反复强调的那句话：**神经网络的每一层本质上都是矩阵乘法。** 一次前向传播，就是一连串 $\mathbf{y}=W\mathbf{x}+\mathbf{b}$。这种运算里，成千上万个乘加彼此独立、可以同时算——这正是 GPU 几千个核心最擅长的"大规模并行"。这就是为什么同样训练一个网络，GPU 常常比 CPU 快几十倍。

<div class="diagram">
<svg viewBox="0 0 420 210" font-family="-apple-system, 'Segoe UI', sans-serif">
  <defs>
    <marker id="ar" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="var(--dia-accent)"/></marker>
  </defs>
  <!-- CPU side -->
  <rect x="20" y="40" width="120" height="130" rx="6" fill="none" stroke="var(--dia-blue)" stroke-width="1.5"/>
  <text x="80" y="32" font-size="13" fill="var(--dia-blue)" text-anchor="middle">CPU · 少而强</text>
  <rect x="38" y="60" width="38" height="38" rx="4" fill="var(--dia-blue)" opacity="0.7"/>
  <rect x="84" y="60" width="38" height="38" rx="4" fill="var(--dia-blue)" opacity="0.7"/>
  <rect x="38" y="110" width="38" height="38" rx="4" fill="var(--dia-blue)" opacity="0.7"/>
  <rect x="84" y="110" width="38" height="38" rx="4" fill="var(--dia-blue)" opacity="0.7"/>
  <!-- GPU side -->
  <rect x="280" y="40" width="120" height="130" rx="6" fill="none" stroke="var(--dia-accent)" stroke-width="1.5"/>
  <text x="340" y="32" font-size="13" fill="var(--dia-accent)" text-anchor="middle">GPU · 多而简</text>
  <g fill="var(--dia-accent)" opacity="0.75">
    <rect x="294" y="54" width="13" height="13" rx="2"/><rect x="312" y="54" width="13" height="13" rx="2"/><rect x="330" y="54" width="13" height="13" rx="2"/><rect x="348" y="54" width="13" height="13" rx="2"/><rect x="366" y="54" width="13" height="13" rx="2"/>
    <rect x="294" y="72" width="13" height="13" rx="2"/><rect x="312" y="72" width="13" height="13" rx="2"/><rect x="330" y="72" width="13" height="13" rx="2"/><rect x="348" y="72" width="13" height="13" rx="2"/><rect x="366" y="72" width="13" height="13" rx="2"/>
    <rect x="294" y="90" width="13" height="13" rx="2"/><rect x="312" y="90" width="13" height="13" rx="2"/><rect x="330" y="90" width="13" height="13" rx="2"/><rect x="348" y="90" width="13" height="13" rx="2"/><rect x="366" y="90" width="13" height="13" rx="2"/>
    <rect x="294" y="108" width="13" height="13" rx="2"/><rect x="312" y="108" width="13" height="13" rx="2"/><rect x="330" y="108" width="13" height="13" rx="2"/><rect x="348" y="108" width="13" height="13" rx="2"/><rect x="366" y="108" width="13" height="13" rx="2"/>
    <rect x="294" y="126" width="13" height="13" rx="2"/><rect x="312" y="126" width="13" height="13" rx="2"/><rect x="330" y="126" width="13" height="13" rx="2"/><rect x="348" y="126" width="13" height="13" rx="2"/><rect x="366" y="126" width="13" height="13" rx="2"/>
  </g>
  <!-- data flow -->
  <line x1="145" y1="190" x2="275" y2="190" stroke="var(--dia-accent)" stroke-width="1.5" marker-end="url(#ar)"/>
  <text x="210" y="184" font-size="11" fill="var(--dia-stroke-soft)" text-anchor="middle">数据从内存搬到显存</text>
</svg>
</div>
<p class="figure-caption">图 1.1：CPU 是少数几个强核心，擅长复杂串行逻辑；GPU 是成千上万个简单核心，擅长把同一种运算并行铺开。训练前要先把数据从内存搬进显存。</p>

这里出现了一个关键角色：**显存**。

## 显存：训练时最容易"爆"的那块资源

GPU 自己带一块专用内存，叫**显存**（显卡内存，VRAM）。它和你电脑的普通内存（RAM）是两套独立的空间——这一点新手特别容易忽略。GPU 想算什么，数据必须**先从内存搬进显存**，算完再搬回来。这趟搬运是有代价的，待会儿讲瓶颈时你会看到，它常常是拖慢训练的元凶。

训练时，显存里大致装着这么几样东西，按占用从大到小你心里要有数：

- **模型参数**：网络的所有权重，参数越多占得越多；
- **梯度**：反向传播时每个参数对应一份梯度，和参数量同数量级；
- **优化器状态**：像 Adam 这种优化器，会为每个参数额外存一两份动量，所以它比朴素 SGD 更吃显存；
- **激活值（activations）**：前向传播时每一层的中间输出都要暂存下来，留着反向传播用——**它和 batch size 成正比，是训练中最容易膨胀的一块。**

把这四样加起来，就是你这次训练需要的显存。一旦超过显卡的物理上限，PyTorch 就会抛出那个让无数人头疼的报错：`CUDA out of memory`。理解了上面的构成，你就知道爆显存时该往哪几个方向减负了——这正是后面"缩放 batch"和"混合精度"要解决的问题。

## 把训练搬上 GPU：`.to("cuda")` 三件套

道理讲清楚了，我们动手。在 PyTorch 里，CUDA 是 NVIDIA 提供的、让代码调用 GPU 的底层接口；你不用直接碰它，PyTorch 已经替你封装好了。把训练搬上 GPU，核心就记住一句话：**模型要上 GPU，数据也要上同一个 GPU，两者必须在一起。**

下面这段是最标准的写法，我们一行行看：

```python
import torch

# 第一步：检测有没有可用的 GPU，没有就退回 CPU——这样代码在哪都能跑
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("训练设备:", device)

model = MyNet().to(device)        # 第二步：把模型的所有参数搬到 GPU

for x, y in dataloader:
    x = x.to(device)              # 第三步：每个 batch 的数据也搬到同一设备
    y = y.to(device)
    pred = model(x)               # 此时计算就在 GPU 上发生
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

这里最容易犯的错，就是**只搬了模型、忘了搬数据**（或者反过来）。一旦模型在 GPU、数据还在 CPU，PyTorch 会报 `Expected all tensors to be on the same device`。记住"模型在哪，数据就得跟到哪"，这个坑就能躲掉一大半。

## 让 DataLoader 不拖后腿

GPU 算得飞快，但它得有数据喂才行。如果数据准备得太慢，就会出现一种很憋屈的局面：**GPU 大部分时间在干等 CPU 把下一批数据读好、预处理好。** 这时候即使你有顶级显卡，也快不起来。

`DataLoader` 提供了两个关键参数来解决这个问题：

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,        # 用 4 个子进程并行地读取、预处理数据
    pin_memory=True,      # 把数据锁在固定内存，搬进显存时更快
)
```

`num_workers` 让多个进程在后台提前准备数据，这样 GPU 算完一批，下一批已经备好了，不用空等。`pin_memory=True` 则让"内存到显存"那趟搬运走得更快。这两个参数配合 `.to(device)`，就能让数据流尽量跟得上 GPU 的算力。

!!! tip "一个小习惯"
    搬数据时加上 `x.to(device, non_blocking=True)`，配合 `pin_memory=True`，搬运可以和计算重叠进行，进一步减少 GPU 的等待。

## 混合精度：用一半的位数，换近一倍的速度

现在讲一个在 Round 2 几乎必用的加速技巧——**混合精度训练**（mixed precision）。

它的动机很直接。默认情况下，PyTorch 用 32 位浮点数（float32）来存每个数。可对神经网络来说，很多计算其实没那么需要高精度，用 16 位浮点数（float16 / bfloat16）就够了。位数减半带来两个好处：**显存占用大约少一半**（于是你能塞下更大的模型或更大的 batch），而且新显卡上的专用单元（Tensor Core）算 16 位特别快，**训练能再快上不少**。

那精度不会出问题吗？这正是"混合"二字的精髓：**对乘法这类不敏感的运算用 16 位算得快，对求和、更新参数这类敏感的环节仍用 32 位保精度**，两者混着来。PyTorch 用 `autocast` 和 `GradScaler` 把这套逻辑自动管好了，你只需照着模板套：

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()             # 负责把梯度放大再缩回，避免 16 位下小梯度被舍成 0

for x, y in loader:
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()

    with autocast():              # 这个范围内的前向计算自动用 16 位
        pred = model(x)
        loss = loss_fn(pred, y)

    scaler.scale(loss).backward() # 放大后的损失再反向传播
    scaler.step(optimizer)        # 内部会自动把梯度缩回正常尺度再更新
    scaler.update()               # 动态调整放大倍数
```

跑起来你会发现，同样的模型，显存占用降了、每个 epoch 也快了，而精度几乎没有损失。这就是为什么但凡显卡支持，我们几乎总会顺手把混合精度开上。

## 训练慢了，到底卡在哪

学会搬上 GPU 还不够。实战里你经常会遇到"开了 GPU 却没快多少"的情况，这时候**别瞎调参，先定位瓶颈**。训练慢通常卡在三处之一：

1. **数据加载（CPU 端）卡脖子**：GPU 在等数据。症状是 `nvidia-smi` 里 GPU 利用率忽高忽低、长期偏低。对策是加大 `num_workers`、开 `pin_memory`、把重的预处理提前做好。
2. **GPU 算力本身打满**：利用率长期接近 100%，那说明 GPU 已经在拼命干了，想更快只能换更小的模型或更强的卡。
3. **显存不够 (`CUDA out of memory`)**：根本跑不起来。对策是减小 batch size、开混合精度、或用梯度累积。

定位的第一工具是命令行里的 `nvidia-smi`，它能实时告诉你 GPU 利用率和显存占用。在代码里，你也可以这样查显存：

```python
print(torch.cuda.memory_allocated() / 1e9, "GB 已用")   # 当前占用
print(torch.cuda.max_memory_allocated() / 1e9, "GB 峰值") # 历史峰值
```

记住一句口诀：**GPU 利用率低，是数据的锅；显存爆了，是 batch 太大的锅。** 对症下药，比盲目调参高效得多。

## batch 缩放：大 batch 不是免费的午餐

最后说 batch size 的缩放。很自然的想法是：显存够大，那把 batch 调到很大、一次喂更多数据，不就训练得又快又稳？这里有两个常被忽略的事实。

第一，**batch 越大越吃显存**——前面说过，激活值的占用和 batch size 成正比。所以放大 batch 的上限，是被显存卡死的。当显存不够又想要大 batch 的稳定效果时，可以用**梯度累积**：连着算几个小 batch 的梯度，先攒着不更新，攒够了再一次性更新，效果约等于一个大 batch，却不用一次性占那么多显存。

第二，**batch 变了，学习率通常也要跟着调**。一个广为流传的经验法则叫"线性缩放"：batch size 放大 $k$ 倍，学习率也大致放大 $k$ 倍。直觉上，大 batch 算出的梯度方向更准、噪声更小，所以可以放心迈大一点的步子。用公式写出来就是：

$$\eta_{\text{new}} = \eta_{\text{base}} \cdot \frac{B_{\text{new}}}{B_{\text{base}}}.$$

这只是个起点而非铁律——学习率怎么设、为什么影响这么大，回顾 [2.2 梯度下降与反向传播](../ch2-round1/2-2-2-backprop.md) 和 [2.5 优化与正则化](../ch2-round1/2-2-5-optimization-regularization.md) 会让你理解得更透。真正比赛里，你往往要在"显存放得下"和"学习率配得稳"之间做权衡。

## 容易踩的坑

- **模型和数据不在同一设备**：报 `Expected all tensors to be on the same device`。永远记住模型 `.to(device)` 了，每个 batch 也要 `.to(device)`。
- **拿 GPU 张量直接 `print` 或喂给 NumPy**：要先 `.cpu()` 搬回来，比如 `loss.item()` 或 `tensor.detach().cpu().numpy()`，否则报错或拖慢速度。
- **在循环里反复 `.to(device)` 搬模型**：模型只需搬一次，放在循环外；只有数据才每个 batch 搬。
- **忘了 `optimizer.zero_grad()`**：PyTorch 的梯度是累加的，不清零会让上一批的梯度残留进来，训练直接学歪。
- **`CUDA out of memory` 后不释放**：调试时显存可能没及时回收，可以 `torch.cuda.empty_cache()`，或干脆重启内核再试。
- **用计时判断快慢却忘了同步**：GPU 是异步执行的，计时前要 `torch.cuda.synchronize()`，否则你测到的时间是假的。

## 它在后面会怎么用到

GPU 训练是 Round 2 一切实验的地基，后面几乎每一节都默认你已经会用了：

- 微调一个预训练模型时，显存和混合精度是你天天打交道的东西（见 [1.3 模型微调](3-1-3-finetuning.md) 与 [4.4 预训练大模型](3-4-4-pretrained-llms.md)）。
- 训练 Transformer 时，长序列会让显存急剧膨胀，这一节的 batch 缩放和混合精度就是你的救命工具（见 [2.2 Transformer 架构](3-2-2-transformer.md)）。
- 做目标检测、图像分割这类大模型大图任务，瓶颈定位的功夫直接决定你一晚上能跑几个实验（见 [3.1 目标检测](3-3-1-object-detection.md)、[3.2 图像分割](3-3-2-segmentation.md)）。
- 到了集训营，如何把有限的算力榨出最多实验，是 [1.2 调参与实验管理](../ch4-camp/4-1-2-experiment-management.md) 的核心议题。

## 练习

??? note "基础练习"
    1. 写一段代码，先检测当前环境有没有 GPU，再分别在 CPU 和 GPU 上做一次 $4096\times4096$ 的矩阵乘法并计时（GPU 计时记得先 `torch.cuda.synchronize()`），对比两者快了多少倍。
    2. 把一个你在 Round 1 写过的小网络，完整地搬上 GPU 训练：模型、数据都 `.to(device)`，跑通一个 epoch，并用 `torch.cuda.max_memory_allocated()` 打印出峰值显存。

??? note "进阶练习"
    1. 给上面的训练循环加上混合精度（`autocast` + `GradScaler`），对比开启前后的显存占用和单 epoch 耗时，看看省了多少、快了多少。
    2. 故意把 batch size 调到爆显存，复现 `CUDA out of memory`；然后用"梯度累积"在不增大单次 batch 的前提下，达到等效的大 batch 训练效果，并把学习率按线性缩放法则相应调整。

## 小结

- **GPU 快在"大规模并行"**：神经网络的核心是海量独立的矩阵乘加，正好喂饱 GPU 几千个核心。
- **显存是稀缺资源**，由参数、梯度、优化器状态、激活值四部分组成；爆显存就从减小 batch、开混合精度入手。
- 搬上 GPU 三件套：检测 `device`、模型 `.to(device)`、每个 batch 数据 `.to(device)`，外加配好 `DataLoader`。
- **混合精度几乎是白送的加速**：省显存又提速，支持就开。
- 慢的时候先定位瓶颈：**GPU 利用率低是数据的锅，显存爆了是 batch 的锅。**

想系统看一遍，推荐 [PyTorch 官方的 CUDA 语义文档](https://pytorch.org/docs/stable/notes/cuda.html) 和 [自动混合精度教程](https://pytorch.org/docs/stable/amp.html)；想直观理解 GPU 为何如此并行，可看 NVIDIA 关于 CUDA 编程模型的入门介绍。
