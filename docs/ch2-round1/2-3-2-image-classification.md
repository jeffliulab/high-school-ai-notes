# 3.2 图像分类实战

> **难度** ⭐⭐⭐☆☆ · **前置**：[3.1 卷积与池化](2-3-1-conv-pooling.md)、[2.4 PyTorch 基础](2-2-4-pytorch-basics.md)

!!! abstract "读完这一节，你会"
    - 说清一张图片从硬盘到模型之间要经过哪几步预处理（缩放、转张量、归一化），以及为什么少一步都不行
    - 用 PyTorch 亲手搭一个 2~3 个卷积块的小型 CNN，并讲得出每一层把张量的形状改成了什么
    - 写出一个完整的"训练一轮 + 验证一轮"循环，在 CIFAR-10 上把准确率从 10% 的瞎猜提到 70% 以上
    - 知道训练时最容易让你"白跑一晚上"的几个坑分别长什么样

## 从"会卷积"到"能认图"，中间还差一层窗户纸

上一节我们把卷积和池化这两块积木拆开看了个透：卷积负责"在图上滑动着找局部花纹"，池化负责"把图缩小、留住要紧的信息"。可光有积木还不算会盖房子。这一节，我们要把这些积木真的拼成一个能用的模型，喂给它几万张图片，让它自己学会区分猫、狗、飞机、卡车——这件事就叫**图像分类（image classification）**。

我们用的练手数据集叫 **CIFAR-10**，它是这个领域的"Hello World"：六万张 32×32 的小彩色图，平均分成 10 个类别。图很小，所以哪怕你只有一台没有显卡的笔记本，也能在可接受的时间里跑完——这正符合 Round 1 资格赛"CPU·基础"的定位。

这一节的主线非常清楚，就四步，我们会一步一步走完：

1. **预处理**：把杂乱的原始图片整理成模型能直接吃的张量；
2. **搭模型**：用卷积块堆出一个小型 CNN；
3. **训练**：写一个循环，让模型一轮一轮地从数据里学；
4. **评估**：在它没见过的数据上考一考，看它到底学没学会。

把这四步在脑子里串成一条流水线，你就掌握了几乎所有深度学习项目的骨架——后面换数据集、换模型，无非是在这条流水线上替换零件而已。

<div class="diagram">
<svg viewBox="0 0 460 130" font-family="-apple-system, 'Segoe UI', sans-serif">
  <defs>
    <marker id="pa" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="var(--dia-accent)"/></marker>
  </defs>
  <!-- 四个流程框 -->
  <rect x="14" y="46" width="86" height="38" rx="4" fill="var(--dia-bg-card)" stroke="var(--dia-stroke-soft)" stroke-width="1.5"/>
  <rect x="134" y="46" width="86" height="38" rx="4" fill="var(--dia-bg-card)" stroke="var(--dia-stroke-soft)" stroke-width="1.5"/>
  <rect x="254" y="46" width="86" height="38" rx="4" fill="var(--dia-bg-card)" stroke="var(--dia-stroke-soft)" stroke-width="1.5"/>
  <rect x="374" y="46" width="72" height="38" rx="4" fill="var(--dia-accent-soft)" stroke="var(--dia-accent)" stroke-width="1.5"/>
  <!-- 框内文字 -->
  <text x="57" y="62" text-anchor="middle" font-size="12" fill="var(--dia-stroke)">预处理</text>
  <text x="57" y="77" text-anchor="middle" font-size="10" fill="var(--dia-stroke-soft)">缩放·归一化</text>
  <text x="177" y="62" text-anchor="middle" font-size="12" fill="var(--dia-stroke)">小型 CNN</text>
  <text x="177" y="77" text-anchor="middle" font-size="10" fill="var(--dia-stroke-soft)">卷积块×3</text>
  <text x="297" y="62" text-anchor="middle" font-size="12" fill="var(--dia-stroke)">训练循环</text>
  <text x="297" y="77" text-anchor="middle" font-size="10" fill="var(--dia-stroke-soft)">前向·反向</text>
  <text x="410" y="62" text-anchor="middle" font-size="12" fill="var(--dia-accent)">评估</text>
  <text x="410" y="77" text-anchor="middle" font-size="10" fill="var(--dia-stroke-soft)">准确率</text>
  <!-- 箭头 -->
  <line x1="102" y1="65" x2="130" y2="65" stroke="var(--dia-accent)" stroke-width="1.5" marker-end="url(#pa)"/>
  <line x1="222" y1="65" x2="250" y2="65" stroke="var(--dia-accent)" stroke-width="1.5" marker-end="url(#pa)"/>
  <line x1="342" y1="65" x2="370" y2="65" stroke="var(--dia-accent)" stroke-width="1.5" marker-end="url(#pa)"/>
  <text x="230" y="26" text-anchor="middle" font-size="11" fill="var(--dia-stroke-tertiary)">一条贯穿所有深度学习项目的流水线</text>
</svg>
</div>
<p class="figure-caption">图 3.2-1：图像分类的四步流水线。换数据集、换模型，都只是在这条流水线上替换零件。</p>

## 第一步：把图片"喂得进去"——预处理

模型不认识 JPG、PNG，它只认数字张量。所以在训练之前，每张图都要被整理成统一的、规整的张量。这一步叫**预处理（preprocessing）**，它看起来琐碎，却是新手最容易翻车的地方——很多人模型搭得漂亮，结果准确率上不去，问题恰恰出在这里。

我们需要做三件事，逐一说清为什么。

**第一件，统一尺寸。** 一个网络的全连接层要求输入大小固定，所以所有图都得缩放到同样的宽高（CIFAR-10 本身就是 32×32，已经对齐，但换别的数据集时这一步必不可少）。

**第二件，转成张量。** 一张彩色图在电脑里是"高×宽×3 通道"的像素值，范围 0~255。我们要把它变成 PyTorch 张量，并且顺手把范围压到 0~1，同时把通道维提到最前面，变成"通道×高×宽"——这是 PyTorch 卷积层期待的摆放顺序。`transforms.ToTensor()` 一行就替我们办好了这两件事。

**第三件，归一化（normalization）。** 这一步最容易被忽略，却最关键。我们把每个通道的像素值减去均值、再除以标准差，让数据大致落在"零均值、单位方差"附近。为什么？因为前面学梯度下降时讲过，输入的数值范围如果忽大忽小，损失曲面会被拉得又扁又斜，模型怎么调都收敛不动。归一化就是把这个曲面"扶正"，让训练又快又稳。

下面把这三步用 `torchvision.transforms` 串成一条预处理流水线：

```python
import torch
import torchvision
import torchvision.transforms as T

# 一条预处理流水线：依次执行下面几步
transform = T.Compose([
    T.ToTensor(),                       # PNG/JPG → 张量，像素压到 [0,1]，并转成 C×H×W
    T.Normalize(mean=(0.4914, 0.4822, 0.4465),   # CIFAR-10 三个通道的均值
                std=(0.2470, 0.2435, 0.2616)),   # 三个通道的标准差
])

# 自动下载并加载数据集；train=True 是训练集，False 是测试集
train_set = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform)
```

注意那几个写死的均值和标准差，它们是前人在 CIFAR-10 的训练集上统计出来的"标准答案"，直接拿来用即可。这里也藏着一个原则：**统计量只能来自训练集**，绝不能用测试集的数据去算，否则就等于偷看了答案（这个"数据泄漏"的道理，在 [4.3 Pandas 与数据处理](../ch1-intro/1-4-3-pandas-data.md) 里已经强调过）。

数据集准备好了，但我们不会一张一张地喂——那太慢。我们用 **`DataLoader`** 把数据打包成一小批一小批（mini-batch），还能顺手帮我们打乱顺序：

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=64, shuffle=False)
```

`batch_size=64` 表示每次喂 64 张图。训练集要 `shuffle=True` 打乱，免得模型记住了图片出现的顺序；测试集不需要打乱，因为我们只是去考它，不在乎顺序。

## 第二步：搭一个小型 CNN——把卷积块堆起来

预处理把数据备好了，轮到主角登场。我们要搭的不是什么庞然大物，而是一个"小而完整"的卷积神经网络。它的设计思路非常典型，记住这个套路，你能套用到一大半的图像任务上：

> **反复地"卷积 → 激活 → 池化"，把图越缩越小、把通道越变越多；最后把张量摊平，接两层全连接，输出 10 个类别的分数。**

为什么是"图越缩越小、通道越变越多"？因为越往后，网络关心的东西越抽象：浅层在看边缘、颜色这些细枝末节，所以保留高分辨率；深层在看"这是不是一只猫"这种整体概念，不需要那么多像素，但需要更多通道来表达丰富的语义。我们把这"一组卷积+激活+池化"叫做一个**卷积块（conv block）**，下面这个模型一共堆了三块。

```python
import torch.nn as nn
import torch.nn.functional as F

class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 三个卷积块：通道 3 → 32 → 64 → 128，每块后接一次池化把边长减半
        self.conv1 = nn.Conv2d(3,   32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32,  64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)        # 每次把高宽都减半
        # 三次池化后，32×32 变成 4×4，通道是 128，摊平就是 128*4*4
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.25)        # 随机丢弃一部分神经元，防过拟合

    def forward(self, x):                      # x 进来是 (N, 3, 32, 32)
        x = self.pool(F.relu(self.conv1(x)))   # → (N, 32, 16, 16)
        x = self.pool(F.relu(self.conv2(x)))   # → (N, 64,  8,  8)
        x = self.pool(F.relu(self.conv3(x)))   # → (N, 128, 4,  4)
        x = torch.flatten(x, start_dim=1)      # 摊平成 (N, 128*4*4)，保留批维
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)                     # 输出 (N, 10)，10 个类别的原始分数
```

请特别留意 `forward` 里每行末尾的注释——它们标出了张量形状是怎么一步步变化的。**搭 CNN 时最常犯的错，就是算错形状，导致全连接层的输入维度对不上而报错。** 养成在脑子里（或注释里）追踪形状的习惯，能帮你省下大量调试时间。这里 32×32 经过三次"减半"变成 4×4，通道升到 128，所以摊平后是 `128*4*4`，这个数字必须和 `fc1` 的输入维度严丝合缝。

还有一个细节：模型最后输出的是 10 个**原始分数（logits）**，我们**没有**自己加 softmax。这是故意的——等会儿用的损失函数 `CrossEntropyLoss` 内部已经包含了 softmax，你要是再手动加一遍，就重复了。这是另一个高频坑，先记在心里。

## 第三步：训练循环——让模型一轮一轮地学

模型搭好了，但此刻它的参数还是随机的，纯粹瞎猜。要让它真的学会认图，就得反复地"看数据 → 算错得多离谱 → 调整参数往对的方向挪一点"。这个反复的过程，就是**训练循环（training loop）**。

它的骨架是固定的五步，每一批数据都走一遍。我们先把这五步用大白话讲清，再上代码：

1. **前向传播**：把一批图喂进模型，得到预测分数；
2. **算损失**：用损失函数比较"预测"和"真实标签"差多少；
3. **清空旧梯度**：把上一批留下的梯度清零（PyTorch 默认会累加，不清就会乱套）；
4. **反向传播**：调用 `loss.backward()`，自动算出每个参数该往哪调；
5. **更新参数**：优化器 `optimizer.step()` 照着梯度把参数挪一小步。

这五步里的反向传播，背后正是 [2.2 梯度下降与反向传播](2-2-2-backprop.md) 讲的链式法则，只不过 PyTorch 的 `autograd` 替我们自动完成了。我们先准备好损失函数和优化器，再把训练逻辑封装成一个函数：

```python
model = SmallCNN()
criterion = nn.CrossEntropyLoss()                       # 分类任务的标准损失（内含 softmax）
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()                                       # 切到训练模式（让 Dropout 生效）
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:                       # 每次取出一个 batch
        outputs = model(images)                         # 1. 前向传播
        loss = criterion(outputs, labels)               # 2. 算损失

        optimizer.zero_grad()                           # 3. 清空旧梯度
        loss.backward()                                 # 4. 反向传播，自动求梯度
        optimizer.step()                                # 5. 更新参数

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total          # 平均损失、训练准确率
```

那个 `model.train()` 别漏掉。它告诉模型"现在是训练时间"，于是 Dropout 会随机丢神经元；评估时我们会换成 `model.eval()` 关掉这个随机性。这一对开关忘了切，是训练结果飘忽不定的常见原因。

## 第四步：评估——在没见过的数据上考一考

训练时模型的准确率再高，也不能算数——它可能只是把训练集"背"下来了。真正的本事，要看它在**从没见过的测试集**上表现如何。这一步叫**评估（evaluation）**，逻辑和训练几乎一样，但有两处关键差别：

第一，要切到 `model.eval()` 模式，关掉 Dropout；第二，要包在 `torch.no_grad()` 里，告诉 PyTorch "我只是来考试的，不需要算梯度"——这样能省一大半内存、快不少。

```python
@torch.no_grad()                                        # 评估不需要梯度
def evaluate(model, loader):
    model.eval()                                        # 切到评估模式（关掉 Dropout）
    correct, total = 0, 0
    for images, labels in loader:
        preds = model(images).argmax(dim=1)             # 取分数最高的类别当预测
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total                              # 测试准确率
```

现在把训练和评估串起来，跑上几个 **epoch**（一个 epoch 表示把整个训练集完整过一遍）。在 CPU 上，CIFAR-10 一个 epoch 大约要几分钟，跑个 10 轮足够看出明显进步：

```python
for epoch in range(10):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    te_acc = evaluate(model, test_loader)
    print(f"Epoch {epoch+1:2d} | "
          f"训练损失 {tr_loss:.3f} | 训练准确率 {tr_acc:.3f} | 测试准确率 {te_acc:.3f}")
```

跑起来你会看到，测试准确率从一开始接近 0.10（10 个类别瞎猜的水平）一路爬升，几个 epoch 后稳稳越过 0.70。这就是流水线跑通的标志——你的第一个图像分类器，活了。

<div class="diagram">
<svg viewBox="0 0 380 200" font-family="-apple-system, 'Segoe UI', sans-serif">
  <defs>
    <marker id="ca" markerWidth="8" markerHeight="8" refX="5" refY="3" orient="auto"><path d="M0 0 L5 3 L0 6 Z" fill="var(--dia-stroke-tertiary)"/></marker>
  </defs>
  <!-- 坐标轴 -->
  <line x1="46" y1="168" x2="350" y2="168" stroke="var(--dia-stroke-tertiary)" stroke-width="1" marker-end="url(#ca)"/>
  <line x1="46" y1="168" x2="46" y2="24" stroke="var(--dia-stroke-tertiary)" stroke-width="1" marker-end="url(#ca)"/>
  <!-- 10% 瞎猜基准线 -->
  <line x1="46" y1="150" x2="340" y2="150" stroke="var(--dia-stroke-tertiary)" stroke-width="1" stroke-dasharray="4 3"/>
  <!-- 测试准确率上升曲线 -->
  <path d="M50 150 Q120 96 180 70 T330 46" fill="none" stroke="var(--dia-accent)" stroke-width="2.5"/>
  <!-- 文字标注，全部放空白处 -->
  <text x="44" y="18" text-anchor="middle" font-size="11" fill="var(--dia-stroke-soft)">准确率</text>
  <text x="350" y="186" text-anchor="end" font-size="11" fill="var(--dia-stroke-soft)">epoch →</text>
  <text x="300" y="38" font-size="11" fill="var(--dia-accent)">≈ 0.70+</text>
  <text x="92" y="145" font-size="10" fill="var(--dia-stroke-tertiary)">瞎猜基准 0.10</text>
</svg>
</div>
<p class="figure-caption">图 3.2-2：随着 epoch 增加，测试准确率从瞎猜的 0.10 一路爬升，越过 0.70——这就是流水线跑通的样子。</p>

## 从零和库版：这里的"库版"指什么

前面讲数学时，我们习惯先"从零（NumPy）手写"再"用库（PyTorch）对照"。但到了图像分类这一层，**从零用 NumPy 手写一个能训练的 CNN 已经不现实了**——光是高效实现卷积的反向传播就足够写一整章，而且慢到没法实际训练。所以在工程里，PyTorch 这层就是我们的"底座"，没人再往下手写。

那"对照"体现在哪呢？体现在**你心里要清楚每一行库代码背后在算什么**。比如 `loss.backward()` 这一行，背后就是 [2.2 梯度下降与反向传播](2-2-2-backprop.md) 里你亲手推过的链式法则；`nn.Conv2d` 背后就是 [3.1 卷积与池化](2-3-1-conv-pooling.md) 里那个滑动窗口的加权求和。**库帮你把脏活累活自动化了，但它没有变魔术。** 你越清楚每行代码对应哪个原理，调起 bug 来就越有底气——这正是"从零理解 + 库版实现"在这一节真正的含义。

## 容易踩的坑

- **忘了归一化**：跳过 `Normalize`，或者均值标准差填错通道，模型往往收敛得又慢又差。这是图像任务里最隐蔽、最高频的坑。
- **手动多加了一层 softmax**：`CrossEntropyLoss` 内部已含 softmax，你若在模型最后再加一个 `softmax`，相当于做了两次，梯度会被压扁、训练变慢。模型最后一层直接输出 logits 就对了。
- **全连接层输入维度算错**：三次池化把 32×32 变 4×4，所以摊平是 `128*4*4`；填错这个数会直接报 `mat1 and mat2 shapes cannot be multiplied`。改 `padding`、改池化次数后，记得重算这个维度。
- **忘了 `optimizer.zero_grad()`**：PyTorch 的梯度默认是累加的，不清零，这一批的梯度会叠到上一批上，训练彻底乱套。
- **`train()` / `eval()` 模式没切换**：评估时忘了 `model.eval()`，Dropout 还在随机丢神经元，准确率会无端偏低且每次都不一样。
- **CPU 上贪心调大模型**：Round 1 是 CPU 环境，盲目加深网络、加大 batch，只会让一个 epoch 慢到等不起。先用小模型把流水线跑通，再谈提升。

## 它在后面会怎么用到

这一节是你第一次把"卷积"真正用起来训练出一个能用的模型，它往两个方向延伸：

- **嫌从零训练太慢、准确率上不去？** 那就别从零学了——直接拿别人在海量图片上预训练好的网络，改一改最后几层来用，这叫迁移学习，准确率和速度往往都碾压从零训练（见 [3.3 迁移学习与预训练编码器](2-3-3-transfer-learning.md)）。
- **想系统地评估模型好坏？** 准确率只是最粗的一个指标，类别不均衡时会骗人。怎么看混淆矩阵、精确率、召回率，回去复习 [1.4 分类评估指标](2-1-4-classification-metrics.md)。
- 这条"预处理 → 模型 → 训练 → 评估"的流水线，本身就是后面所有 CV 任务（如 [3.1 目标检测](../ch3-round2/3-3-1-object-detection.md)）的通用骨架。

## 练习

??? note "基础练习"
    1. 把上面四段代码拼成一个完整脚本跑起来，在 CIFAR-10 上训练 10 个 epoch，记录每一轮的测试准确率，确认它能稳定越过 0.70。
    2. 故意把 `transform` 里的 `Normalize` 那一行删掉，重新训练，对比一下收敛速度和最终准确率差多少。亲眼看看归一化到底有多重要。
    3. 把模型最后一层后面手动加一个 `F.softmax`，观察训练是否变慢或变差，体会"为什么不该重复加 softmax"。

??? note "进阶练习"
    1. 给训练集的 `transform` 加上随机水平翻转 `T.RandomHorizontalFlip()` 和随机裁剪 `T.RandomCrop(32, padding=4)`（这叫数据增强），再训练一遍，看测试准确率能不能再提几个点。
    2. 写一个函数，挑出模型分类**错得最离谱**的 10 张图（即预测分数很高但其实答错的），把它们和真实/预测标签一起打印出来，分析模型容易把哪两类搞混。
    3. 给每个卷积块后面加上批归一化 `nn.BatchNorm2d`，对比加之前/之后的收敛曲线，体会它为什么能让训练更稳更快。

## 小结

- 图像分类就是一条四步流水线：**预处理 → 搭 CNN → 训练循环 → 评估**，换任务无非是替换零件。
- 预处理三件事——缩放、转张量、归一化——一步都不能少，尤其归一化最容易被漏掉又最关键。
- 小型 CNN 的套路是"卷积块越堆，图越小、通道越多，最后接全连接输出 logits"；搭的时候**一定要追踪张量形状**。
- 训练循环固定五步（前向、算损失、清梯度、反向、更新），评估时记得切 `eval()` 并用 `no_grad()`。
- 库帮你自动化了脏活，但没变魔术——你越清楚每行代码对应哪条原理，调 bug 越有底气。

想看更细的官方教程，可以跟一遍 [PyTorch 官方的 CIFAR-10 分类教程](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)，它和本节流程几乎一一对应；想把准确率推得更高，下一节的 [3.3 迁移学习与预训练编码器](2-3-3-transfer-learning.md) 会告诉你一条捷径。
