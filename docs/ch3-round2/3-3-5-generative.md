# 3.5 生成模型 GAN·扩散

> **难度** ⭐⭐⭐⭐☆ · **前置**：[2.3 自编码器](3-2-3-autoencoders.md)、概率基础

!!! abstract "读完这一节，你会"
    - 说清"生成"和"判别"到底有什么区别，理解一个模型怎么才算"会画画"
    - 看懂 GAN 的对抗思路：生成器和判别器是怎么一边骗一边识破、互相逼对方变强的
    - 理解扩散模型的"先加噪、再学着去噪"两步走，知道它为什么训练起来比 GAN 稳
    - 在脑子里摆出一张 VAE / GAN / 扩散的对比表，知道竞赛里给一个任务该想到哪一个

## 先分清楚：判别模型和生成模型在干两件不同的事

前面学的分类器，比如逻辑回归、CNN 图像分类，它们都在回答同一个问题：**"给你一张图，它是猫还是狗？"** 这类模型只关心怎么在不同类别之间画一条分界线，我们叫它**判别模型（discriminative model）**。

可这一节要讲的事情完全不同。我们想让模型**自己造出一张以前没见过、但看起来很真的猫的图片**。它要回答的不再是"这是不是猫"，而是"猫长什么样、我能不能照着画一只出来"。能做到这件事的模型，就叫**生成模型（generative model）**。

换个说法你会更有感觉：判别模型像一个鉴定师，只负责判断真假；生成模型像一个画家，要无中生有。本节的三位主角——VAE、GAN、扩散模型——都是画家，只是各自的作画方法天差地别。

VAE 你在 [2.3 自编码器](3-2-3-autoencoders.md) 里已经见过雏形了：它是"编码-解码"再加上一层概率约束，让中间那个隐空间变得平滑、可采样。所以这一节我们把重点放在另外两位：GAN 和扩散。

## GAN：一个造假者和一个鉴定师的拉锯战

GAN 的全名是**生成对抗网络（Generative Adversarial Network）**，它的点子非常巧妙，巧到值得你专门记住。

想象一个造假币的人和一个验钞的警察。一开始造假者技术很烂，印出来的钞票一眼假；警察轻松识破。可造假者会偷看警察是怎么识破的，然后改进；警察也跟着升级自己的鉴别能力。两人你来我往地较劲，**逼着造假者的手艺越来越精**，到最后印出来的假币连警察都分不出真假——这时候，我们要的"以假乱真的生成器"就练成了。

GAN 里就有这么两个网络：

- **生成器（generator，记作 $G$）**：吃一串随机噪声 $\mathbf{z}$，吐出一张假图 $G(\mathbf{z})$。它就是那个造假者。
- **判别器（discriminator，记作 $D$）**：吃一张图，输出"这是真图的概率"。它就是那个警察。

<div class="diagram">
<svg viewBox="0 0 380 180" font-family="-apple-system, 'Segoe UI', sans-serif">
  <defs>
    <marker id="ga" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="var(--dia-stroke-soft)"/></marker>
  </defs>
  <!-- 噪声 -->
  <text x="20" y="60" font-size="12" fill="var(--dia-stroke-soft)">噪声 z</text>
  <line x1="55" y1="56" x2="95" y2="56" stroke="var(--dia-stroke-soft)" stroke-width="1.5" marker-end="url(#ga)"/>
  <!-- 生成器 -->
  <rect x="98" y="38" width="74" height="38" rx="5" fill="none" stroke="var(--dia-accent)" stroke-width="2"/>
  <text x="135" y="61" text-anchor="middle" font-size="12" fill="var(--dia-accent)">生成器 G</text>
  <line x1="172" y1="56" x2="212" y2="56" stroke="var(--dia-stroke-soft)" stroke-width="1.5" marker-end="url(#ga)"/>
  <text x="192" y="40" text-anchor="middle" font-size="10" fill="var(--dia-stroke-soft)">假图</text>
  <!-- 真图 -->
  <text x="160" y="142" font-size="11" fill="var(--dia-green)">真实图片</text>
  <line x1="172" y1="138" x2="212" y2="80" stroke="var(--dia-stroke-soft)" stroke-width="1.5" marker-end="url(#ga)"/>
  <!-- 判别器 -->
  <rect x="215" y="42" width="80" height="40" rx="5" fill="none" stroke="var(--dia-blue)" stroke-width="2"/>
  <text x="255" y="66" text-anchor="middle" font-size="12" fill="var(--dia-blue)">判别器 D</text>
  <line x1="295" y1="62" x2="335" y2="62" stroke="var(--dia-stroke-soft)" stroke-width="1.5" marker-end="url(#ga)"/>
  <text x="338" y="58" font-size="11" fill="var(--dia-stroke-soft)">真/假</text>
</svg>
</div>
<p class="figure-caption">图 3.5-1：GAN 的对抗结构。生成器把噪声变成假图，判别器在真图和假图之间做判断；两者各自变强，最后生成器骗过判别器。</p>

那"较劲"这件事，用数学怎么写？两个网络共享同一个目标函数，只不过一个想最小化它、一个想最大化它：

$$
\min_{G}\ \max_{D}\ \mathbb{E}_{x\sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z\sim p_z}[\log\bigl(1 - D(G(z))\bigr)].
$$

别被它吓到，我们拆开看。判别器 $D$ 想把真图判成真（让 $\log D(x)$ 大）、把假图判成假（让 $D(G(z))$ 小，于是 $\log(1-D(G(z)))$ 大），所以它在 **max**。生成器 $G$ 反着来，它要让假图骗过判别器，也就是让 $D(G(z))$ 尽量大，于是那一项变小，所以它在 **min**。一个 $\min$ 一个 $\max$ 套在一起，这正是"对抗"两个字的来历。

如果我们还想**指定生成什么**——比如"画一只猫"而不是随便画——只要把类别标签 $c$ 同时喂给生成器和判别器就行。这种带条件的版本叫**条件 GAN（conditional GAN, cGAN）**，后来"输入一句文字、输出一张图"的文生图模型，骨子里就是这个思路的放大版。

下面用 PyTorch 把这套对抗训练的核心循环写出来。注意看：**每一轮我们要分两步，先更新判别器，再更新生成器**，因为它俩的目标是打架的，不能一起更新：

```python
import torch, torch.nn as nn

G = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 784), nn.Tanh())   # 噪声->假图(28x28拉平)
D = nn.Sequential(nn.Linear(784, 256), nn.LeakyReLU(0.2), nn.Linear(256, 1), nn.Sigmoid())  # 图->真的概率
bce = nn.BCELoss()
optG = torch.optim.Adam(G.parameters(), lr=2e-4)
optD = torch.optim.Adam(D.parameters(), lr=2e-4)

real = next_real_batch()                 # 一批真实图片，形状 (B, 784)，假设已准备好
z = torch.randn(real.size(0), 64)        # 一批随机噪声
fake = G(z)

# 第一步：训练判别器——真图标 1、假图标 0，让它学会分辨
optD.zero_grad()
loss_D = bce(D(real), torch.ones(real.size(0), 1)) \
       + bce(D(fake.detach()), torch.zeros(real.size(0), 1))   # detach: 这步先别动生成器
loss_D.backward(); optD.step()

# 第二步：训练生成器——它希望判别器把自己的假图判成"真"(标 1)
optG.zero_grad()
loss_G = bce(D(fake), torch.ones(real.size(0), 1))
loss_G.backward(); optG.step()
```

代码里有个容易忽略的关键细节：训练判别器时用了 `fake.detach()`，意思是"这一步只更新判别器，别让梯度顺着假图流回生成器"。少了这个 `detach`，两个网络的梯度会搅在一起，训练直接乱套——这是写 GAN 时的头号坑。

## 扩散模型：先把图片"搅成噪声"，再学着一步步还原

GAN 的对抗训练虽然巧妙，但出了名地难调：两个网络稍微失衡，要么生成器偷懒只会画几种图（叫**模式坍塌（mode collapse）**），要么干脆训练崩掉。于是近几年的主流换成了思路更稳的**扩散模型（diffusion model）**，现在你听过的那些惊艳的文生图工具，背后基本都是它。

它的想法朴素得有点反直觉，我们分两个方向理解。

**前向过程（加噪）**：拿一张清晰的图，一点点往上撒高斯噪声，撒很多步（比如 1000 步）之后，原图就被彻底搅成了一团纯噪声。这一步不需要学习，是我们人为定义的、固定的"破坏"流程，每一步加多少噪声由一个**噪声调度（noise schedule）**说了算。第 $t$ 步的图可以一步到位地写出来：

$$
x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\qquad \epsilon\sim\mathcal N(0, I).
$$

这里 $x_0$ 是原图，$\bar\alpha_t$ 是调度给出的系数，$t$ 越大、 $\bar\alpha_t$ 越小，图就越接近纯噪声 $\epsilon$。

**逆向过程（去噪）**：真正要学的是反方向——从一团纯噪声出发，一步步把噪声去掉，最后还原出一张清晰的图。我们训练一个神经网络（通常是 U-Net），它的任务出奇地简单：**看一眼第 $t$ 步的带噪图 $x_t$，把里头掺进去的那个噪声 $\epsilon$ 猜出来。** 损失就是猜得准不准：

$$
L = \mathbb{E}_{x_0,\epsilon,t}\,\bigl\|\,\epsilon - \epsilon_\theta(x_t, t)\,\bigr\|^2.
$$

<div class="diagram">
<svg viewBox="0 0 380 130" font-family="-apple-system, 'Segoe UI', sans-serif">
  <defs>
    <marker id="da" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="var(--dia-blue)"/></marker>
    <marker id="db" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="var(--dia-green)"/></marker>
  </defs>
  <rect x="20" y="50" width="40" height="40" rx="4" fill="none" stroke="var(--dia-stroke-soft)" stroke-width="1.5"/>
  <text x="40" y="74" text-anchor="middle" font-size="11" fill="var(--dia-stroke-soft)">清晰</text>
  <rect x="170" y="50" width="40" height="40" rx="4" fill="none" stroke="var(--dia-stroke-soft)" stroke-width="1.5"/>
  <text x="190" y="74" text-anchor="middle" font-size="11" fill="var(--dia-stroke-soft)">半噪</text>
  <rect x="320" y="50" width="40" height="40" rx="4" fill="none" stroke="var(--dia-stroke-soft)" stroke-width="1.5"/>
  <text x="340" y="74" text-anchor="middle" font-size="11" fill="var(--dia-stroke-soft)">纯噪</text>
  <!-- 前向加噪 -->
  <line x1="62" y1="60" x2="168" y2="60" stroke="var(--dia-blue)" stroke-width="1.5" marker-end="url(#da)"/>
  <line x1="212" y1="60" x2="318" y2="60" stroke="var(--dia-blue)" stroke-width="1.5" marker-end="url(#da)"/>
  <text x="190" y="32" text-anchor="middle" font-size="11" fill="var(--dia-blue)">前向：加噪（固定）</text>
  <!-- 逆向去噪 -->
  <line x1="318" y1="82" x2="212" y2="82" stroke="var(--dia-green)" stroke-width="1.5" marker-end="url(#db)"/>
  <line x1="168" y1="82" x2="62" y2="82" stroke="var(--dia-green)" stroke-width="1.5" marker-end="url(#db)"/>
  <text x="190" y="112" text-anchor="middle" font-size="11" fill="var(--dia-green)">逆向：去噪（学出来的）</text>
</svg>
</div>
<p class="figure-caption">图 3.5-2：扩散模型的两个方向。蓝色的前向加噪是固定流程，绿色的逆向去噪才是神经网络要学的本事；生成时就从纯噪声沿绿色箭头走回清晰图。</p>

到了**生成（采样）**的时候，我们就从一团纯噪声出发，让网络反复预测噪声、减掉一点、再预测，循环很多步，一张全新的图就这么"显影"出来了。和 GAN 比，它没有对抗、只有一个回归式的损失，所以训练稳得多——这正是它后来居上的根本原因。

```python
import torch
# x: 一批真实图片 (B,C,H,W)；T: 总步数；net: 一个预测噪声的网络 eps_theta(x_t, t)
T = 1000
betas = torch.linspace(1e-4, 0.02, T)        # 噪声调度：每一步加多少噪声
abar = torch.cumprod(1 - betas, dim=0)        # 累积系数 \bar{alpha}_t

t = torch.randint(0, T, (x.size(0),))         # 每张图随机抽一个时间步
eps = torch.randn_like(x)                     # 真正掺进去的噪声
a = abar[t].view(-1, 1, 1, 1)
x_t = a.sqrt() * x + (1 - a).sqrt() * eps     # 一步到位造出第 t 步的带噪图

eps_pred = net(x_t, t)                        # 让网络去猜这个噪声
loss = ((eps - eps_pred) ** 2).mean()         # 猜得越准越好
loss.backward()
```

读这段代码，你会发现扩散模型的训练循环反而比 GAN 简单——没有两个网络互搏，就是"造带噪图、猜噪声、算均方误差"三步。难点不在训练，而在采样要跑很多步、比较慢，这也是后续很多研究在想办法加速的地方。

## 三种生成模型，到底该选哪个

讲到这里，三位画家都登场了。它们的差别一句话各能概括：**VAE 靠概率编码-解码，GAN 靠两网对抗，扩散靠去噪还原。** 但你更需要的是一张能在考场上一眼对号入座的表：

| 维度 | VAE（[2.3](3-2-3-autoencoders.md)） | GAN | 扩散模型 |
| --- | --- | --- | --- |
| 核心思路 | 编码到概率隐空间再解码 | 生成器 vs 判别器对抗 | 加噪后学着逐步去噪 |
| 训练稳定性 | 稳，但容易糊 | 不稳，易模式坍塌 | 稳，损失就是回归 |
| 生成质量 | 偏模糊 | 锐利但有时不真 | 目前最高、最逼真 |
| 生成速度 | 快（一次解码） | 快（一次前向） | 慢（要迭代很多步） |
| 隐空间可控 | 强，便于插值 | 一般 | 一般 |

怎么用这张表？如果题目强调**生成质量、文生图**，先想扩散；如果强调**实时、一次出图**，GAN 仍有速度优势；如果需要一个**结构清晰、能在隐空间做插值**的轻量生成器，VAE 够用。竞赛里出现"用哪种生成模型更合适"这类问法时，照着这几行权衡就能答得有理有据。

## 容易踩的坑

- **GAN 训练时忘了 `detach`**：更新判别器那一步，假图必须 `.detach()`，否则梯度会窜回生成器，把训练搅乱。这是 GAN 代码最常见的 bug。
- **把模式坍塌当成"收敛了"**：GAN 的损失看着平稳，不代表训得好——它可能只学会画那么几张图。判断 GAN 好不好要看生成样本的多样性，不能只盯 loss。
- **扩散模型嫌它"训练慢"其实是采样慢**：扩散的训练一步就够，真正慢的是生成时要迭代上百上千步。别把这两件事搞混。
- **噪声调度不能乱设**：$\beta_t$ 从小到大的安排直接影响效果，照搬论文里成熟的线性或余弦调度，别自己瞎调。
- **生成模型的"评价"很微妙**：生成任务没有标准答案，不能用准确率衡量；要用 FID 这类指标比较生成分布和真实分布有多接近。

## 它在后面会怎么用到

- 扩散模型里负责去噪的主干网络，正是你在 [3.2 图像分割](3-3-2-segmentation.md) 见过的 **U-Net**；而文生图把文字喂进去靠的是 [3.4 视觉-文本编码器 CLIP](3-3-4-clip.md) 那套图文对齐的思路。
- 生成器和判别器的对抗，本质还是反向传播在驱动，底子是 [2.2 梯度下降与反向传播](../ch2-round1/2-2-2-backprop.md)；VAE 那一支则直接承接 [2.3 自编码器](3-2-3-autoencoders.md)。
- 条件 GAN 和扩散的"加条件"思路，到了 [4.4 预训练大模型](3-4-4-pretrained-llms.md) 会以"给生成过程加提示"的形式再次出现。
- 真题里它常和 [5.2 多域综合模拟](3-5-2-mock.md) 的图像生成题挂钩，备赛时值得拿一道练手。

## 练习

??? note "基础练习"
    1. 用自己的话解释：GAN 的目标函数里，为什么生成器在 $\min$、判别器在 $\max$？把"造假者/警察"的类比对应到 $\log D(x)$ 和 $\log(1-D(G(z)))$ 这两项上。
    2. 扩散模型的前向加噪那一步需不需要训练？为什么说真正"学"的只有逆向去噪？
    3. 给定噪声调度让 $\bar\alpha_t$ 随 $t$ 增大而减小，想一想：$t$ 很大时 $x_t$ 更接近原图还是纯噪声？

??? note "进阶练习"
    1. 在上面的 GAN 代码里，故意去掉 `fake.detach()` 跑一跑，观察训练为什么会出问题，并解释梯度是怎么"窜错"的。
    2. 把扩散训练代码改成**条件版**：除了 $x_t$ 和 $t$，再把类别标签也喂给网络。想清楚标签该在哪里拼进去，并说说这和条件 GAN 的"加条件"是不是一回事。
    3. 查一下 **FID（Fréchet Inception Distance）** 是怎么算的，解释为什么生成任务不能用分类准确率来评价，而要比较两个分布的距离。

## 小结

一句话记住这三位画家：**VAE 是编码-解码加概率、GAN 是造假者和鉴定师对抗、扩散是先把图搅成噪声再学着还原**；其中扩散因为训练稳、质量高，成了当下文生图的主力。想把扩散的来龙去脉看明白，推荐读 Lilian Weng 的博客 [*What are Diffusion Models?*](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)；想补 GAN 的原始直觉，可以看 Goodfellow 那篇 [*Generative Adversarial Nets*](https://arxiv.org/abs/1406.2661) 的引言部分。
