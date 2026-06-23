# 2.3 自编码器

> **难度** ⭐⭐⭐☆☆ · **前置**：[3.3 概率与统计](../ch1-intro/1-3-3-probability-statistics.md)、[2.1 感知机与 MLP](../ch2-round1/2-2-1-perceptron-mlp.md)

!!! abstract "读完这一节，你会"
    - 理解"编码-解码"这套对称结构，知道为什么中间要卡一个又窄又小的瓶颈
    - 会用 PyTorch 写出一个能压缩、能重构的自编码器，并看懂重构损失在监督什么
    - 搞懂变分自编码器（VAE）多出来的两样东西——重参数化技巧和 KL 散度——各自在解决什么问题
    - 知道自编码器在压缩、去噪、异常检测里怎么用，以及它和后面生成模型的血缘关系

## 一个没有标签也能学的网络

到目前为止，你见过的网络几乎都要有"标准答案"——图片配着类别标签，句子配着情感标签，模型对照答案改进自己。可现实里，**绝大多数数据是没有标签的**：你手机里几万张照片，没人一张张给它们打标。这时候还能学点什么吗？

答案是能。我们换个思路：不给模型外部答案，而是**让它把答案藏在数据自己身上**。具体做法很妙——逼着网络先把一张输入"压扁"成一小串数字，再仅凭这一小串数字尽量原样还原出输入。如果它真能还原得八九不离十，那说明这一小串数字抓住了输入里最要紧的信息。这个"先压缩、再还原、拿输入自己当答案"的网络，就叫**自编码器（autoencoder）**。

它为什么值得学？因为这套"自己监督自己"的思路，是后面自监督学习、表示学习、乃至生成模型的共同祖先。把它吃透，你会发现 Round 2 里好几个看似不相干的主题，底子都是它。

## 编码、瓶颈、解码：一个对称的沙漏

自编码器的结构非常好记，它长得像一个沙漏：两头宽、中间细。我们顺着数据流走一遍。

数据先进入**编码器（encoder）**，这是一串逐渐变窄的网络层，把高维输入一步步压成一个低维向量。这个被压到最窄处的向量，我们叫它**隐编码（latent code）**，它所在的低维空间就是**隐空间（latent space）**。然后**解码器（decoder）**接手，它的结构和编码器恰好镜像对称，把这个小向量一层层放大，重新还原成和输入一样大小的输出。下面这张图把整条路画了出来：

<div class="diagram">
<svg viewBox="0 0 420 200" font-family="-apple-system, 'Segoe UI', sans-serif">
  <defs>
    <marker id="ae-a" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="var(--dia-accent)"/></marker>
  </defs>
  <!-- 输入 x（高） -->
  <rect x="30" y="50" width="26" height="100" rx="3" fill="var(--dia-blue)" opacity="0.85"/>
  <!-- 编码器收窄 -->
  <polygon points="70,55 130,80 130,120 70,145" fill="var(--dia-accent-soft)" stroke="var(--dia-accent)" stroke-width="1.5"/>
  <!-- 瓶颈 z（窄） -->
  <rect x="150" y="88" width="22" height="24" rx="3" fill="var(--dia-accent)"/>
  <!-- 解码器变宽 -->
  <polygon points="192,80 252,55 252,145 192,120" fill="var(--dia-accent-soft)" stroke="var(--dia-accent)" stroke-width="1.5"/>
  <!-- 输出 x̂（高） -->
  <rect x="266" y="50" width="26" height="100" rx="3" fill="var(--dia-blue)" opacity="0.5"/>
  <!-- 重构损失箭头：放在底部空白处 -->
  <path d="M43 168 L43 175 L279 175 L279 168" fill="none" stroke="var(--dia-stroke-tertiary)" stroke-width="1"/>
  <!-- 文字标注全部放空白处 -->
  <text x="43" y="42" text-anchor="middle" font-size="12" fill="var(--dia-blue)">输入 x</text>
  <text x="100" y="190" text-anchor="middle" font-size="11" fill="var(--dia-stroke-soft)">编码器</text>
  <text x="161" y="80" text-anchor="middle" font-size="12" fill="var(--dia-accent)">z</text>
  <text x="161" y="128" text-anchor="middle" font-size="10" fill="var(--dia-stroke-soft)">瓶颈</text>
  <text x="222" y="190" text-anchor="middle" font-size="11" fill="var(--dia-stroke-soft)">解码器</text>
  <text x="279" y="42" text-anchor="middle" font-size="12" fill="var(--dia-stroke-soft)">输出 x̂</text>
  <text x="340" y="175" font-size="11" fill="var(--dia-stroke-soft)">重构要尽量 ≈ 输入</text>
</svg>
</div>
<p class="figure-caption">图 2.3-1：自编码器是个对称的沙漏——编码器把 x 压成瓶颈 z，解码器再把 z 还原成 x̂，训练目标是让 x̂ 尽量接近 x。</p>

用数学写出来就很简洁。设编码器是函数 $z=f_\theta(x)$，解码器是 $\hat{x}=g_\phi(z)$，那么整个自编码器做的就是

$$\hat{x}=g_\phi\big(f_\theta(x)\big),$$

而我们希望 $\hat{x}$ 尽量等于原来的 $x$。

这里有个关键设计，你一定要琢磨明白：**为什么中间非要卡一个又窄又小的瓶颈？** 因为如果不卡，让隐编码和输入一样宽，网络完全可以偷懒——把输入原封不动抄过去就行了，那它什么也没学到。正是这个窄瓶颈逼着网络做取舍：信息塞不下了，它只能保留最有代表性的特征、丢掉次要的噪声。所以这个瓶颈不是缺陷，而是自编码器能学到东西的**根本原因**。

## 重构损失：拿输入自己当答案

既然没有外部标签，那训练时拿什么算损失？答案前面其实已经埋下了——**拿输入 $x$ 自己当答案**。我们衡量"还原得像不像"，这个差距就叫**重构损失（reconstruction loss）**。

对于像素值、音频这类连续数据，最常用的是均方误差，把还原图和原图逐像素相减再平方：

$$L_{\text{rec}}=\frac{1}{n}\sum_{i=1}^{n}\big\|x_i-\hat{x}_i\big\|^2.$$

如果数据是 0/1 的二值图（比如手写数字 MNIST 归一化后），也常用二元交叉熵。无论哪种，核心都一样：**还原得越像，损失越小**。训练就是不断调整编码器和解码器的参数，把这个损失压下去。

讲到这你应该已经看出门道了：自编码器表面上是"无监督"，但它其实偷偷把问题变成了一个"自己生成标签"的监督问题——输入既是题目，又是答案。这种巧妙的转化，正是**自监督学习（self-supervised learning）**的精髓，后面 [3.3 数据增强与自监督](3-3-3-augmentation-ssl.md) 会专门展开。

## 动手搭一个最朴素的自编码器

道理讲完，我们用 PyTorch 把它搭出来。任务很经典：把 MNIST 手写数字（每张 $28\times28=784$ 个像素）压成一个 32 维的小向量，再还原回去。

先定义网络。注意编码器和解码器是镜像的：一个从 784 收窄到 32，另一个从 32 放大回 784。

```python
import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        # 编码器：784 → 128 → 32，逐层收窄到瓶颈
        self.encoder = nn.Sequential(
            nn.Linear(784, 128), nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        # 解码器：32 → 128 → 784，与编码器镜像对称
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 784), nn.Sigmoid(),   # 像素归一到 [0,1]，故用 Sigmoid
        )

    def forward(self, x):
        z = self.encoder(x)        # 压成瓶颈向量 z
        x_hat = self.decoder(z)    # 再从 z 还原
        return x_hat
```

然后是训练循环。这里最该留意的一行，是**损失拿 `x_hat` 和 `x` 自己比**——没有任何外部标签参与：

```python
model = AutoEncoder()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()           # 用均方误差当重构损失

for x, _ in train_loader:        # 注意：标签 _ 我们根本不用
    x = x.view(x.size(0), -1)    # 把 28×28 拉平成 784 维
    x_hat = model(x)
    loss = loss_fn(x_hat, x)     # 还原图 vs 原图，越像损失越小
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

训练几轮后，你把任意一张数字喂进去，输出会是一张略微模糊、但能认出是同一个数字的图。模糊是正常的——32 个数字毕竟装不下全部 784 个像素的细节，网络只能保住"这是个 8、大致什么形状"这种主干信息。**这点模糊恰恰证明瓶颈在起作用**：它在帮你做有损压缩。

顺便说一句，如果你把网络里的非线性（ReLU）全去掉、只留线性层，那这个自编码器学出来的隐空间会和 **PCA 主成分分析**几乎一致（见 [1.8 无监督学习](../ch2-round1/2-1-8-unsupervised.md)）。所以你可以把自编码器理解成"PCA 的非线性升级版"——它能捕捉 PCA 那种直线投影抓不住的弯曲结构。

## 从自编码器到 VAE：让隐空间变得"能采样"

普通自编码器很会压缩，可它有个软肋：**隐空间里到处是洞**。训练时每张图被映射成一个孤零零的点，点与点之间的空白区域，解码器从没见过，你随便取一个点丢进解码器，多半还原出一团乱码。这意味着，普通自编码器**不能用来生成新数据**——你没法可靠地"凭空造一个点"让它变出一张合理的图。

**变分自编码器（variational autoencoder，简称 VAE）**就是来补这个洞的。它的核心改动只有一句话：**编码器不再把输入压成一个固定的点，而是压成一个概率分布。** 具体说，编码器对每个输入输出两组数——一个均值 $\mu$ 和一个（对数）方差，描述一个高斯分布；隐编码 $z$ 不是直接取那个均值，而是**从这个分布里随机采样**得到的。

为什么这么一改就解决问题了？因为每个输入现在对应的不是一个点，而是一小团"云"，相邻的云会互相重叠、把空白填满。于是隐空间变得连续而光滑，你在里面任取一点，解码出来都是合理的图——VAE 因此能用来**生成**全新的样本。下面这张图对比了两者的隐空间：

<div class="diagram">
<svg viewBox="0 0 420 190" font-family="-apple-system, 'Segoe UI', sans-serif">
  <!-- 左：普通 AE，离散的点，有空洞 -->
  <rect x="20" y="30" width="150" height="120" rx="4" fill="none" stroke="var(--dia-stroke-tertiary)" stroke-width="1"/>
  <circle cx="55" cy="60" r="3.5" fill="var(--dia-blue)"/>
  <circle cx="120" cy="55" r="3.5" fill="var(--dia-blue)"/>
  <circle cx="70" cy="120" r="3.5" fill="var(--dia-blue)"/>
  <circle cx="135" cy="125" r="3.5" fill="var(--dia-blue)"/>
  <circle cx="95" cy="90" r="3.5" fill="var(--dia-blue)"/>
  <text x="95" y="22" text-anchor="middle" font-size="11" fill="var(--dia-stroke-soft)">普通 AE：孤立的点</text>
  <text x="95" y="172" text-anchor="middle" font-size="10" fill="var(--dia-stroke-tertiary)">点之间是空洞</text>
  <!-- 右：VAE，重叠的高斯团，填满空间 -->
  <rect x="250" y="30" width="150" height="120" rx="4" fill="none" stroke="var(--dia-stroke-tertiary)" stroke-width="1"/>
  <circle cx="290" cy="70" r="22" fill="var(--dia-accent-soft)" stroke="var(--dia-accent)" stroke-width="1"/>
  <circle cx="335" cy="95" r="22" fill="var(--dia-accent-soft)" stroke="var(--dia-accent)" stroke-width="1"/>
  <circle cx="305" cy="115" r="22" fill="var(--dia-accent-soft)" stroke="var(--dia-accent)" stroke-width="1"/>
  <text x="325" y="22" text-anchor="middle" font-size="11" fill="var(--dia-stroke-soft)">VAE：重叠的高斯团</text>
  <text x="325" y="172" text-anchor="middle" font-size="10" fill="var(--dia-accent)">连续、可采样</text>
</svg>
</div>
<p class="figure-caption">图 2.3-2：普通自编码器把数据压成孤立的点、点间留洞；VAE 把每个输入压成一团高斯分布，互相重叠填满隐空间，于是任取一点都能解码出合理样本。</p>

### 重参数化：让"随机采样"也能求导

可这里冒出一个棘手的问题。我们说 $z$ 是"从分布里随机采样"得到的——但**随机采样这个动作没法求导**，梯度一碰到它就断了，反向传播就传不回编码器了。这下编码器还怎么训练？

解决办法叫**重参数化技巧（reparameterization trick）**，思路出奇地简单：把随机性从"主干道"上挪到"旁路"。我们不直接对分布采样，而是先从一个固定的标准正态分布里抽一个噪声 $\epsilon\sim\mathcal{N}(0,1)$，再用编码器输出的 $\mu$ 和 $\sigma$ 把它"拼装"成想要的样本：

$$z=\mu+\sigma\odot\epsilon,\qquad \epsilon\sim\mathcal{N}(0,1).$$

你看这一步的妙处：$\mu$ 和 $\sigma$ 是编码器算出来的、可以正常求导；随机的 $\epsilon$ 被挪到了一边，它不带任何参数，梯度直接绕过它就行。于是"采样"被改写成了一个对 $\mu$、$\sigma$ 完全可导的式子，反向传播畅通无阻。用 PyTorch 写就三行：

```python
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)        # 由 log 方差还原出标准差 σ
    eps = torch.randn_like(std)          # 旁路：从标准正态抽噪声 ε
    return mu + std * eps                # 拼装：z = μ + σ·ε，对 μ、σ 可导
```

这个小技巧第一次见会觉得绕，但它正是 VAE 能端到端训练的命门，值得多看两遍。

### KL 散度：把每团云都拉回原点附近

光让编码器输出分布还不够。如果不加约束，网络会偷偷把每团"云"缩得极小、推得极远，退化回普通自编码器的那些孤立点，前功尽弃。所以我们要再加一项约束：**逼着每个输入对应的分布，都尽量靠近一个标准正态分布 $\mathcal{N}(0,1)$。**

衡量"两个分布差多远"的工具，叫 **KL 散度（KL divergence）**。你现在只需记住它的直觉：**KL 散度越大，说明两个分布差得越远；越小，说明越像**（注意它不对称，不是真正的"距离"，但当成差异度量来用足够了，详见 [3.3 概率与统计](../ch1-intro/1-3-3-probability-statistics.md)）。我们把"编码器输出的分布"和"标准正态"之间的 KL 散度也加进损失里，就能把所有的云都温柔地拉回原点附近、让它们彼此重叠。

于是 VAE 的总损失由两项拔河组成：

$$L_{\text{VAE}}=\underbrace{L_{\text{rec}}}_{\text{还原得像}}+\underbrace{\beta\cdot D_{\text{KL}}\big(q(z\mid x)\,\|\,\mathcal{N}(0,1)\big)}_{\text{隐空间要规整}}.$$

这两项在互相拉扯：重构损失想让每个点各就各位、还原得越准越好；KL 项却想把所有点都挤向原点、让隐空间规规矩矩。最后训练出的平衡点，既能较好地还原，又有一个连续、能采样的隐空间——这正是我们想要的。前面那个系数 $\beta$ 是个旋钮：调大它，隐空间更整齐但还原更糊；调小，还原更清晰但隐空间更松散。

## 它能拿来干什么

讲了这么多原理，自编码器在竞赛和实际里到底有什么用？主要是三件事。

**第一，压缩与降维。** 瓶颈向量本身就是输入的一份浓缩表示，维度远低于原始数据。你可以拿它当特征喂给别的模型，或者干脆用来做数据可视化——把图像压到 2 维隐空间画出来，相似的图会自动聚到一起。

**第二，去噪。** 训练时故意给输入加噪声、却让网络去还原**干净**的原图，它就被逼着学会"什么是信号、什么是噪声"，从而具备去噪能力。这种变体叫**去噪自编码器（denoising autoencoder）**。

**第三，异常检测（anomaly detection），这是竞赛里最常考的用法。** 道理很巧妙：如果你只用"正常"数据训练自编码器，它就只学会了还原正常样本。等来了一个异常样本（比如一张不合群的图、一段反常的传感器读数），网络从没见过这种模式，**重构损失会异常地高**。于是你只要盯着重构误差，超过某个阈值就报警，完全不需要任何异常样本的标签——这在欺诈检测、工业质检、医疗筛查里都极其实用。

## 容易踩的坑

- **瓶颈太宽 = 白学**：如果隐空间维度接近甚至等于输入维度，网络会直接"抄答案"，重构损失低得漂亮却什么也没学到。瓶颈一定要够窄，逼它做取舍。
- **VAE 忘了重参数化**：直接写 `z = torch.normal(mu, std)` 会让梯度断在采样这一步，编码器永远训不动。必须用 `mu + std * eps` 的写法把随机性挪到旁路。
- **方差直接输出会爆**：方差必须非负，网络却可能输出负数。标准做法是让网络输出**对数方差** `logvar`，再用 `exp(0.5*logvar)` 还原标准差，天然保证为正、数值也稳。
- **KL 权重失衡**：$\beta$ 给太大，模型会"摆烂"——把所有输入都编码成原点附近的同一团云，还原出来千篇一律（这叫 posterior collapse）；给太小又退化回普通 AE。训练初期常用 KL 退火，让 $\beta$ 从 0 慢慢升上来。
- **异常检测的阈值要在正常数据上定**：重构误差的报警阈值，应该用一批正常样本的误差分布来确定（比如取 95 分位数），而不是拍脑袋随便选一个数。

## 它在后面会怎么用到

- VAE 是最早的一类**深度生成模型**，它和 GAN、扩散模型同属"教模型凭空造数据"这条线，三者的对比和取舍见 [3.5 生成模型 GAN·扩散](3-3-5-generative.md)。
- 自编码器"自己监督自己"的思路，正是**自监督学习**的源头，在视觉里的现代版本（如掩码重建）见 [3.3 数据增强与自监督](3-3-3-augmentation-ssl.md)。
- 编码器产出的瓶颈向量，本质上就是一种**嵌入表示**，和 [1.2 嵌入表示](3-1-2-embeddings.md) 讲的是同一件事的不同侧面。
- VAE 用到的 KL 散度、高斯分布、采样这些概率工具，根子都在 [3.3 概率与统计](../ch1-intro/1-3-3-probability-statistics.md)。

## 练习

??? note "基础练习"
    1. 把上面那个 MNIST 自编码器的瓶颈维度从 32 依次改成 2、8、64，各训一遍，对比还原图的清晰度。你会直观感受到"瓶颈越窄、还原越糊"这条规律。
    2. 用纯文字解释：为什么普通自编码器不能像 VAE 那样"随机生成一张新数字"？它的隐空间缺了什么？
    3. 手算一下重参数化：若 $\mu=2$、$\sigma=0.5$、抽到的噪声 $\epsilon=-1$，那么 $z$ 等于多少？

??? note "进阶练习"
    1. 把普通自编码器改造成**去噪自编码器**：训练时给输入加上高斯噪声，但让损失对照**干净**的原图算。对比它和普通版在含噪输入上的还原效果。
    2. 实现一个完整的 VAE：编码器输出 `mu` 和 `logvar`，用重参数化采样 $z$，损失写成"重构损失 + KL 项"。训练后从标准正态里随机采样几个 $z$ 喂给解码器，看它能不能生成像样的数字。
    3. 做一个**异常检测器**：只用 MNIST 里的数字"1"训练自编码器，然后把数字"8"喂进去，比较两者的重构误差，验证异常样本的误差确实明显更高。

## 小结

- 自编码器是个对称沙漏：编码器压成瓶颈、解码器再还原，**用输入自己当答案**算重构损失，所以无需标签也能学。
- 那个又窄又小的**瓶颈是灵魂**——它逼网络丢掉噪声、留住主干，这才是它学到有用表示的根本原因。
- **VAE** 把"压成一个点"改成"压成一团分布"，靠**重参数化**让采样可导、靠 **KL 散度**把隐空间拉规整，从而获得了"凭空生成"的能力。
- 实战三大用途：压缩降维、去噪、以及竞赛里最爱考的**异常检测**（盯着重构误差报警）。

想把 VAE 的直觉补牢，强烈推荐看 [Lilian Weng 的《From Autoencoder to Beta-VAE》](https://lilianweng.github.io/posts/2018-08-12-vae/)，它把这条脉络梳理得非常清楚；想读原始论文，就看 Kingma & Welling 2013 的 *Auto-Encoding Variational Bayes*。
