# 4.5 音频模型

> **难度** ⭐⭐⭐☆☆ · **前置**：[1.2 嵌入表示](3-1-2-embeddings.md)、[2.2 Transformer 架构](3-2-2-transformer.md)

!!! abstract "读完这一节，你会"
    - 明白声音是怎么从"一段波形"变成模型能吃的频谱图、再变成 MFCC 特征的
    - 理解 HuBERT 这类自监督音频编码器在预训练时到底学了什么，以及它和 BERT 的相似之处
    - 会用预训练编码器搭一个简单的**音频分类**模型，并跑通 Whisper 做**语音转文字**
    - 能识别采样率、长度补齐、特征泄漏这几个音频任务里最容易出错的地方

## 声音进电脑，第一步是把它"画"出来

我们在前面几节里处理的都是图像和文字，这一节换个新对象：声音。但你会发现，套路其实没变——**只要能把声音变成一个规整的数字矩阵，后面那套 Transformer、注意力、预训练编码器就能原封不动地搬过来用。** 所以这一节真正的难点不在模型，而在"声音怎么变成矩阵"这第一步。

声音本身是空气的振动，麦克风把它记录成一条随时间起伏的曲线，叫**波形（waveform）**。计算机每隔极短的时间就采一个点，比如每秒采 16000 次（这个数字叫**采样率**，记作 16 kHz），于是一秒的声音就成了 16000 个数。问题来了：这么一长串原始数字，既冗长又看不出规律，直接喂给模型很难学。

人耳其实不是这么"听"声音的。我们听到的是**音高（频率）**——同一时刻里，低音、中音、高音各占多少能量。于是有了一个关键操作：把波形切成一小段一小段（比如每 25 毫秒一段），对每一段做**傅里叶变换**，算出它包含哪些频率、每个频率多强。把这些结果按时间排成一列列，就得到了一张二维的"图"——横轴是时间、纵轴是频率、颜色深浅代表能量大小。这张图就叫**频谱图（spectrogram）**。

<div class="diagram">
<svg viewBox="0 0 380 210" font-family="-apple-system, 'Segoe UI', sans-serif">
  <defs>
    <marker id="aa" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="var(--dia-accent)"/></marker>
  </defs>
  <!-- 左：波形 -->
  <text x="70" y="22" text-anchor="middle" font-size="12" fill="var(--dia-stroke-soft)">波形（1 维）</text>
  <path d="M16 110 Q26 70 36 110 T56 110 T76 110 T96 110 T116 110" fill="none" stroke="var(--dia-blue)" stroke-width="1.6"/>
  <line x1="14" y1="150" x2="126" y2="150" stroke="var(--dia-stroke-tertiary)" stroke-width="1"/>
  <text x="70" y="166" text-anchor="middle" font-size="10" fill="var(--dia-stroke-tertiary)">时间 →</text>
  <!-- 箭头 -->
  <line x1="134" y1="110" x2="172" y2="110" stroke="var(--dia-accent)" stroke-width="1.8" marker-end="url(#aa)"/>
  <text x="153" y="100" text-anchor="middle" font-size="9" fill="var(--dia-accent)">分帧+FFT</text>
  <!-- 右：频谱图（格子） -->
  <text x="290" y="22" text-anchor="middle" font-size="12" fill="var(--dia-stroke-soft)">频谱图（2 维）</text>
  <g>
    <rect x="190" y="34" width="22" height="22" fill="var(--dia-accent-deep)"/>
    <rect x="212" y="34" width="22" height="22" fill="var(--dia-stroke-tertiary)"/>
    <rect x="234" y="34" width="22" height="22" fill="var(--dia-accent)"/>
    <rect x="256" y="34" width="22" height="22" fill="var(--dia-bg-deep)"/>
    <rect x="278" y="34" width="22" height="22" fill="var(--dia-accent)"/>
    <rect x="300" y="34" width="22" height="22" fill="var(--dia-stroke-tertiary)"/>
    <rect x="190" y="56" width="22" height="22" fill="var(--dia-stroke-tertiary)"/>
    <rect x="212" y="56" width="22" height="22" fill="var(--dia-accent)"/>
    <rect x="234" y="56" width="22" height="22" fill="var(--dia-bg-deep)"/>
    <rect x="256" y="56" width="22" height="22" fill="var(--dia-accent-deep)"/>
    <rect x="278" y="56" width="22" height="22" fill="var(--dia-stroke-tertiary)"/>
    <rect x="300" y="56" width="22" height="22" fill="var(--dia-accent)"/>
    <rect x="190" y="78" width="22" height="22" fill="var(--dia-bg-deep)"/>
    <rect x="212" y="78" width="22" height="22" fill="var(--dia-stroke-tertiary)"/>
    <rect x="234" y="78" width="22" height="22" fill="var(--dia-accent)"/>
    <rect x="256" y="78" width="22" height="22" fill="var(--dia-stroke-tertiary)"/>
    <rect x="278" y="78" width="22" height="22" fill="var(--dia-accent-deep)"/>
    <rect x="300" y="78" width="22" height="22" fill="var(--dia-bg-deep)"/>
  </g>
  <text x="246" y="118" text-anchor="middle" font-size="10" fill="var(--dia-stroke-tertiary)">时间 →</text>
  <text x="180" y="70" text-anchor="middle" font-size="10" fill="var(--dia-stroke-tertiary)" transform="rotate(-90 180 70)">频率 ↑</text>
  <text x="246" y="158" text-anchor="middle" font-size="11" fill="var(--dia-stroke-soft)">每个格子 = 某时刻某频率的能量</text>
</svg>
</div>
<p class="figure-caption">图 4.5：把一维波形分成小段做傅里叶变换，就得到二维频谱图——之后它就能像一张图片一样被模型处理。</p>

到这一步你应该恍然大悟：**频谱图本质上就是一张"图片"。** 一张灰度图也是"横纵两个轴 + 每个像素一个强度值"，频谱图也是。所以处理图像的卷积网络、处理序列的 Transformer，都能直接用来处理声音——这正是音频任务能搭上深度学习这趟车的根本原因。

## 频谱图和 MFCC：两种最常用的声音特征

频谱图已经很好用了，但它有个小毛病：人耳对频率的感知不是线性的。低音区里相差 100 Hz 我们听得很清楚，高音区里相差 100 Hz 几乎听不出来。如果直接用线性频率轴，模型就会在我们根本分不清的高频上浪费大量精力。

为了贴合人耳，人们把频率轴换成一种叫 **Mel（梅尔）刻度**的非线性刻度——低频拉得密、高频压得疏。在 Mel 刻度上算出来的频谱图，叫 **Mel 频谱图（Mel-spectrogram）**，它是今天绝大多数音频深度学习模型（包括 Whisper）真正的输入。

那 **MFCC** 又是什么？它的全名是"梅尔频率倒谱系数（Mel-Frequency Cepstral Coefficients）"，你可以把它理解成 Mel 频谱图的"压缩浓缩版"。在做法上，它在 Mel 频谱图之后又做了一次变换（取对数再做离散余弦变换），把每一帧浓缩成十几个最有代表性的数字。在深度学习兴起之前，MFCC 是语音识别的绝对主力；如今深度模型更偏爱信息更全的 Mel 频谱图，但 MFCC 在算力有限、或者做传统机器学习音频分类时，依然是又轻又好用的特征。

记住这条递进关系就够了：**波形 → 频谱图 → Mel 频谱图 → MFCC**，越往后越"浓缩"、越贴合人耳，但也越丢细节。

我们用 `torchaudio` 把这条链子走一遍，亲眼看看每一步的形状：

```python
import torchaudio
import torchaudio.transforms as T

# 读入一段音频，wav 形状是 (声道数, 采样点数)，sr 是采样率
wav, sr = torchaudio.load("speech.wav")
print(wav.shape, sr)            # 例如 torch.Size([1, 32000]) 16000

# 第一步：算 Mel 频谱图。n_mels=64 表示纵轴分成 64 个频率档
mel = T.MelSpectrogram(sample_rate=sr, n_fft=400,
                       hop_length=160, n_mels=64)(wav)
print(mel.shape)               # (1, 64, 帧数)，这就是一张"图片"

# 第二步：在它基础上再算 MFCC，每帧浓缩成 13 个系数
mfcc = T.MFCC(sample_rate=sr, n_mfcc=13)(wav)
print(mfcc.shape)              # (1, 13, 帧数)，明显更"瘦"
```

跑完你会看到，Mel 频谱图的纵轴有 64 档，而 MFCC 只剩 13 个系数——这就是"浓缩"在形状上的直观体现。横轴的"帧数"取决于音频多长、`hop_length`（每帧之间挪多远）多大，这一点后面补齐长度时要特别小心。

## HuBERT：让模型像 BERT 读句子那样"听"声音

有了频谱图，我们当然可以直接拿带标签的数据从头训练。但语音数据的"转写标签"非常贵——要请人把每一秒声音听写成文字。相比之下，**没有标签的语音录音却多得是**。能不能像文字领域的 BERT 那样，先在海量无标签语音上做自监督预训练，学出一个好用的编码器，再用少量标注数据微调？这正是 **HuBERT** 要解决的问题。

回忆一下 BERT 是怎么学的（见 [4.2 语言建模](3-4-2-language-modeling.md)）：把句子里的一些词盖住，让模型根据上下文猜被盖住的是哪个词。HuBERT 几乎照搬了这个思路，只把对象从"词"换成了"声音片段"。

但这里有个麻烦：文字天然是离散的，"苹果"就是词表里的一个确定编号，盖住后有明确答案可以对照。声音却是连续的波形，没有现成的"答案标签"。HuBERT 的巧办法是：**先用聚类（k-means，见 [1.8 无监督学习](../ch2-round1/2-1-8-unsupervised.md)）给每一小段声音强行分配一个"伪标签"。** 比如把所有声音片段聚成 100 类，每段就贴上 0–99 中的一个号——这就人造出了一套"声音的词表"。有了伪标签，剩下的就和 BERT 一模一样了：随机盖住一部分音频帧，让模型根据没被盖住的上下文，去预测被盖住那些帧属于哪个聚类编号。

为了让你看清两者的对应关系：

| 任务 | BERT（文字） | HuBERT（语音） |
| --- | --- | --- |
| 输入单元 | 一个个词（token） | 一帧帧声音特征 |
| 盖住后要猜的目标 | 被盖词的真实词编号 | 被盖帧的聚类伪标签 |
| 学出来的东西 | 通用文本编码器 | 通用语音编码器 |

预训练完成后，HuBERT 就成了一个强大的**音频编码器**：你喂给它一段波形，它吐回一串富含语义和发音信息的向量（也就是音频的**嵌入表示**，概念见 [1.2 嵌入表示](3-1-2-embeddings.md)）。后面无论是做分类还是识别，都站在这串向量的肩膀上，而不必从零学起。顺带一提，它的"前辈" **wav2vec 2.0** 思路非常接近，你在文献里会经常看到它俩一起出现。

## 用预训练编码器做音频分类

讲完原理，我们落到一个最常见的任务上：**音频分类**——给一段声音，判断它属于哪一类。它可能是"这段是猫叫、狗叫还是鸟叫""说话人的情绪是开心还是生气""这个口令是开灯还是关灯"。

套路和图像迁移学习（见 [3.3 迁移学习与预训练编码器](../ch2-round1/2-3-3-transfer-learning.md)）一模一样，分三步：

**第一步，用预训练的 HuBERT 把整段音频编码成一串帧向量。** 这一步借的是别人花海量算力学好的"耳朵"。

**第二步，把这一串向量压成一个固定长度的整体表示。** 最简单的办法是沿时间轴做平均池化——把所有帧的向量取个平均，得到一个能代表"整段声音"的向量。

**第三步，在上面接一个小小的分类头**（就是一层全连接 + softmax），输出每个类别的概率。

下面用 Hugging Face 的 `transformers` 把这套流程跑通。注意我们这里**冻结**了 HuBERT 主体、只训练最后那个分类头，这在数据不多时又快又稳：

```python
import torch
from transformers import AutoFeatureExtractor, AutoModel

# 加载预训练 HuBERT 编码器和它配套的特征提取器
name = "facebook/hubert-base-ls960"
extractor = AutoFeatureExtractor.from_pretrained(name)
encoder = AutoModel.from_pretrained(name)
encoder.eval()                                  # 冻结主体，不更新它的参数

def embed(wav, sr=16000):
    """把一段 16kHz 波形编码成一个固定长度的向量。"""
    inputs = extractor(wav, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():                       # 不算梯度，省显存也更快
        out = encoder(**inputs).last_hidden_state   # (1, 帧数, 768)
    return out.mean(dim=1)                       # 沿时间轴平均 -> (1, 768)

# 在编码器之上接一个轻量分类头（这里假设 5 个类别）
classifier = torch.nn.Linear(768, 5)
feat = embed(torch.randn(16000))                 # 假装这是 1 秒音频
logits = classifier(feat)
print(logits.shape)                              # torch.Size([1, 5])
```

训练时，你只需要把真实标签和 `logits` 丢进交叉熵损失，反向传播去更新 `classifier` 这一层就行。你会惊讶于：哪怕每类只有几十条样本，靠 HuBERT 这只现成的"好耳朵"，分类准确率也能相当高——**这就是预训练编码器的威力，它把"听懂声音"这件难事提前替你做完了。**

## Whisper：一句话把语音变成文字

最后看一个你天天都在用的任务：**语音转文字**，正式名叫**自动语音识别（Automatic Speech Recognition, ASR）**——手机语音输入、视频自动字幕，背后都是它。这个领域现在的当红模型是 OpenAI 的 **Whisper**。

Whisper 的结构其实就是我们在 [4.3 Encoder-Decoder](3-4-3-encoder-decoder.md) 里学过的那套**编码器-解码器 Transformer**，只不过输入端换成了声音：

- **编码器**吃进音频的 Mel 频谱图（还记得吗，它就是一张"图片"），把声音理解成一串向量；
- **解码器**像写文章一样，一个字一个字地把对应的文字"生成"出来。

把它和我们之前学的机器翻译对照一下，你会发现它们是同一个模子：翻译是"中文序列进、英文序列出"，而 ASR 不过是"声音序列进、文字序列出"——任务变了，骨架没变。Whisper 真正厉害的地方在于它用了海量（68 万小时）多语种数据训练，所以拿来即用、几乎不用微调就很准，还顺手支持几十种语言和翻译。

用起来也简单得出奇，几行就能把一段录音转成文字：

```python
from transformers import pipeline

# 直接用 pipeline 封装好的语音识别管线
asr = pipeline("automatic-speech-recognition",
               model="openai/whisper-small")

result = asr("speech.wav")     # 传入音频文件路径
print(result["text"])          # 打印识别出的文字
```

就这么几行，背后却完整地走了一遍"波形 → Mel 频谱图 → 编码器 → 解码器逐字生成"的全流程。把这一节从头串起来你会发现：**我们前面学的注意力、Transformer、编码器-解码器、预训练，全都在 Whisper 这一个模型里复用了——声音不过是它们的又一个应用场景而已。**

## 容易踩的坑

- **采样率必须对齐**：几乎所有语音预训练模型（HuBERT、Whisper）都假定输入是 16 kHz。如果你的音频是 44.1 kHz 还直接喂进去，模型听到的就是"变调加速"的怪声，结果全错。读入后一定先 `torchaudio` 重采样到 16 kHz。
- **一个 batch 里长度要补齐**：每段音频长短不一，帧数也就不同，没法直接堆成一个张量。要么补零到同一长度（padding），要么裁剪；而且要配一个 mask 告诉模型哪些是补出来的、别当真。
- **MFCC/频谱图的标准化也会泄漏**：和表格数据一样，算均值方差只能用训练集，再套到验证、测试集上，否则就偷看了答案（同 [4.3 Pandas 与数据处理](../ch1-intro/1-4-3-pandas-data.md) 里讲的数据泄漏）。
- **别把编码器和分类头的学习率设成一样**：微调时预训练主体要用很小的学习率（甚至先冻结），分类头可以大一些；一上来就用大学习率整体训练，很容易把预训练学到的好特征"冲垮"。
- **单声道 vs 立体声**：模型一般只吃单声道。立体声有两个声道，记得先取一个声道或求平均，否则形状对不上。

## 它在后面会怎么用到

- 这一节把"声音变成矩阵"之后的所有处理，都建立在你学过的 [2.2 Transformer 架构](3-2-2-transformer.md) 和 [2.1 注意力机制](3-2-1-attention.md) 之上——音频只是它们的又一个落脚点。
- HuBERT 的自监督预训练思路，和图像里的 [3.3 数据增强与自监督](3-3-3-augmentation-ssl.md)、文字里的 [4.2 语言建模](3-4-2-language-modeling.md) 是同一个大家族，三者对照着看会更通透。
- Whisper 的编码器-解码器结构直接复用了 [4.3 Encoder-Decoder](3-4-3-encoder-decoder.md)；到了 Camp 阶段的 [2.3 信号·优化类真题](../ch4-camp/4-2-3-signal-opt-problems.md)，音频信号处理还会再登场。

## 练习

??? note "基础练习"
    1. 用 `torchaudio` 读入任意一段 wav，分别算出它的 Mel 频谱图和 MFCC，打印两者的形状，并说说为什么 MFCC 的纵轴更"瘦"。
    2. 把一段 44.1 kHz 的音频用 `torchaudio.transforms.Resample` 重采样到 16 kHz，对比重采样前后采样点数的变化，想一想为什么语音模型都要求 16 kHz。
    3. 用上面的 `embed` 函数把两段不同的声音各编码成一个 768 维向量，算它们的余弦相似度，体会"相似的声音向量也相近"。

??? note "进阶练习"
    1. 找一个小的音频分类数据集（比如语音指令 Speech Commands），用冻结的 HuBERT + 一个线性分类头训练，报告测试准确率；再试试解冻 HuBERT 用很小的学习率微调，看准确率涨了多少。
    2. 亲手实现 HuBERT 预训练的"伪标签"那一步：对一堆音频帧向量做 k-means 聚类，给每帧打上聚类编号，再随机盖住 15% 的帧，搭一个让模型预测被盖帧编号的小任务，体会自监督是怎么"无中生有"造出监督信号的。
    3. 用 Whisper 给一段带噪声的录音转写，再用 [1.4 分类评估指标](../ch2-round1/2-1-4-classification-metrics.md) 的思路，按字错率（WER）评估它的准确度，看看噪声让错误率涨了多少。

## 小结

- 处理声音的第一步永远是**把波形变成频谱图**——一旦变成这张"图片"，图像和序列模型就能照搬过来用。
- 记住这条浓缩链：**波形 → 频谱图 → Mel 频谱图 → MFCC**，越往后越贴合人耳、越省算力，但也越丢细节。
- **HuBERT** 把 BERT 的"完形填空"搬到了语音上，靠聚类伪标签在海量无标签录音里自监督学出通用音频编码器；**Whisper** 则是一个直接拿来就能用的编码器-解码器 ASR 模型。
- 实操上最该盯紧的三件事：**采样率对齐到 16 kHz、batch 内补齐长度、标准化别泄漏**。

想动手，Hugging Face 的 [Audio Course](https://huggingface.co/learn/audio-course) 是最系统也最友好的中文/英文入门，从频谱图一路讲到 Whisper 微调，强烈推荐跟着把代码敲一遍。
