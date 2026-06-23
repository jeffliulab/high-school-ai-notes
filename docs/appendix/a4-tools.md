# A4 环境与工具

> **难度** ⭐⭐☆☆☆ · **前置**：[4.5 Colab 与 Markdown](../ch1-intro/1-4-5-colab-markdown.md)

!!! abstract "本页目标"
    - 选对适合自己的练习环境：什么时候用 Colab，什么时候在本机装 PyTorch
    - 记住几条最常用的安装、查环境、查显存命令，遇到报错知道从哪儿查起
    - 手上有一份"赛前一晚就能照着过一遍"的环境准备清单，进考场不慌

## 先想清楚：你到底需要一个什么样的环境

很多同学一上来就纠结"该用哪个工具"，其实顺序反了。先问自己一个更实在的问题：**这一步我是在学新知识、跑小实验，还是要训练一个吃显卡的大模型？** 不同阶段的需求差很远，硬要用一套环境从头扛到尾，往往两头不讨好。

刚开始学经典机器学习（Round 1 那些表格题、sklearn 的模型），数据小、算得快，一台普通笔记本的 CPU 就完全够用。可到了 Round 2，要训卷积网络、要微调 Transformer，没有 GPU 你可能跑一个 epoch 就得等上半小时——这时候就该把活儿挪到带显卡的环境里去。所以"用什么"这件事，本质上是被"算什么"决定的。

下面这张表帮你快速对号入座：

| 你想干的事 | 推荐环境 | 为什么 |
| --- | --- | --- |
| 学语法、跑小例子、做表格题 | 本机 Jupyter / VS Code | 启动快、不依赖网络、文件就在手边 |
| 临时要个免费 GPU 跑深度学习 | Google Colab | 零配置、白嫖一块 GPU，开浏览器就能用 |
| 长期、可复现地训练 | 本机 conda 环境 + 自己的 GPU | 环境固定、不会被云端随时断线 |
| 团队协作 / 提交可复现脚本 | requirements.txt + 固定随机种子 | 别人照着装就能复现你的结果 |

## Colab：先把这块免费 GPU 用明白

Google Colab 是入门阶段性价比最高的选择——它本质上是一个跑在云端的 Jupyter，最大的好处是**免费送你一块 GPU**，而且什么都不用装，打开浏览器、登录谷歌账号就能写代码。对还没有独立显卡的同学来说，这几乎是练深度学习的唯一门槛级方案。

不过要真正用上 GPU，得先手动切换一下。菜单里点 **代码执行程序 → 更改运行时类型 → 硬件加速器选 GPU**，否则你默认还是在 CPU 上跑。切好之后，写代码前先用一行确认它真的认到了卡：

```python
import torch

print(torch.cuda.is_available())          # True 才说明 GPU 接上了
print(torch.cuda.get_device_name(0))      # 看看分到的是哪块卡（常见是 T4）
```

如果第一行打印出 `False`，那不管你后面 `.cuda()` 写得多顺，模型还是在 CPU 上慢慢爬——**这是新手最常忽略的一步，养成跑实验前先确认环境的习惯，能帮你省下大把"为什么这么慢"的困惑。**

Colab 还有两个坑得提前知道。一是它会**自动断开**：闲置太久或单次连太长就被回收，没存的东西全没了，所以重要数据和模型权重要及时存到 Google Drive。二是每次新开会话，环境都是干净的，你额外装的库（`!pip install ...`）下次得重装。

## 在本机装 PyTorch：别凭感觉乱装

到了要长期训练、或者想摆脱"被云端随时踢下线"的阶段，你会想在自己电脑上配一套固定环境。这里最容易出问题的就是 PyTorch 的安装——因为它要和你显卡的 CUDA 版本对上，凭感觉 `pip install torch` 很可能装到一个跑不动 GPU 的版本。

正确的做法是**去官网拿对应你系统的安装命令**。PyTorch 官网（[pytorch.org](https://pytorch.org/get-started/locally/)）有一个交互选择器，你勾选操作系统、包管理器、CUDA 版本，它就生成那行专属命令。例如有 GPU 的 Linux 机器，命令长这样：

```bash
# CPU 版（没有独立显卡，或只想跑小实验）
pip install torch torchvision

# GPU 版（要指定和驱动匹配的 CUDA，下面 cu121 = CUDA 12.1）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

强烈建议**先建一个独立环境再装**，别把所有库都堆在系统全局里——不然不同项目的版本会互相打架。用 conda 是这样：

```bash
conda create -n ai python=3.11   # 建一个叫 ai 的独立环境
conda activate ai                # 进去
pip install torch torchvision    # 在里面装，污染不到别的项目
```

装完照例验证一句，确认 PyTorch 认到了显卡，再开始正式工作。

## 几条天天会用到的命令

工具配好了，下面这几条命令你几乎每次开工都会敲到，先混个眼熟，遇到问题也好对症下药：

```bash
nvidia-smi          # 看 GPU 利用率、显存占用、当前是谁在用卡
pip list            # 列出当前环境装了哪些库、各是什么版本
pip freeze > requirements.txt   # 把环境快照存下来，方便别人复现
python -c "import torch; print(torch.__version__)"   # 快速查 PyTorch 版本
```

其中 `nvidia-smi` 最值得记住。训练时如果它显示显存几乎满了，多半要把 batch size 调小；如果 GPU 利用率长期是 0%，那说明你的代码根本没用上卡，得回头检查是不是忘了 `.to(device)`。

调试时还有两个轻量手段特别实用。一是直接 `print` 张量的形状——深度学习里十有八九的报错都是**维度对不上**，`print(x.shape)` 一打，问题往往立刻现形。二是在 Jupyter 里把代码切成小格子（cell）逐块运行，哪块出错就停在哪块，比一口气跑整个脚本好定位得多。

## 进考场前：照着这份清单过一遍

赛前最忌讳的就是"临场才发现环境没配好"。把下面这份清单当成出发前的检查表，**比赛前一晚照着从头走一遍**，每一项都确认通过，进场就能把精力全放在题目上：

- [ ] 能打开比赛指定的环境（Colab 或本地），并成功跑通一个"Hello World"格子；
- [ ] `torch.cuda.is_available()` 返回 `True`，确认 GPU 真的可用；
- [ ] 常用库都在且版本正常：`numpy`、`pandas`、`scikit-learn`、`matplotlib`、`torch`、`torchvision`；
- [ ] 知道怎么上传 / 读取数据文件（Colab 挂载 Drive，或本地放对路径）；
- [ ] 准备好一段"固定随机种子"的代码，保证结果可复现（很多赛事的评分要求复现）；
- [ ] 清楚提交格式：要交的是 `.ipynb`、`.py` 脚本，还是预测结果 `.csv`；
- [ ] 手边留一份 `requirements.txt` 或安装命令，万一环境崩了能快速重建。

关于"固定随机种子"，这里给一段可以直接抄进 notebook 第一格的模板：

```python
import torch, numpy as np, random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)   # 多卡也一起固定
```

把它放在最前面跑一次，你每次重跑的结果就基本一致了——这在调参对比和复现要求里非常关键。

## 容易踩的坑

- **以为切了 GPU 就一定在用**：Colab 选了 GPU 运行时，代码里若没把模型和数据 `.to("cuda")`，照样跑在 CPU 上。每次开工先验一句 `torch.cuda.is_available()`。
- **`pip install torch` 装到 CPU 版**：不按官网选择器给的命令装，很可能拿到一个不带 CUDA 的轮子，怎么调都用不上显卡。
- **不建独立环境**：所有库堆在系统全局里，迟早不同项目版本互相冲突。养成一个项目一个 conda / venv 环境的习惯。
- **Colab 断线丢东西**：闲置被回收后未保存的全没了，重要数据和权重要随手存到 Drive。
- **忘了固定随机种子**：结果每次都变，调参时根本分不清是改动起了作用还是随机波动，复现要求也过不了。

## 它在后面会怎么用到

这页是"工具底座"，几乎贯穿整本笔记。真正大量吃 GPU、要配好环境的，是从 [1.1 GPU 训练](../ch3-round2/3-1-1-gpu-training.md) 开始的 Round 2 内容；而你练手用的 PyTorch 写法，会在 [2.4 PyTorch 基础](../ch2-round1/2-2-4-pytorch-basics.md) 里系统学到。到了国际赛阶段，[1.2 规则与工具](../ch5-champion/5-1-2-ioai-rules.md) 会明确规定考场允许用哪些环境和库，本页的清单正好和它对照着看。需要找数据集和更多资料时，再回头看 [A1 资料中心](a1-resources.md) 和 [A2 真题与数据集](a2-datasets.md)。

## 练习

??? note "基础练习"
    1. 在 Colab 上新建一个 notebook，切换到 GPU 运行时，写三行代码确认 `torch.cuda.is_available()` 为 `True` 并打印出分到的显卡型号。
    2. 在本机建一个名为 `ai` 的 conda 环境，装上 PyTorch，再用 `python -c "import torch; print(torch.__version__)"` 验证版本。

??? note "进阶练习"
    1. 把上面"固定随机种子"的模板抄进 notebook，连续跑两次同一段训练代码，确认两次的初始损失值完全一样。
    2. 给你的环境生成一份 `requirements.txt`，删掉环境重建一个，用这份文件一键还原，体会一下"可复现"是怎么做到的。

## 小结

一句话：**工具是为算力需求服务的——学语法用本机 Jupyter，缺 GPU 找 Colab，要长期训练就在本机配独立 conda 环境装对版本的 PyTorch。** 剩下的就是养成两个好习惯：开工先验环境，赛前照清单过一遍。

想深入了解安装细节，认准 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 的命令选择器；Colab 的使用技巧可看 [Colab 官方欢迎教程](https://colab.research.google.com/notebooks/intro.ipynb)。
