# 第 4 章 大模型核心算子全景与 GPU 加速原理

> **本章情绪定心丸**：如果把大模型比作一座宏伟的乐高城堡，那么**算子（Operator / Kernel）**就是搭建城堡的那七种标准积木块。不管模型有 6 亿还是 6000 亿参数，不管是语音模型 Qwen3-ASR、文本模型 DeepSeek，还是视觉模型 Qwen-VL，把它们拆到最底层，反反复复跑的也就是这 7 类算子！读完这一章，你不仅知道每个算子的公式，还能看穿为什么 FlashAttention 和 Triton 算子融合能把速度直接拉高 3–5 倍。

---

## 4.1 算子全景图：大模型训练里时间到底花在哪？

把一次完整的 Transformer 前向 + 反向传播放进 NVIDIA Nsight Systems 性能剖析器里，你会看到算子清晰地分为两大阵营：

| 阵营 | 核心算子 | Roofline 属性 | 典型耗时占比 | 优化核心手段 |
| :--- | :--- | :---: | :---: | :--- |
| **矩阵密集型** | `GEMM`（线性层 Q/K/V/O、SwiGLU、lm_head）、`Conv2d` | Compute-Bound（大 Batch） | 65%–80% | Tensor Core、BF16/FP8、分块 Tiling、维度 128 对齐 |
| **访存敏感型** | `Attention`、`RMSNorm/LayerNorm`、`RoPE`、`SiLU/GELU`、`Cross-Entropy`、`AdamW` | Memory-Bound | 20%–35%（长序列可飙升至 70%+！） | **算子融合（Kernel Fusion）**、FlashAttention、Online Softmax、Fused CE、Triton |

下面我们逐一揭开这 7 大算子的数学公式、硬件瓶颈与加速绝活！

---

## 4.2 第一大算子：GEMM 矩阵乘法与分块 Tiling

几乎所有 `nn.Linear` 底层都调用 NVIDIA 的 **cuBLAS / CUTLASS** 执行通用矩阵乘法（General Matrix Multiplication, GEMM）：

$$\mathbf{C}_{M \times N} = \mathbf{A}_{M \times K} \cdot \mathbf{B}_{K \times N}$$

### 为什么直接写三层 `for` 循环在 GPU 上慢得像爬？
如果不做分块，算 $\mathbf{C}$ 里的每一个元素 $C_{ij}$，都要从慢速 HBM 里把 $\mathbf{A}$ 的一整行（$K$ 个数）和 $\mathbf{B}$ 的一整列（$K$ 个数）各读一遍。全矩阵共有 $M \times N$ 个元素，总 HBM 读取量高达 $M N \cdot 2K$ 次——同一个元素被反反复复从大冷库（HBM）搬出来几千次！

### GPU 的加速魔法：分块 Tiling + Shared Memory 复用 + 双缓冲
1. **分块（Tiling）**：把大矩阵切成能刚好塞进片上高速 **SRAM（Shared Memory）** 的小砖块（比如 $128 \times 128$ 块，内循环沿 $K$ 维每次取 $32$ 或 $64$ 宽度的小条）。
2. **一次搬运，重复榨干**：把小块 $\mathbf{A}_{\text{tile}}$ 和 $\mathbf{B}_{\text{tile}}$ 从 HBM 搬进 SRAM 后，这 128 行和 128 列在高速案板上做完整的外积累加——**每个搬进来的数字被复用了 128 次！** 瞬间把 HBM 流量压低 128 倍，算术强度直接冲破 Roofline 拐点。
3. **异步双缓冲（Double Buffering / Software Pipelining）**：A100/H100 支持 `cp.async` 与 TMA（Tensor Memory Accelerator）——在 Tensor Core 正在疯狂计算**第 $k$ 块**的同时，搬运引擎在后台悄悄把**第 $k+1$ 块**提前运进另一半 SRAM！算完第 $k$ 块立刻无缝切换，把 500 周期的 HBM 等待延迟完全隐藏！

::: tip 开发者能直接用上的 GEMM 优化铁律
- **形状对齐**：尽量让隐藏层维度 `hidden_size`、LoRA 秩 `r`（如 `8, 16, 32, 64`）、词表大小（如 Qwen3 的 `151,936 = 1187 × 128`）和 Batch Token 数是 **8、16 或 64 的倍数**，这样 Tensor Core 才能触发最高吞吐指令路径，避免边角料补零浪费。
- **合并 QKV 投影**：很多推理框架把三个独立的 `q_proj`、`k_proj`、`v_proj` 权重拼成一个大矩阵 `qkv_proj`，1 次大 GEMM 替代 3 次小 GEMM，减少 Kernel 发射次数并提高算术强度。
:::

---

## 4.3 第二大算子：Attention 与 FlashAttention（Online Softmax 数学揭秘）

Transformer 最核心的公式是**缩放点积注意力（Scaled Dot-Product Attention, SDPA）**：

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}\right)\mathbf{V}$$

其中 $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{N \times d_k}$，$N$ 是序列长度，$d_k$ 是单头维度（Qwen3-0.6B 为 `128`），$\mathbf{M}$ 是因果掩码（上三角填 $-\infty$，防止偷看未来的词）。为什么要除以 $\sqrt{d_k}$？因为两个标准正态随机向量的点积方差是 $d_k$，如果不除以 $\sqrt{d_k}$，点积值太大（比如 $\pm 30$），送进 Softmax 会瞬间饱和成 `[1, 0, 0, ...]`，梯度几乎全变 0！

### 朴素 Attention 的两大死穴
在 PyTorch 里按公式直译只需三行代码，但它会在 HBM 里生出两张巨大的 $N \times N$ 中间矩阵 $\mathbf{S} = \mathbf{Q}\mathbf{K}^\top / \sqrt{d_k}$ 和 $\mathbf{P} = \text{Softmax}(\mathbf{S})$：

- **显存 $O(N^2)$ 爆炸**：当 $N = 8192$、Batch = 4、Heads = 16 时，光单层存 $\mathbf{S}$ 和 $\mathbf{P}$ 就要吃掉 $4 \times 16 \times 8192^2 \times 2\text{ B} \times 2 \approx \mathbf{17.2\text{ GB}}$！
- **带宽彻底打爆**：为了算这三步，GPU 要把 $N \times N$ 矩阵来回读写 HBM 好几次，明明矩阵乘很快，全卡在读写 $\mathbf{S}$ 和 $\mathbf{P}$ 上！

### FlashAttention：永远不让 $N \times N$ 矩阵落地 HBM！

![图 4-1 朴素 Attention vs FlashAttention：从 O(N²) 显存爆炸到片上 SRAM 分块流式计算](assets/diagrams/07_gemm_tiling_flashattn.svg)

你肯定会立刻发问：**“等等！Softmax 的分母 $\sum_{j=1}^N e^{s_j - m}$ 需要看完整一整行的 $N$ 个数才能知道全局最大值 $m$ 和总和 $l$，如果分块算，怎么可能不先把整行写进显存呢？”**

问到点子上了！斯坦福团队（Tri Dao 等人）正是靠**在线 Softmax（Online Softmax）**这一绝妙的高中数学恒等变换，彻底解决了这个问题！

::: example 小数字手算证明：Online Softmax 凭什么和全局 Softmax 100% 数学等价？
假设某一行完整的注意力打分有 4 个数 $\mathbf{s} = [1.0,\ 2.0,\ 4.0,\ 0.0]$，对应 4 个值标量 $\mathbf{v} = [10,\ 20,\ 30,\ 40]$。

**方法 A（传统全局一次算）**：

- 全局最大值 $m = 4.0$
- 指数项：$[e^{-3}, e^{-2}, e^{0}, e^{-4}] = [0.0498,\ 0.1353,\ 1.0000,\ 0.0183]$，总和 $l = 1.2034$
- 加权输出：
  $$O = \frac{0.0498 \times 10 + 0.1353 \times 20 + 1.0000 \times 30 + 0.0183 \times 40}{1.2034} = \frac{33.936}{1.2034} = \mathbf{28.200}$$

**方法 B（FlashAttention 分两块流式算，每次只看 2 个数！）**：

- **第 1 块** $[1.0, 2.0]$，对应 $v = [10, 20]$：
  - 局部最大 $m_1 = 2.0$
  - 局部指数和 $l_1 = e^{1-2} + e^{2-2} = 0.3679 + 1.0 = 1.3679$
  - 局部未归一分子 $\tilde{O}_1 = 0.3679 \times 10 + 1.0 \times 20 = 23.679$
- **第 2 块** $[4.0, 0.0]$，对应 $v = [30, 40]$：
  - 发现更大值！新全局最大 $m_2 = \max(m_1, \max(4.0, 0.0)) = 4.0$
  - **关键修正因子**：$\alpha = e^{m_1 - m_2} = e^{2.0 - 4.0} = e^{-2} = 0.1353$！（把第 1 块按新基准 $m_2$ 整体缩小一下，高中指数律：$e^{s - m_1} \cdot e^{m_1 - m_2} = e^{s - m_2}$！）
  - 更新分母：$l_2 = \alpha \cdot l_1 + (e^{4-4} + e^{0-4}) = 0.1353 \times 1.3679 + (1.0 + 0.0183) = 0.1851 + 1.0183 = \mathbf{1.2034}$！
  - 更新分子：$\tilde{O}_2 = \alpha \cdot \tilde{O}_1 + (1.0 \times 30 + 0.0183 \times 40) = 0.1353 \times 23.679 + 30.732 = 3.204 + 30.732 = \mathbf{33.936}$！
- **最后统一除以分母**：$O = \tilde{O}_2 / l_2 = 33.936 / 1.2034 = \mathbf{28.200}$！

**精确到小数点后每一位，零近似误差！** 整个过程我们只维护了两个标量 $(m, l)$，根本不需要保存巨大的 $N \times N$ 矩阵！而且反向传播时，FlashAttention 也不存 $\mathbf{P}$，只存每个 Token 一个极小的对数配分标量 $L_i = m_i + \log l_i$，反向时在高速 SRAM 里顺手重算局部块——显存复杂度从 $O(N^2)$ 彻底降维到 **$O(N)$**！
:::

### FlashAttention-1 → 2 → 3 演进与 PyTorch SDPA 调用
- **FlashAttention-1**：开创 IO-Aware Tiling + Online Softmax，HBM 读写从 $\Theta(N^2)$ 降到 $\Theta(N^2 d^2 / M)$（$M$ 是 SRAM 大小）。
- **FlashAttention-2**：把外循环和内循环对调（外循环遍历 $\mathbf{Q}$、内循环遍历 $\mathbf{K},\mathbf{V}$），减少了非矩阵乘的缩放开销，并在序列维度 $N$ 上并行切分给更多 SM，在 A100 上利用率翻倍达到 ~70% 理论峰值。
- **FlashAttention-3**：针对 Hopper（H100）专属优化，利用异步 TMA + WarpGroup MMA 重叠流水线，并原生支持 FP8。
- **在 PyTorch 2.x 中怎么用？** 直接调用官方 `torch.nn.functional.scaled_dot_product_attention`（本项目配置中 `runtime.attention: sdpa`），PyTorch 会根据你的显卡架构和输入 mask 自动分派到 **FlashAttention-2**、**cuDNN Attention** 或 **Memory-Efficient Attention (xFormers)** 后端！

下面这 16 行对比脚本，可以让你在自己的 GPU 上亲眼见证序列长度 `N=4096` 时，FlashAttention/SDPA 比朴素 Attention 省下多少倍显存、快出多少倍：

```python
import time, math, torch
import torch.nn.functional as F

def compare_naive_vs_sdpa(B: int = 2, H: int = 16, N: int = 4096, D: int = 128):
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    for name, fn in [
        ("Naive Attention", lambda: F.softmax((q @ k.transpose(-2, -1)) / math.sqrt(D), dim=-1) @ v),
        ("Flash / SDPA   ", lambda: F.scaled_dot_product_attention(q, k, v, is_causal=False)),
    ]:
        torch.cuda.reset_peak_memory_stats()
        _ = fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            _ = fn()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 100
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"{name} | {ms:6.2f} ms | Peak VRAM: {peak_mb:7.1f} MB")
```

*(推理侧补充：**PagedAttention**——vLLM 提出的显存分页管理算子，把连续的 KV Cache 拆成固定大小的物理块（如每块存 16 个 Token），像操作系统虚拟内存一样消除长短不一请求造成的 60% 显存碎片，大幅提升线上并发量。)*

---

## 4.4 第三、四、五大算子：RMSNorm、SwiGLU 与 RoPE

现代开源大模型（Qwen3、LLaMA-3、DeepSeek、Mistral）不约而同地完成了架构收敛。图 4-2 直观总结了这三大招牌算子：

![图 4-2 现代大模型三大招牌算子：RoPE 旋转位置编码 · SwiGLU 门控网络 · RMSNorm](assets/diagrams/08_rope_swiglu_norm.svg)

### 1. LayerNorm vs RMSNorm vs Qwen3 QK-Norm
深层网络中，激活值经过几十次矩阵乘后很容易变得极大或极小，导致训练失稳。归一化层负责把每一层向量的尺度拉回健康范围。

- **传统 LayerNorm**（Qwen3-ASR 的 18 层 Audio Encoder 仍在使用）：
  $$\text{LayerNorm}(\mathbf{x}) = \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \boldsymbol{\gamma} + \boldsymbol{\beta}, \quad \mu = \frac{1}{d}\sum_{i=1}^d x_i, \quad \sigma^2 = \frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2$$
- **现代 RMSNorm**（Qwen3-0.6B 解码器 28 层全部采用）：
  $$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \odot \boldsymbol{\gamma}$$
  **少做了什么？** 直接删掉了减均值 $\mu$ 和加偏置 $\boldsymbol{\beta}$！研究发现归一化起决定作用的是“缩放不变性”而非“平移不变性”。删掉均值计算，GPU 融合 Kernel 少扫一次数据、少做一次 Warp 归约，**白赚 10%–20% 速度且效果丝毫不降**。

- **Qwen3 专属绝活——Per-Head QK-Norm**：在每层注意力把 `q_proj`、`k_proj` 拆成 128 维的各个头之后、做 RoPE 之前，分别对每个头的 $\mathbf{q}$ 和 $\mathbf{k}$ 做一次 128 维的 RMSNorm（`q_norm` 和 `k_norm`）。它把 $\mathbf{q} \cdot \mathbf{k}$ 的点积幅度牢牢锁在安全区间，彻底消灭了长序列大步长训练时常见的“Attention Logits 爆炸（Loss Spike）”现象。

### 2. ReLU → GELU → SiLU → SwiGLU
- **ReLU**：$\max(0, x)$，负数区导数恒为 0，容易出现神经元永久失活（Dying ReLU）。
- **GELU**（Qwen3-ASR 音频编码器使用）：$x \cdot \Phi(x) \approx 0.5x\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}(x + 0.044715 x^3)\right)\right)$，在 0 附近平滑过渡，负半轴保留柔和梯度。
- **SiLU / Swish**：$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$，公式极简、处处光滑。
- **SwiGLU**（Qwen3 / LLaMA-3 解码器 MLP 标配）：
  $$\text{SwiGLU}(\mathbf{x}) = \left(\text{SiLU}(\mathbf{x}\mathbf{W}_{\text{gate}}) \odot (\mathbf{x}\mathbf{W}_{\text{up}})\right) \mathbf{W}_{\text{down}}$$
  用生活类比：`up_proj` 负责把 1024 维特征放大到 3072 维“知识抽屉”，`gate_proj` 经过 SiLU 充当“动态调光开关”，按当前语境决定每个抽屉开多大缝隙，最后 `down_proj` 汇总回 1024 维。因为多了一个矩阵（3 个替代传统 FFN 的 2 个），大多数模型把中间维度设在约 $\frac{8}{3}H$ 以保持参数量持平（`Qwen3-0.6B` 则慷慨地设为了整数倍 $I = 3072 = 3H$，赋予小模型更强的知识容量）。

### 3. RoPE（旋转位置编码）：高中三角函数的封神之作！
自注意力本身像一个“词袋子”，打乱词序算出来的注意力一模一样。早期 Transformer（以及 Qwen3-ASR 的音频编码器）把固定的正弦波直接**加**在输入向量上（绝对位置编码）；而现代 LLM 全员采用苏剑林提出的 **RoPE（Rotary Position Embedding）**。

核心思想美得像一首诗：把每个 128 维的 Query/Key 向量两两配对，拆成 64 个二维平面小箭头 $(x_{2i}, x_{2i+1})$。第 $m$ 个位置的向量，就把它的第 $i$ 根小箭头逆时针旋转角度 $m \theta_i$（其中频率 $\theta_i = \text{base}^{-2i/d}$，不同维度转得有快有慢，像时钟的秒针、分针和时针）：

$$\begin{pmatrix} q'_{2i} \\ q'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}$$

现在用高中和差化积公式算位置 $m$ 的 $\mathbf{q}'_m$ 和位置 $n$ 的 $\mathbf{k}'_n$ 的点积，神奇的事发生了：

- 旋转矩阵满足 $\mathbf{R}_m^\top \mathbf{R}_n = \mathbf{R}_{n-m}$！
- 所以点积 $(\mathbf{R}_m \mathbf{q})^\top (\mathbf{R}_n \mathbf{k}) = \mathbf{q}^\top \mathbf{R}_{n-m} \mathbf{k}$——**绝对位置 $m$ 和 $n$ 奇迹般地消失了，点积只依赖它们的相对距离 $(n - m)$！**
- **为什么 Qwen3 把 `rope_theta` 从早期的 `10,000` 拉高到 `1,000,000`？** 分母变大后，旋转角速度 $\theta_i$ 变慢，长序列里高频指针不容易转完一圈发生“位置混叠”，从而原生支持超长上下文和长音频转写！

---

## 4.5 第六、七大算子：Conv2d 卷积与 Fused Cross-Entropy（词表显存救星）

### 1. Conv2d 卷积的三种加速路径
在 `Qwen3-ASR-0.6B` 的听觉前端，3 层 `Conv2d(kernel=3, stride=2, padding=1)` 负责把二维梅尔频谱图的高频局部纹理提取出来并做 $8\times$ 降采样。在 GPU（cuDNN）内部，卷积并非笨拙的四层循环，而是按卷积核尺寸自动选择三种引擎：

- **im2col + GEMM**：把滑动窗口展开成矩阵，直接甩给 Tensor Core 跑 GEMM；
- **Winograd 算法**：专门针对 $3 \times 3$ 小卷积核，用代数变换把乘法次数减少 $2.25\times$；
- **FFT 卷积**：当卷积核很长时（例如本项目 `scripts/slr28_audio.py` 用几万采样点长的真实房间脉冲响应 RIR 给 16kHz 音频加物理混响），时域卷积复杂度 $O(N \cdot K)$ 会慢到天荒地老；利用“时域卷积 = 频域逐点相乘”，通过快速傅里叶变换（FFT）瞬间降为 $O(N \log N)$！

### 2. 隐藏的显存杀手：大词表 Cross-Entropy 与 Fused Linear CE
很多新手算完模型权重和激活后，发现显存莫名其妙多爆了 **5 GB**！真凶往往是最后一层 `lm_head` + `CrossEntropy`！
算一笔触目惊心的账（以 Qwen3 词表 $V = 151,936$、Batch $B=4$、长度 $S=2048$ 为例）：

- 隐状态 `hidden` 只是 `[4, 2048, 1024]`，BF16 仅占 **16 MB**；
- 经过 `lm_head` 投影到词表变成 `logits` 形状 `[4, 2048, 151936]`，为了算 Softmax 数值稳定必须转成 **FP32（4 Bytes）**！
- **光这一个 `logits` 张量就占：$4 \times 2048 \times 151,936 \times 4\text{ B} = \mathbf{4.98\text{ GB}}$！** 反向传播还要再占一份梯度，瞬间吃掉近 10 GB！

**加速与省显存绝活（Chunked / Fused Linear Cross-Entropy，如 Liger-Kernel）**：
根本不需要一次性把 8192 个 Token 的 15 万维 `logits` 全部物化到 HBM！把序列按块切分（比如每次只算 512 个 Token 的 `hidden @ W_lm^T` → 立刻就地算出这 512 个 Token 的 Loss 和梯度 → 扔掉这小块 `logits`），**显存占用直接暴降 16 倍（从 5 GB 缩成 ~300 MB）**，同时因为减少巨型 HBM 写回，速度反而更快！

---

## 4.6 动手开眼界：用 30 行 Python/Triton 看透“算子融合”

过去写 GPU 算子必须啃极其硬核的 C++ CUDA（手动算线程块、防 Bank Conflict、写同步屏障）。OpenAI 开源的 **Triton** 让我们能用极像 Python/NumPy 的代码，直接写出媲美手写 CUDA 的高速融合算子！

对比下面两种实现 **RMSNorm** 的方式：

- **朴素 PyTorch 写法**：`x.pow(2)`（读 HBM 1次 + 写 1次）→ `.mean()`（读 1次 + 写 1次）→ `+ eps` → `rsqrt()`（读+写）→ `* x`（读2次+写1次）→ `* weight`（读2次+写1次）——**前前后后把张量在 HBM 里搬了十来趟！**
- **Triton 融合 Kernel**：把一整行 `x` 一次性搬进片上 SRAM/寄存器，在芯片里一口气完成平方、求和、开方倒数、乘权重，最后只写回 HBM 1 次！

```python
import triton
import triton.language as tl

@triton.jit
def fused_rmsnorm_kernel(X_ptr, W_ptr, Y_ptr, stride_row, N: tl.constexpr, eps, BLOCK_SIZE: tl.constexpr):
    # 每个 GPU Program (线程块) 负责处理矩阵的 1 行（即 1 个 Token 的隐藏向量）
    row_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # 1. 仅从 HBM 读 1 次输入 x 和权重 w 到高速寄存器/SRAM！
    x = tl.load(X_ptr + row_id * stride_row + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # 2. 全部中间计算都在芯片内部极速完成（零 HBM 往返！）
    mean_sq = tl.sum(x * x, axis=0) / N
    rstd = tl.rsqrt(mean_sq + eps)
    y = (x * rstd) * w

    # 3. 仅向 HBM 写回 1 次最终结果！
    tl.store(Y_ptr + row_id * stride_row + cols, y.to(tl.bfloat16), mask=mask)
```

看懂这 15 行核心代码，你就彻底领悟了“GPU 算子优化”的全部灵魂——**融合（Fusion）+ 分块（Tiling）+ 低精度 Tensor Core（MMA）**，万变不离其宗！

::: note 本章小结
1. GEMM 靠 **Tiling 分块进 SRAM + 异步双缓冲 + Tensor Core + 维度对齐** 吃满算力。
2. FlashAttention 靠 **Online Softmax + 分块流式计算** 彻底消灭 $O(N^2)$ 的 HBM 读写，数值 100% 精确等价。
3. 现代 LLM 三大件：**RMSNorm（+ QK-Norm 稳注意力）+ SwiGLU（动态门控）+ RoPE（相对距离旋转编码）**。
4. 遇到 15 万大词表 OOM 时，别忘了检查 `lm_head` 的 FP32 Logits，优先用 Chunked/Fused CE 或控制 micro-batch 长度。
:::
