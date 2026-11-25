# 第 5 章 显存精算与显卡选型实战：每一字节去哪了？

> **本章情绪定心丸**：新手做模型后训练，最令人崩溃的瞬间莫过于辛辛苦苦配好环境，代码刚跑 3 秒，屏幕弹出鲜红的 `torch.cuda.OutOfMemoryError`。大多数人只能像盲人摸象一样乱改 `batch_size` 或盲目加钱租更贵的显卡。读完这一章，你手里会握有一张**精确到 MB 的显存财务报表**——还没开机租卡，你就能提前 5 分钟算准需要多少 GB 显存、该租哪张卡最省钱、爆了显存先拧哪个开关！

---

## 5.1 训练显存的“四大吞金兽”精确公式

设模型总参数量为 $P$，其中参与梯度更新的可训练参数量为 $P_{\text{train}}$（全参微调时 $P_{\text{train}} = P$；LoRA 微调时 $P_{\text{train}}$ 通常仅占 $0.5\% \sim 2\%$）。训练时 GPU 显存被四大块牢牢瓜分：

$$\text{VRAM}_{\text{总}} = \underbrace{M_{\text{权重}}}_{\text{模型本身}} + \underbrace{M_{\text{梯度}}}_{\text{反向导数}} + \underbrace{M_{\text{优化器}}}_{\text{最大吞金兽!}} + \underbrace{M_{\text{激活}}}_{\text{前向留存}} + \underbrace{M_{\text{缓冲}}}_{\text{CUDA/碎片}}$$

![图 5-1 大模型训练显存四大吞金兽拆解对比（7B 模型 & Qwen3-ASR-0.6B）](assets/diagrams/09_vram_breakdown.svg)

下面逐项列出精确到字节的计算公式：

### 1. 模型权重（Parameters）
- **FP32 全精度**：$4 P$ 字节（每参数 4 Bytes）
- **BF16 / FP16 混合精度**：$2 P$ 字节（每参数 2 Bytes）
- **INT8 量化**：$1 P$ 字节
- **NF4 4-bit 量化（QLoRA）**：约 $0.55 P \sim 0.60 P$ 字节（0.5 字节权重 + 量化缩放系数开销）

### 2. 梯度（Gradients）
**只有 `requires_grad=True` 的参数才需要存梯度！** 这是 LoRA 狂省显存的第一大秘密。

- 若可训练参数是 BF16：$2 P_{\text{train}}$ 字节
- 若可训练参数是 FP32（如我们推荐的防 ULP 下溢模式）：$4 P_{\text{train}}$ 字节

### 3. 优化器状态（Optimizer States）——全参训练的头号巨兽！
以最主流的混合精度 **AdamW** 为例，对每一个可训练参数，优化器都需要额外保存：

- **标准 BF16 模型 + FP32 混合精度 AdamW**：
  - FP32 主权重副本：$4 P_{\text{train}}$ 字节
  - 一阶动量 $m$（FP32）：$4 P_{\text{train}}$ 字节
  - 二阶动量 $v$（FP32）：$4 P_{\text{train}}$ 字节
  - 合计：**$12 P_{\text{train}}$ 字节！**（光优化器就比模型本身大 **6 倍**！）
- **若使用 FP32 LoRA 权重直接喂给 AdamW**（无需另存一份主权重副本）：
  - 权重本身已占 $4 P_{\text{train}}$ + 梯度 $4 P_{\text{train}}$ + 动量 $m, v$ 占 $8 P_{\text{train}}$ = **合计 $16 P_{\text{train}}$ 字节**（但因为 $P_{\text{train}}$ 极小，总和也就几十 MB！）。
- **省钱替代品 8-bit AdamW（`bitsandbytes`）**：将 $m, v$ 量化为 8-bit，优化器状态从 $8 P_{\text{train}}$ 骤降为 **$2 P_{\text{train}}$**！

::: tip 黄金口诀：静态显存一眼速算（单位：GB，参数量 $P$ 单位为十亿 B）
- **全参混合精度微调（BF16 + FP32 AdamW）**：权重 $2P$ + 梯度 $2P$ + 优化器 $12P$ = **$16 P\text{ GB}$**（加上激活与缓冲约 **$18 \sim 20 P\text{ GB}$**！所以 7B 全参需要 $\approx 128\text{ GB}$！）
- **BF16 LoRA（训 1% 参数）**：基座 $2P$ + LoRA 全套仅 $\approx 0.2 P$ ≈ **$2.2 P\text{ GB}$ + 激活**（7B 约需 18–20 GB！）
- **QLoRA（NF4 基座 + LoRA）**：基座 $0.6 P$ + LoRA 全套 $\approx 0.2 P$ ≈ **$0.8 P\text{ GB}$ + 激活**（7B 仅需 ~10 GB！）
:::

### 4. 中间激活值（Activations）——随长度和 Batch 线性/平方飙升的变数！
根据经典的 Megatron-LM 推导，对于一个 $L$ 层、隐藏维度 $H$、注意力头数 $A$、序列长度 $S$、微批次大小 $B$ 的标准 Transformer，在 BF16 下：

- **不开启 Gradient Checkpointing（使用 FlashAttention 消除 $N^2$ 注意力矩阵后）**：每层仍需保存 RMSNorm、QKV、SwiGLU 门控/升维等中间张量，单层约 $34 B S H$ 字节（若不用 FlashAttention 还要再加 $5 B A S^2$ 字节！），$L$ 层合计：
  $$M_{\text{act, no-ckpt}} \approx 34 \cdot L \cdot B \cdot S \cdot H \text{ Bytes}$$
- **开启 Gradient Checkpointing 后**：每层只需常驻入口的 1 个隐状态张量 $[B, S, H]$，外加反向重算单层时的临时峰值，全网络骤降至：
  $$M_{\text{act, ckpt}} \approx 2 \cdot L \cdot B \cdot S \cdot H + 34 \cdot B \cdot S \cdot H \text{ Bytes}$$
- **别忘了最后一层的 FP32 Logits 峰值**：$4 \cdot B \cdot S \cdot V$ 字节（$V$ 为词表大小，Qwen3 中 $V = 151,936$）。
- **再加上框架固定开销 $M_{\text{缓冲}}$**：CUDA Context + cuBLAS/cuDNN workspace + PyTorch Caching Allocator 碎片缓冲，通常预留 **1.0 – 2.5 GB**。

---

## 5.2 实战手算：本项目 `Qwen3-ASR-0.6B` 与常见 7B–70B 全景账本

### 案例 A：精确手算 `Qwen3-ASR-0.6B`（基座 `782,426,112` + LoRA `4,587,520`）
参数：$P = 0.7824\text{ B}$，$P_{\text{train}} = 0.004588\text{ B}$，$L = 28$，$H = 1024$，$V = 151,936$，设 $B = 1$、音频+文本总序列长 $S = 1024$（开启 Flash/SDPA + Gradient Checkpointing）：

| 显存分项 | 全参微调 (BF16 + FP32 AdamW) | 本项目 LoRA (BF16 基座 + FP32 LoRA) | 说明 |
| :--- | :---: | :---: | :--- |
| 模型权重 | $782.4\text{M} \times 2 = \mathbf{1.565\text{ GB}}$ | $1.565\text{ GB} + 18.4\text{ MB} = \mathbf{1.583\text{ GB}}$ | LoRA 权重存 FP32 防 ULP 下溢 |
| 梯度 | $782.4\text{M} \times 2 = \mathbf{1.565\text{ GB}}$ | $4.59\text{M} \times 4 = \mathbf{18.4\text{ MB}}$ | 冻结参数零梯度！ |
| AdamW 状态 | $782.4\text{M} \times 12 = \mathbf{9.389\text{ GB}}$ | $4.59\text{M} \times 8 = \mathbf{36.7\text{ MB}}$ | 整整缩小 **256 倍**！ |
| 激活值 (Ckpt) | $\approx 0.12\text{ GB}$ | $\approx 0.12\text{ GB}$ | 28 层入口 + 1 层重算峰值 |
| Logits (FP32) | $1 \times 1024 \times 151936 \times 4 = \mathbf{0.622\text{ GB}}$ | $\mathbf{0.622\text{ GB}}$ | 反向梯度再翻倍 ≈ 1.24 GB |
| 音频编码器前向+CUDA 缓冲 | $\approx 1.8\text{ GB}$ | $\approx 1.8\text{ GB}$ | 30s 音频 Mel/Conv + 碎片 |
| **预估总峰值显存** | **≈ 15.6 GB** | **≈ 4.8 – 6.0 GB** | **省出 10 GB，连 12GB 消费卡都能跑！** |

> **DPO 和 RL 会不会显存翻倍？**
> 很多人以为 DPO/RL 要同时在 GPU 里放“策略模型 + 参考模型”两份权重，显存直接翻倍！**完全不必！**
> 1. **DPO 阶段**（见 `train/reference_logprob.py`）：提前离线用参考模型把所有偏好对的 `ref_chosen_logp`、`ref_rejected_logp` 算好存进 JSONL 缓存——训练时 GPU 里**压根不用加载参考模型**！
> 2. **RL/GRPO 阶段**（见 `train/train_rl.py`）：因为策略和参考共用同一个冻结的 1.56GB 基座，参考模型和策略模型的区别仅仅是那 **18 MB 的 LoRA 权重**！把参考 LoRA 备份在 CPU 内存里，需要算 `ref_logp` 时临时换进去算完立刻换回来——**额外 GPU 显存开销 = 0 MB！**

### 案例 B：主流开源大模型训练显存速查表（Seq = 2048, B = 1, 开启 GradCkpt + SDPA）

| 模型规模 | 全参 BF16 微调 | BF16 LoRA (`r=16`, ~1%参数) | QLoRA (NF4 + LoRA) | 最低推荐训练显卡（单卡 / 多卡） |
| :--- | :---: | :---: | :---: | :--- |
| **0.6B–1.5B** | ~15–28 GB | **5–8 GB** | **3–5 GB** | RTX 4060Ti 16G / T4 16G / L4 24G |
| **7B–8B** | ~128–145 GB | **19–22 GB** | **10–12 GB** | LoRA: 4090/L4 24G；QLoRA: T4 16G；全参: 2×A100 80G |
| **14B** | ~250 GB | **34–38 GB** | **15–18 GB** | QLoRA: 4090/L4 24G；LoRA: A100 40G/80G |
| **32B** | ~570 GB | **70–75 GB** | **24–28 GB** | QLoRA: A100 40G；LoRA: A100/H100 80G |
| **70B–72B** | ~1.26 TB | ~155 GB | **46–52 GB** | QLoRA: 单卡 A100/H100 80G；全参: 2×8×H100 |

---

## 5.3 推理显存精算：KV Cache 公式与 GQA 省了多少显存？

模型训完上线推理时，没有梯度、没有优化器、也不存反向激活，显存只剩下两部分：**模型权重 + KV Cache**！

每生成一个新词，解码器都需要查历史所有 Token 在每一层的 Key 和 Value 向量。因此 KV Cache 的精确公式为：

$$\text{KV Cache (Bytes)} = \underbrace{2}_{\text{K 和 V}} \times L \times H_{\text{kv}} \times d_{\text{head}} \times S \times B \times \text{sizeof}(\text{dtype})$$

::: example 手算对比：GQA 到底帮 Qwen3-0.6B 省了多少显存？
代入 `Qwen3-0.6B` 的真实结构参数：$L = 28$ 层，查询头 $H_q = 16$，键值头 $H_{\text{kv}} = 8$（2:1 GQA），单头维度 $d_{\text{head}} = 128$，BF16 占 2 字节：

- **每个 Token 占用的 KV Cache**：
  $$2 \times 28 \times 8 \times 128 \times 2 = 114,688\text{ Bytes} = \mathbf{112\text{ KiB / token}}$$
- 如果它是传统 MHA（$H_{\text{kv}} = 16$），每 Token 就会翻倍到 **224 KiB**；如果像 DeepSeek-V3 用了 **MLA（多头潜在注意力）**，还能再压缩 5–10 倍！
- 推算并发：在单张 24GB 的 L4 显卡上部署 `Qwen3-ASR-0.6B`，权重占 1.56 GB，留出 18 GB 给 KV Cache，每条 30 秒语音约 500 个 Token（占 55 MiB），理论可同时并发处理 **$18,432 \div 55 \approx 330$ 条请求**！
:::

---

## 5.4 显卡选购决策树与 OOM 急救五连招

买卡或租云卡（Colab / AutoDL / RunPod / Lambda / GCP）时，不要只盯着“显存 GB”，一定要同时看三个指标：**显存容量（决定能不能跑）、显存带宽（决定推理吐字快不快）、BF16 TFLOPS（决定训练快不快）**。

| 显卡型号 | 显存容量 | 显存带宽 | BF16 算力 | 原生 BF16 | 典型定位与推荐场景 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **NVIDIA T4** | 16 GB | 320 GB/s | 65 TFLOPS (FP16) | **否** (仅 FP16) | 免费/低价尝鲜，只能跑 FP16/QLoRA 小实验 |
| **RTX 4090** | 24 GB | 1,008 GB/s | 330 TFLOPS* | **是** | **个人性价比之王**：7B LoRA / 14B QLoRA 飞快 |
| **NVIDIA L4** | 24 GB | 300 GB/s | 121 TFLOPS | **是** | 云端廉价推理/小模型补跑（带宽低，Decode 偏慢） |
| **A100 40G / 80G** | 40 / 80 GB | 1.55 / 2.0 TB/s | **312 TFLOPS** | **是** | **黄金标准训练主力**：本项目正式训练与评测环境 |
| **H100 80G SXM** | 80 GB | **3.35 TB/s** | **989 TFLOPS** | **是** (+FP8) | 工业级高速训练、32B/70B 批量采样与大并发 RL |

*\*注：RTX 4090 标称 330 TFLOPS 为 FP16 累加（FP32 累加为 165 TFLOPS），单卡性价比依然无敌。*

![图 5-2 个人与团队选卡 + 训练方案决策流程图](assets/diagrams/10_gpu_selection_matrix.svg)

### 遇到 `CUDA OutOfMemoryError` 的“黄金急救五连招”
按**对效果伤害最小、省显存最立竿见影**的顺序，依次拧动这 5 个开关：

1. **减小 `micro_batch_size`（甚至降到 1），成倍增大 `gradient_accumulation_steps`**：
   - 例：`B=8, accum=1` 改成 `B=1, accum=8`。只要 Loss 按总有效 Token 归一化，**数学梯度完全等价**，激活显存直接砍成 1/8！
2. **开启 Gradient Checkpointing（`use_reentrant=False`）**：
   - 激活显存立省 60%–80%，时间只多 30%。
3. **开启 SDPA / FlashAttention + 截断异常超长样本 + Chunked Cross-Entropy**：
   - 哪怕 1 万条数据里混进 1 条 120 秒的巨长音频，也会在半夜把整个训练炸死！务必像本项目一样在数据校验阶段硬卡 `max_audio_seconds: 30`、`max_input_tokens: 2048`。
4. **解决 PyTorch 显存碎片**：
   - 报错信息里如果写着 `reserved in total by PyTorch` 远大于 `allocated`，说明显存被切碎了！启动前加上环境变量：
     `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   - 并在每次评测前调用 `gc.collect(); torch.cuda.empty_cache()`。
5. **换用 8-bit AdamW 或退守 QLoRA (NF4)**：
   - 当连模型权重本身都快占满卡的 75% 时，启用 4-bit NF4 量化（`bitsandbytes`），显存立省 70%！

```python
# 实用显存精算器：拷贝运行，输入任意模型参数即可算出训练与推理显存预算！
def estimate_vram_gb(
    total_params_b: float,
    trainable_ratio: float = 0.01,
    quant_bits: int = 16,      # 16=BF16, 8=INT8, 4=QLoRA NF4
    layers: int = 28,
    hidden: int = 1024,
    vocab: int = 151936,
    seq_len: int = 2048,
    micro_batch: int = 1,
    grad_ckpt: bool = True,
    fp32_lora: bool = True,
) -> dict[str, float]:
    bytes_per_base = {16: 2.0, 8: 1.0, 4: 0.58}[quant_bits]
    p_base = total_params_b * 1e9
    p_train = p_base * trainable_ratio
    w_gb = (p_base * bytes_per_base + (p_train * 4 if fp32_lora and trainable_ratio < 1 else 0)) / 1e9
    g_gb = (p_train * (4 if fp32_lora else 2)) / 1e9
    opt_gb = (p_train * (8 if fp32_lora else 12)) / 1e9
    act_bytes = (2 * layers + 34) * micro_batch * seq_len * hidden if grad_ckpt else 34 * layers * micro_batch * seq_len * hidden
    logits_gb = 2 * 4 * micro_batch * seq_len * vocab / 1e9  # 前向+反向 FP32 logits
    act_gb = act_bytes / 1e9 + logits_gb
    overhead_gb = 1.5
    return {
        "weights_GB": round(w_gb, 2),
        "grads_GB": round(g_gb, 2),
        "optim_GB": round(opt_gb, 2),
        "activations_GB": round(act_gb, 2),
        "overhead_GB": overhead_gb,
        "total_est_GB": round(w_gb + g_gb + opt_gb + act_gb + overhead_gb, 2),
    }

print("Qwen3-ASR-0.6B LoRA:", estimate_vram_gb(0.7824, trainable_ratio=0.00586, seq_len=1024))
print("7B LLM LoRA:        ", estimate_vram_gb(7.0, trainable_ratio=0.01, layers=32, hidden=4096))
print("7B LLM QLoRA:       ", estimate_vram_gb(7.0, trainable_ratio=0.01, quant_bits=4, layers=32, hidden=4096))
```

::: note 本章小结
1. 记住速算锚点：**全参 BF16 ≈ `18×P GB`**、**BF16 LoRA ≈ `2.2×P + 激活`**、**QLoRA ≈ `0.8×P + 激活`**。
2. DPO 用离线 Reference 缓存、RL 用 CPU 暂存 LoRA 切换，单卡无需放两份基座权重！
3. 推理显存看权重 + KV Cache；GQA 直接把 KV Cache 按 $H_{\text{kv}} / H_q$ 比例削减。
:::
