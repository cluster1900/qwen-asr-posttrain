# 第 8 章 语言解码器 28 层逐层拆解与“任意模型解剖通法”

> **本章情绪定心丸**：恭喜你来到整个现代 AI 世界的“圣殿核心”！这一章解剖的 `Qwen3-0.6B` 文本解码器，和当今最顶级的 `Qwen3-32B`、`LLaMA-3-70B`、`Mistral`、`DeepSeek` 共享**几乎完全相同的骨架**！学会拆这 28 层，以后无论开源社区明天发布什么新模型，你只需 3 行代码就能瞬间看穿它的五脏六腑！

---

## 8.1 入口层：`embed_tokens` 词表查表（151,936 × 1,024）

计算机不认识汉字“今”，只认识它的词表编号（Token ID，例如整数 `20002`）。Qwen3 使用字节级 BPE 分词器（Byte-Level BPE），词表大小 `vocab_size = 151,936`（特意补齐为 128 的整数倍以迎合 GPU Tensor Core）。

**`embed_tokens` 本质上就是一本厚厚的 151,936 行字典**，每一行是一个 1024 维的浮点向量：

- 输入文本 Token ID 序列 `[Batch, Seq]`；
- 查表取出对应行，得到 `[Batch, Seq, 1024]`；
- 接着把第 7 章音频编码器产出的 `[50, 1024]` 听觉向量原位覆盖到 `<|audio_pad|>` 位置！
- 参数量：$151,936 \times 1,024 = \mathbf{155,582,464}$。

---

## 8.2 核心重复模块：单层 Qwen3 Decoder Layer 显微镜（重复 28 次）

现在，融合了“声音向量 + 指令词向量 + 已生成汉字向量”的张量 `[Batch, Seq, 1024]`，开始逐层爬升 28 层楼（`thinker.model.layers.0` 到 `layers.27`）。

请仔细端详图 8-1，**这里藏着一个连 90% 的进阶工程师都会算错的 Qwen3 架构细节！**

![图 8-1 Qwen3-0.6B 单个 Decoder Layer 内部显微镜（28 层重复，单层 15,730,944 参数）](assets/diagrams/13_qwen3_decoder_layer.svg)

### 抓到一个教科书级“反直觉”细节：`num_heads × head_dim = 2048 ≠ hidden_size (1024)`！

在旧的 GPT-2、LLaMA-2/3 中，大家背惯了一个恒等式：`head_dim = hidden_size // num_attention_heads`。
如果按老经验去猜 `Qwen3-0.6B`（`hidden_size = 1024`, `num_attention_heads = 16`），你会以为 `head_dim = 1024 / 16 = 64`，以为 `q_proj` 是 `1024 → 1024`。
**大错特错！** Qwen3-0.6B 为了让超小模型也拥有媲美大模型的注意力表达容量，把单头维度显式设为 **`head_dim = 128`**！
于是：

- **Query 侧（16 个头）总宽度** = $16 \times 128 = \mathbf{2,048}$（竟然是 `hidden_size` 的 **2 倍**！）；
- **Key / Value 侧（8 个头，GQA 2:1）总宽度** = $8 \times 128 = \mathbf{1,024}$；
- **`o_proj` 输出汇总**把 `2,048` 重新压回 `1,024`！

还记得第 6 章我们核对真实 A100 训练日志里的 LoRA 参数量 `4,587,520` 吗？正是因为 `q_proj` 和 `o_proj` 是 `1024 ↔ 2048` 而不是 `1024 ↔ 1024`，单层 LoRA（`r=16`）才是 `49,152 + 32,768 + 32,768 + 49,152 = 163,840`，乘以 28 层恰好等于 **`4,587,520`**，一分不差！

### 逐组件拆解单层全部 9 个子部件（精确合计 `15,730,944` 参数）

| 序号 | 组件代码名 | 精确维度 / 运算 | 参数量 | 形象比喻与核心作用 |
| :---: | :--- | :--- | ---: | :--- |
| 1 | `input_layernorm` | RMSNorm(`1024`) | `1,024` | **入场安检秤**：把上一层传来的向量尺度稳住，防梯度爆炸 |
| 2 | `self_attn.q_proj` | Linear(`1024 → 2048`, 无 bias) | `2,097,152` | **提出 16 组“检索问题”**（★挂 LoRA：当前汉字该去听音频哪一帧、看前文哪个词？） |
| 3 | `self_attn.k_proj` | Linear(`1024 → 1024`, 无 bias) | `1,048,576` | **生成 8 组“内容标签”**（★挂 LoRA：每帧声音和每个词给自己贴上索引标签） |
| 4 | `self_attn.v_proj` | Linear(`1024 → 1024`, 无 bias) | `1,048,576` | **打包 8 组“实际干货”**（★挂 LoRA：一旦匹配成功，把这段信息搬过去） |
| 5 | `q_norm` & `k_norm` | Per-Head RMSNorm(`128`) × 2 | `256` | **Qwen3 独门稳压阀**：防 Q·K 点积失控，长序列训练不炸 Loss |
| — | RoPE + 因果 GQA SDPA | `theta=1e6`, 2 个 Q 头共享 1 个 KV 头 | `0` | **信息路由交换机**：计算每个词对音频和历史词的注意力权重并加权聚合，输出 `[B, S, 2048]` |
| 6 | `self_attn.o_proj` | Linear(`2048 → 1024`, 无 bias) | `2,097,152` | **会议纪要汇总员**（★挂 LoRA）：把 16 个头查回来的线索整合成 1024 维，汇入**残差高速路 1** |
| 7 | `post_attention_layernorm` | RMSNorm(`1024`) | `1,024` | **二次安检秤**：准备进入知识记忆库 |
| 8 | `mlp.gate_proj` & `up_proj` | 两条 Linear(`1024 → 3072`, 无 bias) | `6,291,456` | **3072 格知识记忆抽屉 + SiLU 阀门**：把 1024 维信息展开，激活对应的中文词汇、语法与常识神经元 |
| 9 | `mlp.down_proj` | Linear(`3072 → 1024`, 无 bias) | `3,145,728` | **结论压回主干**：写回 1024 维并汇入**残差高速路 2** |
| — | **单层总计** | Attention(`6.29M`) + MLP(`9.44M`) | **`15,730,944`** | 28 层合计 **`440,466,432`** 参数！ |

::: tip 一句话记住 Attention 和 MLP 的分工！
学术界通过机制可解释性（Mechanistic Interpretability）发现了一个极其好记的分工：

- **Self-Attention（占单层 40% 参数）是“信息搬运工”**：负责跨位置搬运线索——在 ASR 里，它决定“写第 5 个字时，眼睛要盯紧第 18–22 帧音频、同时回头看第 4 个字是什么”，所以**给 Attention 挂 LoRA 是调整“听觉-语义对齐”最精准的手术刀！**
- **SwiGLU MLP（占单层 60% 参数）是“静态知识库”**：像键值对记忆一样存着“松江新浜镇是个地名”“辟谣不写作避谣”等世界知识与词汇搭配。如果在领域迁移时发现模型缺乏专业术语知识，可以把 `gate_proj / up_proj / down_proj` 也纳入 LoRA 目标！
:::

---

## 8.3 终点层：Final RMSNorm、`lm_head` 与自回归推理完整时序

爬完第 27 层（共 28 层），最后过一道 `model.norm`（RMSNorm `1024` 参数），然后通过 `lm_head`（与 `embed_tokens` 共享的 `151,936 × 1,024` 矩阵）把 1024 维隐状态投影成 151,936 个候选词的打分 Logits！

模型是怎么把一整句话逐字“吐”出来的？看懂图 8-2 的**两阶段推理时序图（Prefill vs Decode）**：

![图 8-2 大模型推理完整时序图：Prefill 并行首字填充与 Decode KV Cache 自回归吐字](assets/diagrams/14_inference_prefill_decode.svg)

1. **阶段一：Prefill（预填充，算力密集）**
   - 一次性把 50 个音频 Token + 指令 Prompt 全部并行送进 28 层解码器；
   - 在每一层把这些前缀 Token 的 Key 和 Value 矩阵存进显存 **KV Cache**；
   - 取最后一个位置（`<asr_text>`）的 Logits，选概率最高的词（Greedy 贪心解码 `argmax`），吐出第 1 个汉字 `今`！耗时叫做 **TTFT（首字延迟）**。
2. **阶段二：Decode（自回归吐字，显存带宽密集）**
   - 把刚生成的 `今` 这**单个 Token**喂回模型（音频编码器再也不用跑了！）；
   - 28 层解码器只需算 `今` 自己的 `q, k, v`，把 `(k, v)` 追加到 KV Cache 末尾，然后拿 `q` 去和整个 KV Cache 做注意力，吐出第 2 个字 `天`；
   - 如此循环，直到吐出终止符 `<|im_end|>`（ID `151645`）或达到 `max_new_tokens` 上限！

### 为什么评测用 Greedy，而在线 RL 探索必须用 Sampling？
- **Greedy（贪心解码，`do_sample=False`）**：每一步无脑选概率第 1 名的词。优点是**100% 确定、可复现、方差为零**，因此工业界所有基准测试和阶段门禁（如本项目 `decode.eval`）都固定使用 Greedy！
- **Sampling（采样解码，`temperature, top_p, top_k`）**：按概率掷骰子，允许模型偶尔选第 2 名、第 3 名的同音候选字（如“监管” vs “结款”）。没有多样性探索，强化学习就永远发现不了比当前更好或更差的候选句，所以 `decode.rollout` 才开启采样！

---

## 8.4 举一反三：3 行代码解剖宇宙中“任意”开源大模型！

学会解剖 `Qwen3-ASR-0.6B` 后，明天老板丢给你一个 `Llama-3.1-8B`、`Qwen2.5-VL-7B`、`DeepSeek-V2-Lite`、`Whisper-large-v3` 或 `BGE-M3`，你怎么在 60 秒内把它摸得一清二楚？

**通用解剖五步心法**：

1. **先读 `config.json`**：扫一眼 6 个核心数——`num_hidden_layers`（层数）、`hidden_size`（宽度）、`num_attention_heads` & `num_key_value_heads` & `head_dim`（查清 GQA 比例和是否 `Hq*D != H`！）、`intermediate_size`（MLP 宽度）、`vocab_size` & `tie_word_embeddings`（词表是否绑定）。
2. **跑下面 15 行“X 光透视脚本”**：按模块树统计精确参数量，一眼定位哪部分是 Encoder（视觉/听觉）、哪部分是 Projector、哪部分是 Decoder，以及所有 `nn.Linear` 的准确正则路径！
3. **横向映射常见模型家族**：

| 模型类型 / 代表作 | 核心架构 | 典型 Linear 层路径关键字（用于配置 LoRA `target_modules`） | 常见微调策略 |
| :--- | :--- | :--- | :--- |
| **纯文本 Decoder LLM** (`Qwen3`, `LLaMA-3`, `Mistral`) | 仅因果 Decoder（RMSNorm + GQA + RoPE + SwiGLU） | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` | SFT/DPO/GRPO：常全挂 7 个线性层；极小数据防遗忘可只挂注意力 4 层 |
| **MoE 稀疏专家 LLM** (`DeepSeek-V3`, `Qwen3-MoE`) | Attention + 路由门控 Router + 多个专家 MLP | 注意力 `q/k/v/o`（或 MLA `q_a/b, kv_a/b`）+ 共享专家 + 路由专家 | 优先挂 Attention + 共享专家；谨防微调把 Router 路由训塌（负载均衡崩盘） |
| **视觉多模态 VLM** (`Qwen2.5-VL`, `LLaVA`) | ViT 视觉编码器 + Merger/Projector + LLM 解码器 | `visual.*`（视觉） + `merger.*`（投影） + `model.layers.*`（语言） | 先冻 ViT 训 Projector+LLM；细粒度 OCR 任务可解冻 ViT 顶层 |
| **编码器-解码器 ASR** (`Whisper`) | Audio Encoder + Cross-Attention + Decoder | `q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, `fc2` | Encoder-Decoder 经典交叉注意力，可对 Decoder 或全网挂 LoRA |
| **双向编码器 / 检索** (`BERT`, `BGE`, `RoBERTa`) | 仅双向 Encoder（类似第 7 章 AuT） | `query`, `key`, `value`, `dense` | 对比学习 InfoNCE 微调，常用全参或高秩 LoRA |

```python
# 万能大模型“X 光透视脚本”：拿任意 HuggingFace 模型直接套用！
import torch
from collections import defaultdict

def xray_model(model: torch.nn.Module) -> None:
    total, trainable = 0, 0
    by_prefix = defaultdict(int)
    linear_suffixes = set()

    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
        top = ".".join(name.split(".")[:2])
        by_prefix[top] += n

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_suffixes.add(name.split(".")[-1])

    print(f"=== 模型总参数: {total:,} | 可训练: {trainable:,} ({trainable/max(total,1):.2%}) ===")
    for k, v in sorted(by_prefix.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {k:<32s} : {v:>13,} ({v/total:6.2%})")
    print("可挂载 LoRA 的 Linear 后缀清单:", sorted(linear_suffixes))
```

至此，模型在你面前已经彻底没有任何黑盒了！接下来进入最令人热血沸腾的**第四篇：亲手做后训练（LoRA → SFT → DPO → RL/GRPO → 严谨评测发布）**！
