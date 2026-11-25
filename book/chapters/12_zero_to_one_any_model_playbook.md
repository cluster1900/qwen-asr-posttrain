# 第 12 章 从 0 到 1：任意模型后训练“七步万能实战手册”

> **本章情绪定心丸**：走到这里，你已经完成了从高中数学、PyTorch 底层、GPU 架构、七大算子、显存精算、7.82 亿参数逐层解剖，到 SFT / DPO / RL 工业闭环的完整淬炼！现在，让我们把所有知识浓缩成一套**拿来就能用的万能行动蓝图**——无论明天摆在你面前的是 `Qwen3-8B` 客服/代码助手、`Qwen2.5-VL` 发票 OCR、还是 `Whisper / Qwen3-ASR` 方言语音模型，照着这七步走，你一个人就能从 0 到 1 交付生产级后训练模型！

---

## 12.1 任意模型后训练“七步万能 SOP”

拿到一个全新的业务需求和开源基座模型，永远按下面这个严格顺序推进，**绝不跳步**：

### 第 1 步：X 光解剖模型 + 精算显存选卡（30 分钟）
1. 跑第 8 章的 `xray_model(model)` 打印全模型层结构与精确参数量，确认：
   - 总参数量 $P$、层数 $L$、隐藏维度 $H$、词表大小 $V$、是否有独立 Encoder/Projector；
   - 准确的 `target_modules`（小数据防遗忘先挂注意力 4 层 `q,k,v,o`；大领域知识注入全挂 7 层含 `gate,up,down`）。
2. 跑第 5 章的 `estimate_vram_gb()` 估算显存，决定选 BF16 LoRA 还是 QLoRA (NF4) + 选单卡（24G/40G/80G）或 FSDP 多卡。

### 第 2 步：数据工程与“零泄漏切分”（占项目成败的 60%！）
1. **先做 Base 抽样诊断**：随机抽 50 条真实业务数据跑一遍原始 Base 模型，肉眼逐条看它错在哪——是**格式/标点不合预期**？是**领域术语不懂**？还是**噪声/长上下文失灵**？
2. **对齐先验，绝不与基座硬刚**：确保训练集目标格式（系统提示词、标点、代码块围栏、思维链 `<think>` 标签）与基座的原生 Chat Template 一致！
3. **严格分组切分**：按用户/文档/说话人/房间分组切分 `train / val / test`，把 `test` 彻底锁进保险柜，训练阶段不准看它一眼！

### 第 3 步：建立 Baseline 与冻结自动化门禁（1 小时）
1. 在固定 `val` 上用**确定性 Greedy 解码**跑一遍 Base 模型，把预测结果、核心指标（准确率 / F1 / Pass@1 / CER）、以及“健康度指标”（空输出率、死循环重复率、超长截断率、通用能力防遗忘集分数）固化成 JSON 基线！
2. 提前白纸黑字写死**晋级门禁**：例如“业务核心指标必须提升 $\ge \delta$，且通用能力退化 $\le 1\%$，且异常输出率 $< 1\%$”。

### 第 4 步：5–20 步 Smoke Test 冒烟验收（防白烧显卡钱！）
在投入几小时正式训练前，先用 50 条数据跑一个 **10 步微实验**，逐项打勾：

- [ ] `trainable_params` 名称和数量 100% 精准匹配目标层，`lora_B` 初始全 0；
- [ ] 打印第 1 条样本的 `input_ids` 和 `labels` 解码文本，肉眼确认前缀全是 `-100`、答案完整且末尾包含真实 EOS；
- [ ] 开启 `Gradient Checkpointing (use_reentrant=False)` 后，检查每个 LoRA 参数的 `grad.norm()` 均非零、非 `NaN`；
- [ ] 测一次断点保存 + `--resume` 恢复，确认第 6 步 Loss 与不中断完全一致！

### 第 5 步：阶段一 SFT 监督微调（学会标准答案与领域规范）
- 推荐起步超参甜点：
  - **LoRA**：`r=16, alpha=32, dropout=0.05`（小数据集可设 `0.05`，DPO/RL 阶段建议关掉 `dropout=0.0` 防参考对数概率噪声！），**`trainable_dtype=float32`**；
  - **优化器**：`AdamW`，`betas=(0.9, 0.95)`，`weight_decay=0.01`，`max_grad_norm=1.0`；
  - **学习率**：7B–14B 模型 SFT LoRA 通常从 `1e-4` ~ `2e-4` 起试；0.6B–3B 高精度精调或容易破坏预训练先验的场景从 **`1e-5` ~ `5e-5`** 起试！
  - **有效 Batch Size**：`micro_batch × grad_accum = 8 ~ 64`，Warmup 占总步数 5%–10%，Cosine 退火至 10%。
- 每隔固定步数在 `val` 跑 Greedy 评测，选出**过门禁最优步**（不是最后一步！）并独立进程 Readback 封包为 `SFT Release`！

### 第 6 步：阶段二 DPO 偏好对齐（消除顽固错误、拉大好坏差距）
- 用 `SFT Release` 在训练集上推理，自动收集“模型做错了但我们有标准答案（或高分答案）”的难例构造 `(prompt, chosen, rejected)` 偏好对；
- **关键清洗**：剔除无效平局（Tie）、剔除无信息支撑的伪对、排查长度偏差（避免 Chosen 永远比 Rejected 长太多导致模型变话痨）；局部纠错任务强烈推荐开启第 10 章的 **`mismatched_only` 直通梯度掩码**；
- 离线缓存 `ref_logp`，设置更保守的学习率（通常是 SFT 的 `1/5 ~ 1/10`，如 **`2e-6 ~ 1e-5`**），$\beta \in [0.05, 0.2]$，确认第 1 步 Loss 紧贴 **`0.693147`**！

### 第 7 步：阶段三 GRPO / 在线 RL（可选，冲击可验证指标上限）与终局封存验收
- **适用场景**：只要你的任务有**自动可验证的客观裁判**（数学题答案、代码单元测试、JSON Schema 校验、SQL 执行结果、ASR CER），就可以上 GRPO！
- 每条输入采样 $G = 4 \sim 8$ 个候选，计算组内相对优势 $A_i$ + `k3` KL 惩罚（`kl_beta ≈ 0.02 ~ 0.10`），过滤零优势组，保持极小学习率（`1e-6 ~ 5e-6`）与 **FP32 LoRA 主权重**；
- **终局双盲验收**：只拿通过 `val` 门禁的最终冠军版本，在封存 `test` 集与真实线上 OOD 样本上跑**一次性评测**，用配对 Bootstrap 95% CI 验证显著性；通过后合并 LoRA 权重（`model.merge_and_unload()`）导出为独立推理模型（`safetensors`），零延迟部署至 vLLM / SGLang / TensorRT-LLM！

---

## 12.2 三大主流领域任务速查对照表

| 业务场景 | 典型基座模型 | SFT 数据构造 | DPO / GRPO 信号来源 | 必须盯紧的防回归健康度 |
| :--- | :--- | :--- | :--- | :--- |
| **文本对话 / 客服 / 知识库** | `Qwen3-8B`, `LLaMA-3.1-8B` | 多轮对话 JSONL，只对 `assistant` 回复算 Loss | DPO：人工/强模型打分对；GRPO：规则格式+事实核查 | 通用推理基准不跌、拒答率不飙升、不复读 |
| **代码生成 / 数学推理** | `Qwen2.5-Coder-7B`, `DeepSeek-R1-Distill` | 含正确推理链与通过单测的代码样本 | **GRPO 主场！** 编译器+单元测试通过=1.0，格式对=+0.1 | 语法错误率、死循环超长截断率 |
| **多模态视觉 OCR / 文档** | `Qwen2.5-VL-7B` | 图文对，先冻 ViT 训 Merger+LLM | DPO：易错字/错位表格对比对；GRPO：字段精确匹配 F1 | 幻觉编造不存在字段、漏行率 |
| **鲁棒语音识别 ASR** | `Qwen3-ASR-0.6B/1.7B`, `Whisper` | 真实物理 RIR/噪声增强 + 标点自对齐转写 | DPO：等长声学单字对+差异掩码+Anchor；GRPO：`1 - CER` | Clean 语音不退化、静音零幻觉、标点保持率 |

---

## 12.3 可直接改造的“任意模型 0-to-1 最小端到端训练脚手架”

下面这段 90 行高度凝练的 Python 脚本，融合了全书最精华的工程实践（BF16 基座 + FP32 LoRA 防 ULP 下溢 + `use_reentrant=False` Checkpointing 断链防线 + 精确 Token 归一化梯度累积 + 余弦周期解耦 + 原子保存与无损合并），你可以把它直接当作任何 PyTorch/HuggingFace 项目的启动模板！

```python
"""任意 Causal LM 0-to-1 生产级 LoRA SFT 最小脚手架（可直接改造用于任意 HuggingFace 模型）"""
import math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL_NAME = "Qwen/Qwen3-0.6B"  # 可替换为任意 Llama / Qwen / Mistral / DeepSeek 模型！

def build_lora_model(model_name: str, r: int = 16, alpha: int = 32):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. 基座加载为 BF16 + 原生 SDPA/FlashAttention
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    model.config.use_cache = False  # 训练必关 KV Cache

    # 2. 注入 LoRA（按需改 target_modules）
    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # 3. 铁律一：仅将可训练 LoRA 转为 FP32 主权重，彻底消灭 BF16 小学习率 ULP 下溢！
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    # 4. 铁律二：开启非重入 Checkpointing + Embedding 输出强开梯度，防 LoRA 断链！
    emb = model.get_input_embeddings()
    emb.register_forward_hook(lambda _m, _inp, out: out.requires_grad_(True) if torch.is_tensor(out) else out)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    model.print_trainable_parameters()
    return model, tokenizer

def make_supervised_example(tokenizer, prompt_text: str, answer_text: str, max_len: int = 2048):
    """构造前缀掩码(-100) + 强制包含 EOS 的单条 SFT 样本"""
    prefix_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = tokenizer.encode(prompt_text + answer_text + tokenizer.eos_token, add_special_tokens=False)
    assert full_ids[: len(prefix_ids)] == prefix_ids, "Tokenizer 边界粘连，请检查前缀尾字符！"
    assert len(full_ids) <= max_len, f"超长样本: {len(full_ids)} > {max_len}"
    labels = [-100] * len(prefix_ids) + full_ids[len(prefix_ids) :]
    return {
        "input_ids": torch.tensor([full_ids], dtype=torch.long),
        "attention_mask": torch.ones((1, len(full_ids)), dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
    }

def cosine_lr(step: int, warmup: int, total_horizon: int, base_lr: float, min_ratio: float = 0.1) -> float:
    """解耦余弦调度：total_horizon 独立于 max_steps 早停步数"""
    if step <= warmup:
        return base_lr * (step / max(warmup, 1))
    progress = min(1.0, (step - warmup) / max(total_horizon - warmup, 1))
    return base_lr * (min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress)))

def train_one_accum_window(model, optimizer, window_batches: list[dict], device) -> float:
    """严格按全窗口总监督 Token 数归一化的梯度累积（micro_batch=1 与大 Batch 数学等价！）"""
    optimizer.zero_grad(set_to_none=True)
    # 先统计窗口总监督 Token 数
    total_tokens = sum(int((b["labels"][:, 1:] != -100).sum().item()) for b in window_batches)
    assert total_tokens > 0
    total_nll_val = 0.0
    for batch in window_batches:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        shift_logits = logits[:, :-1, :].float()  # FP32 算 CE 保数值稳定
        shift_labels = labels[:, 1:]
        nll_sum = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="sum",
            ignore_index=-100,
        )
        (nll_sum / total_tokens).backward()       # 除以全窗口总 Token 数！
        total_nll_val += float(nll_sum.detach().item())
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return total_nll_val / total_tokens

# 训完验证过门禁后，一行零延迟合并导出：
# merged = model.merge_and_unload()
# merged.save_pretrained("./release_merged_model")
```

---

## 12.4 结语：从今天起，你也是“手握手术刀”的模型工程师

翻到这一页，请为自己鼓一次掌！
回顾我们在序言里的约定：你或许一开始看着“算术强度”“Online Softmax”“GQA”“配分函数偏移”“Schulman `k3` KL”感到陌生甚至发怵；可现在，你不仅能随手在纸上推导出它们的公式，还能一眼看出 7.82 亿参数里每个数字的去处，能算清一块 A100 或 4090 上每一 MB 显存的花销，更见识了真实生产训练里那些血淋淋却无比珍贵的踩坑与破案过程。

记住大师们的秘密：**顶尖的 AI 工程师从不迷信玄学炼丹——他们敬畏数据分布、看穿硬件瓶颈、用第一性原理做数学排错、用统计学门禁诚实面对每一次实验结果。**

打开你的终端，选一个你感兴趣的小模型和数据集，从跑通第一条 10 步 Smoke Test 开始吧——**真正的 0 到 1，从你敲下第一行 `loss.backward()` 的那一刻已经开始了！**
