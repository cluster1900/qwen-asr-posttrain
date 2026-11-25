# 第四篇：后训练全链路实战

# 第 9 章 后训练第一站：LoRA 数学原理与 SFT 监督微调实战

> **本章情绪定心丸**：恭喜你正式从“看懂模型”跨入“改造模型”的实战阶段！很多新手第一次跑微调，以为把数据往脚本里一塞、看 `train_loss` 往下降就大功告成了。结果上线一测：不仅新能力没学会，连模型原本最拿手的中文标点和安静场景识别都给训残了！这一章，我们先彻底讲透 **LoRA** 为什么能用 0.5% 的参数撬动 7.8 亿大模型，再带你完整走通 **SFT（监督微调）** 的每一个生死细节，并复盘我们在 A100 上亲历的“标点先验大坍塌与绝地翻盘”真实战役！

---

## 9.1 为什么现代 AI 的黄金范式是“预训练 + 三阶段后训练”？

把培养一名专业外科医生比作训练大模型：

1. **预训练（Pre-training）**：从小学读到医学院毕业——啃掉数万亿 Token 或百万小时音频，学会全人类的语言规律和听觉直觉。这步烧掉几百万美元，由大厂完成并开源出 **Base 模型**。
2. **后训练（Post-training / Alignment）**：住院医师规范化培训！花费极低（单张显卡几小时），分三步精准打磨：
   - **SFT（Supervised Fine-Tuning，监督微调）**：老师给标准示范题，手把手教模型**“正确格式与领域知识长什么样”**（如在真实房间混响、背景人声噪声下怎么稳定听清中文）。
   - **DPO（Direct Preference Optimization，直接偏好优化）**：拿模型自己犯过的错题当反面教材，对比“好答案 vs 坏答案”，专门**“拉大好坏差距、消除顽固同音错字”**。
   - **RL / GRPO（在线强化学习）**：把模型放进考场自己反复尝试生成多个答案，用客观打分裁判（如代码单测通过率、数学答案对错、或 ASR 字符错误率 CER）直接给出奖励信号，冲击任务指标上限！

![图 9-2 工业级后训练三阶段流水线（SFT → DPO → RL/GRPO）与不可变门禁发布合同](assets/diagrams/16_posttrain_pipeline_seq.svg)

---

## 9.2 LoRA（低秩适配）核心原理：为什么 0.58% 参数就够了？

在大型基座模型上直接做**全参数微调（Full Fine-Tuning）**有两大硬伤：

1. **显存爆炸**：如第 5 章算过的，优化器状态是权重的 6 倍；
2. **灾难性遗忘（Catastrophic Forgetting）**：小数据集上的几百步梯度，足以把基座在百万小时数据上学到的通用能力冲得七零八落！

2021 年微软研究院提出的 **LoRA（Low-Rank Adaptation）** 基于一个惊人的数学洞察：**大模型在适配下游新任务时，权重真正需要改变的增量矩阵 $\Delta \mathbf{W}$ 是极度“低秩”的**——就像一张 $2048 \times 1024$ 的大画布，其实只沿着十几个关键方向做调整就够了！

![图 9-1 LoRA 低秩适配核心原理：冻结巨型基座 W₀，用细腰旁路 B·A 精准微调](assets/diagrams/15_lora_math_anatomy.svg)

### 数学公式与三大关键设计
对于任意线性层 $\mathbf{h} = \mathbf{W}_0 \mathbf{x}$（其中 $\mathbf{W}_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ 彻底冻结），LoRA 旁边并接两个瘦长的小矩阵 $\mathbf{A} \in \mathbb{R}^{r \times d_{\text{in}}}$ 和 $\mathbf{B} \in \mathbb{R}^{d_{\text{out}} \times r}$（秩 $r \ll \min(d_{\text{in}}, d_{\text{out}})$，例如 $r = 16$）：

$$\mathbf{h} = \mathbf{W}_0 \mathbf{x} + \frac{\alpha}{r} \mathbf{B} \mathbf{A} \mathbf{x}$$

::: example 小数字手算：为什么必须“A 随机初始化、B 全零初始化”？
设 $d_{\text{in}}=4, d_{\text{out}}=4, r=1$：

- 如果 **$\mathbf{A}$ 和 $\mathbf{B}$ 都初始化为 0**：$\Delta \mathbf{W} = \mathbf{B}\mathbf{A} = \mathbf{0}$。但算梯度时：
  $$\frac{\partial L}{\partial \mathbf{B}} = \frac{\partial L}{\partial \mathbf{h}} (\mathbf{A}\mathbf{x})^\top = \mathbf{0}, \qquad \frac{\partial L}{\partial \mathbf{A}} = \mathbf{B}^\top \frac{\partial L}{\partial \mathbf{h}} \mathbf{x}^\top = \mathbf{0}$$
  **两个梯度全是 0！训练彻底死锁，一万年都不会动！**

- 如果 **$\mathbf{A}$ 用高斯/Kaiming 随机初始化（非零），$\mathbf{B}$ 严格全零初始化**：
  - 第 0 步前向：$\mathbf{B}\mathbf{A} = \mathbf{0} \cdot \mathbf{A} = \mathbf{0}$ → **开局输出与原模型 100% 逐位一致，绝不破坏基座！**（本项目 `train/train_sft.py` 第 273 行甚至用代码强断言 `lora_B` 初始和必须严格等于 `0`！）
  - 第 0 步反向：$\frac{\partial L}{\partial \mathbf{B}} = \frac{\partial L}{\partial \mathbf{h}} (\mathbf{A}\mathbf{x})^\top \ne \mathbf{0}$ → **第 1 步 $\mathbf{B}$ 立刻获得有效更新变成非零，第 2 步起 $\mathbf{A}$ 和 $\mathbf{B}$ 双双起飞！**
:::

### LoRA 四大超参怎么选？（老兵经验表）
| 超参 | 常见取值 | 底层含义与调参心法 |
| :--- | :--- | :--- |
| **`r` (秩 Rank)** | `8, 16, 32, 64` | 控制“表达容量上限”。常规风格/对齐/抗噪任务 **`r=16`** 是黄金甜点；注入海量新领域知识（如医学/代码）可升至 `32–64` |
| **`lora_alpha`** | 通常设 **`2×r`**（如 `r=16, α=32`） | 缩放系数 $\frac{\alpha}{r}$ 控制旁路输出幅值。改变 `r` 时若固定 $\alpha/r = 2$（或用 **rsLoRA** $\alpha/\sqrt{r}$），就不需要重新大幅调学习率 |
| **`target_modules`** | 注意力 4 矩阵 (`q,k,v,o`) 或 全 7 矩阵 (+`gate,up,down`) | 小数据/防遗忘优先只挂解码器注意力 4 层（28 层共 224 个张量 = `4,587,520` 参数）；大领域知识注入全挂 7 层 |
| **`trainable_dtype`** | **强烈推荐 `float32`** | 如第 2 章血泪复盘：基座 `bfloat16`，LoRA 权重用 `float32` 主权重，仅多占几 MB，永不遭遇小学习率 BF16 ULP 零更新陷阱！ |

*(进阶变体提示：**DoRA** 把权重拆成“方向 × 幅度”分别更新，逼近全参效果；**QLoRA** 把冻结基座 $\mathbf{W}_0$ 压成 4-bit NF4 存放，前向临时反量化成 BF16 与 LoRA 相加，专治显卡显存不足。)*

---

## 9.3 SFT 监督微调的三大生死细节（90% 的隐蔽 Bug 都藏在这里！）

写一个 SFT 训练循环看似简单，但只要下面三个细节错了一个，你的模型要么越训越傻，要么学会“永远闭嘴（空输出）”！

### 细节 1：Label Masking（前缀全填 `-100`，只监督答案 + EOS！）
回看第 6 章的完整输入序列：
```text
[前缀 Prefix: system + 50个音频Token + "language Chinese<asr_text>"]  [答案 Target: "今","天","天","气","不","错","<|im_end|>"]
```
如果把整条序列都拿去算交叉熵 Loss，模型就会把一半梯度浪费在去预测 `<|im_start|>user` 和无意义的 `<|audio_pad|>` 上！
**正确做法（见 `train/train_sft.py` 的 `collate_example`）**：

1. 分别对 `prefix` 和 `prefix + target + eos_token` 做编码，**严格断言 `prefix_ids == full_ids[:prefix_len]`**（防 BPE 分词在边界处把前缀最后一个字和正文第一个字粘连合并成一个新 Token——这是极其隐蔽的 Tokenizer 边界合并 Bug！）；
2. 将 `labels` 中下标 `< prefix_len` 以及所有 padding 位置全部设为 PyTorch 交叉熵的默认忽略哨兵值 **`-100`**；
3. **死守最后一个 `<|im_end|>`（EOS）**：绝不能因为 padding 掩码写错而把真实的 EOS 也抹成 `-100`！否则模型学不会什么时候停下来，推理时会无限循环复读直到撞上 `max_new_tokens` 截断！

### 细节 2：Shift-by-One（错位一位预测）验证！
自回归语言模型在第 $t$ 个位置的输出 `logits[:, t, :]`，预测的是**第 $t+1$ 个位置的真实词 `labels[:, t+1]`**！
因此计算交叉熵前必须精确错位：
```python
def supervised_nll_sum(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, int]:
    shift_logits = logits[:, :-1, :].float()  # 丢掉最后一个位置的预测（后面没词了）
    shift_labels = labels[:, 1:]              # 丢掉第 0 个输入（前面没上下文预测它）
    nll = torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    )
    mask = shift_labels.reshape(-1) != -100
    return nll[mask].sum(), int(mask.sum().item())
```
检查一下下标：假设 `prefix` 长度为 `P`（下标 `0 .. P-1`），答案第一个字位于 `P`，则在 `shift_logits` 中，恰好是下标 `P-1`（即 `<asr_text>` 所在位置）负责预测 `shift_labels` 的下标 `P-1`（也就是原序列下标 `P` 的答案首字）！

### 细节 3：梯度累积时的“Token 均值 vs 样本均值”陷阱！
假设你的有效 Batch = 8 条样本（`micro_batch_size=1, grad_accum=8`），第 1 条只有 3 个汉字，第 2 条有 40 个汉字：

- **错误写法**：每条微批次先除以本条 Token 数算平均 Loss，再除以 8 累加。后果：3 个字短句里的每个字权重高达 40 字长句的 **13.3 倍**，短句（尤其是噪声/截断短句）主导整个梯度！
- **正确写法（本项目实装方案）**：在累积窗口的 8 条微批次内，先算出整个窗口的**总监督 Token 数 $N_{\text{win}}$**，每条微批次反传 `nll_sum / N_win`！这样整个大 Batch 内**每一个监督汉字的梯度权重严格相等**，无论 `micro_batch_size` 设成 1 还是 8，数学梯度 100% 精确等价！

---

## 9.4 真实 A100 大案复盘：为什么“好数据”比“调学习率”重要十倍？

下面这段来自本仓库 `docs/qwen3-asr/06_risks_and_decisions.md` 和 `00_progress.md` 的真实工程复盘，是任何纯理论教材都给不了你的无价宝藏！

### 第一幕：灾难降临——SFT 跑了 30 步，竟然越训越烂！
在 Phase 6 Track 3 中，我们升级了真实物理混响与房间噪声（OpenSLR 28 实测 RIR），满怀期待地在 AISHELL-1 训练集上启动了常规学习率 SFT（`lr=2e-5` / `1e-5`）。
结果第 30 步验证集评测一出，所有人都傻眼了：

- 不仅抗噪效果微弱，语言流畅度还明显倒退（比如把“工业、建筑”听错成“工业借助”）；
- 抽查输出发现：**Base 模型原本 500/500（100%）条都会自然加上中文逗号、句号；SFT 才训 30 步，竟然把 97% 的标点符号抹得干干净净！**

### 第二幕：法医式排错——揪出“隐形数据分布冲突”
真相一查就水落石出：

- `Qwen3-ASR-0.6B` 官方底座是在带标点的海量中文语料上预训练出来的，它内部把“语音停顿 → 预测逗号 `，` → 开启下一分句”织进了极深的语言先验里；
- 而开源数据集 **AISHELL-1 的官方金标文本是完全去掉所有标点的纯汉字串！**
- 于是每当音频里出现一个自然停顿，Base 模型本能地想输出 `，`，SFT 的交叉熵损失就啪啪扇它耳光：“错！不准打逗号！”——结果 **90% 的 LoRA 梯度能量全耗在了“暴力抹除预训练标点先验”上，顺带把停顿处的中文语言模型概率分布给砸烂了！** 只有不到 10% 的梯度在真正学习抗混响和抗噪声！

### 第三幕：外科手术级解药——Clean 自对齐标点恢复（零字符改动强断言！）
怎么既保留 AISHELL-1 100% 准确的官方金标汉字，又不破坏底座的中文标点先验？
我们在 `scripts/align_train_punctuation.py` 设计了一个极其干净的算法（**严格只用于训练集，绝不碰验证集/测试集/OOD！**）：

1. 用**冻结的 Base 模型**去解码训练集对应的**干净音频（Clean）**，拿到带自然韵律标点的预测文本；
2. 用 Levenshtein 字符对齐算法，把 Base 在 Clean 音频里自然打出的标点符号，精准“投影”回官方金标汉字 `text` 之间，生成新字段 `text_punct`；
3. 对全量 19,685 条训练样本执行**铁血零容错断言**：`normalize_zh_cer_v1(text_punct) == normalize_zh_cer_v1(text)`（`char_mismatches = 0`，一个字都不许篡改！）。

### 第四幕：奇迹发生——真实会议室 Sim-to-Real 历史性逆转！
切换为 `target_text_field: text_punct` 重跑 SFT（`sft_06b_zh_20261124T194600_step000240`）后，结果令人拍案叫绝：

- **标点保持率**：`100.0%`（一举消除先验冲突！）；
- **域内 `val`（2500 条）**：总字编辑从 Base 的 `2347` 暴降至 **`2230`（净减 `-117` 编辑！）**，物理退化三场景净减 **`-58` 编辑（Bootstrap 胜率 $P = 99.65\%$）**；
- **封存 `test`（2500 条）**：总字编辑 `2053 → 1985`（**`-68` 编辑**，95% CI `[-0.3256, -0.0438] pp` 严格为负，$P = 99.70\%$）；
- **零样本真实会议 `AliMeeting Eval` OOD（1000 条）**：彻底逆转了早期合成数据上远场负迁移（`682 → 695`）的魔咒，实现**近讲 `-16` 编辑 + 真实 8 麦阵列远场 `-21` 编辑 = 总体 `-37` 编辑**的真实世界突破！

::: warning 这一仗给所有后训练工程师的一条永恒金律
**在怀疑模型不行或猛调学习率之前，先打开 20 条你的训练样本和 Base 模型的原始输出逐字对比！** 只要你的训练标签格式（标点、大小写、数字全半角、思维链格式、空格换行）与基座的预训练先验打架，SFT 就会把宝贵的容量全烧在“洗脑改习惯”上，而不是学真本事！
:::

SFT 已经把模型从“听不清噪声”教成了“高分好学生”（净减 117 个错字）。但好学生依然会在某些特别相似的同音词上犯迷糊（比如“监管”写成“结款”，“失业”写成“事业”）。下一章，我们祭出后训练第二把手术刀——**DPO 直接偏好优化**，专门专治这些顽固错题！
