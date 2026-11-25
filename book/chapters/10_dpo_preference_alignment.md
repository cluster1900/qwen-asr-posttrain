# 第 10 章 后训练第二站：DPO 直接偏好优化与“静音幻觉”数学破案

> **本章情绪定心丸**：很多人在论文里看到 DPO（Direct Preference Optimization）那长长的一串希腊字母、KL 约束和 Bradley-Terry 偏好模型，当场就头晕了。别慌！这一章我们只用高中数学的**对数加减法**和**指数消元**，5 分钟让你自己把 DPO 公式从头推到尾！更精彩的是，我们还要带你亲历一桩堪比侦探小说的真实 A100 事故——为什么标准 DPO 会把一句中文“就有一零个，加级贫困县”活生生训出 **`Human Rights Watch`**，以及如何仅用一行 `.detach()` 从数学根源将其彻底秒杀！

---

## 10.1 SFT 已经很强了，为什么还需要 DPO？

SFT 只教模型一件事：**“把标准答案 $y_w$ 的概率拉高”**。但它从来没明确告诉模型：**“哪个错题 $y_l$ 特别容易和正确答案混淆、一定要压下去！”**

比如在带噪语音里，SFT 模型听“纳入监管”时，给正确答案“监管”打了 `48%` 概率，给形近/音近错词“结款”打了 `50%` 概率——一做贪心解码 `argmax`，就差这区区 2% 功亏一篑！

- 如果你接着跑 SFT，它对所有原本就 99% 对的字也会胡乱推梯度，很容易过拟合；
- **DPO 的思想极其精准**：直接把一对**偏好样本 $(x, y_w, y_l)$**（$y_w$ 是好答案 Chosen，$y_l$ 是模型自己常犯的错答案 Rejected）摆在模型面前——**一手把 $y_w$ 往上托，一手把 $y_l$ 往下踩，同时用一根橡皮筋（冻结的参考模型 $\pi_{\text{ref}}$）拴住它，不许它跑太偏毁掉原本的好习惯！**

---

## 10.2 用高中数学 4 步推导出 DPO 闭式公式！

早期 OpenAI 做 ChatGPT 用的是 **RLHF-PPO**：先训一个单独的奖励模型（Reward Model）给回答打分，再用极其难调的 PPO 强化学习去优化策略。2023 年斯坦福团队发现：**根本不用单独训奖励模型，最优策略和奖励函数之间有一条简单的代数换算式！**

跟着我一步步推，每一步都不超过高一数学：

1. **第一步：写出带“橡皮筋（KL 约束）”的强化学习目标**
   我们想找一个策略 $\pi_\theta(y|x)$，既让奖励 $r(x, y)$ 尽量大，又不要离初始参考模型 $\pi_{\text{ref}}(y|x)$ 太远（$\beta > 0$ 控制橡皮筋松紧，常见取值 `0.1`）：
   $$\max_{\pi_\theta} \; \mathbb{E}_{y \sim \pi_\theta}\!\left[ r(x, y) - \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} \right]$$

2. **第二步：这个优化问题有精确闭式解！**
   高中不等式（或拉格朗日乘子法）可以证明，最优策略满足：
   $$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\!\left(\frac{1}{\beta} r(x, y)\right)$$
   其中 $Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp\!\left(\frac{1}{\beta} r(x, y)\right)$ 是只跟输入 $x$ 有关、与输出 $y$ 无关的归一化常数！

3. **第三步：两边取对数 $\ln$，把未知奖励 $r(x, y)$ 反解出来！（高一移项）**
   $$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

4. **第四步：代入好坏对比（Bradley-Terry 模型），讨厌的 $\log Z(x)$ 神奇消掉！**
   人类偏好 $y_w$ 胜过 $y_l$ 的概率定义为 $\sigma(r(x, y_w) - r(x, y_l))$（其中 $\sigma(u) = \frac{1}{1+e^{-u}}$ 是 Sigmoid 函数）。把第 3 步的 $r$ 代进去相减——**因为 $y_w$ 和 $y_l$ 对应同一个输入 $x$，两个 $+\beta \log Z(x)$ 一减正好归零！** 于是我们直接得到闻名世界的 **DPO 损失函数**：

$$\mathcal{L}_{\text{DPO}}(\theta) = -\log \sigma\!\left(\beta \left[ \underbrace{\left(\log \pi_\theta(y_w|x) - \log \pi_\theta(y_l|x)\right)}_{\text{当前策略的偏好边际 Margin}} - \underbrace{\left(\log \pi_{\text{ref}}(y_w|x) - \log \pi_{\text{ref}}(y_l|x)\right)}_{\text{参考模型的原始边际 Ref Margin}} \right]\right)$$

::: example 小数字手算：一眼看懂 DPO 在干嘛！
设 $\beta = 0.1$，某条音频上 SFT 参考模型把好句 $y_w$ 的整句对数概率算作 `-12.0`，把错句 $y_l$ 算作 `-10.0`（参考模型目前更偏爱错句！参考边际 `ref_margin = -12.0 - (-10.0) = -2.0`）。

- **第 0 步刚开始训练时**：$\pi_\theta \equiv \pi_{\text{ref}}$，所以 `policy_margin` 也是 `-2.0`，括号里的差值 $\Delta = 0.1 \times (-2.0 - (-2.0)) = \mathbf{0}$！
  - 此时 DPO 损失精确等于 $-\log \sigma(0) = -\log(0.5) = \mathbf{0.693147}$！
  - **绝佳 Sanity Check**：以后跑任何 DPO 实验，**第 1 步打印出的 loss 必须无限接近 `0.693147`**！如果不是（比如一开始就 0.2 或 5.0），99% 是你的 Reference Logprob 缓存和策略前向没对齐！
- **训练几步后**：策略学会偏好正确答案了，`policy_margin` 升到 `+1.0`，则 $\Delta = 0.1 \times (1.0 - (-2.0)) = +0.3 > 0$，损失降为 $-\log\sigma(0.3) = \mathbf{0.554}$！而且如果某个样本已经拉得够开（$\Delta$ 很大），Sigmoid 导数 $1 - \sigma(\Delta)$ 自动趋近 0，**自动刹车不过拟合**！
:::

---

## 10.3 工程提速与省显存利器：离线 Reference Logprob 缓存

看上面的 DPO 公式，参考模型 $\pi_{\text{ref}}$ 是**全程冻结一动不动的**！它的 $\log \pi_{\text{ref}}(y_w|x)$ 和 $\log \pi_{\text{ref}}(y_l|x)$ 在整个训练过程中永远是常数！

- **笨办法**：训练时在显存里同时加载一份策略模型、一份参考模型，每步多跑两遍无意义的前向，显存和时间双双翻倍。
- **工业最佳实践（见本项目 `train/reference_logprob.py`）**：
  1. DPO 启动前，单独跑一遍 `reference_logprob.py`，用冻结的 SFT 发布版一次性算出全部 train/val 偏好对的 FP32 序列和 `ref_chosen_logp`、`ref_rejected_logp`，写入不可变 JSONL 缓存；
  2. 写完自动随机抽检 20–40 条重算一遍，断言 `max_abs <= 1e-4`（本项目实测 `max_abs = 0.0`！）；
  3. 正式跑 `train_dpo.py` 时，**Step 0 先抽查 20 条核对当前初始策略与离线缓存之差为 0**，确认对齐后全程只需在 GPU 里放**一份模型**！

---

## 10.4 侦探级真实事故复盘：为什么 DPO 把中文静音训出 `Human Rights Watch`？

如果你照搬市面上的开源 DPO 脚本（如 TRL 的默认 `DPOTrainer`）直接去训语音识别或任意等长/不等长序列纠错任务，**大概率会撞上下面这个让无数团队百思不得其解的鬼故事！**

### 案发现场（2026-10-17 至 2026-10-20 真实 A100 日志）
在 `Qwen3-ASR-0.6B` 上，我们用 SFT Step 30 自己产生的预测错误构造了 7,456 对训练偏好对（金标为 `chosen`，SFT 错预测为 `rejected`），满怀信心跑标准整句序列和 DPO。结果连续四轮都在 Step 10 触发门禁熔断！
打开出错样本一看，简直离谱到让人怀疑人生：

- 音频是一段中间被静音丢包（`dropout`）的中文新闻，SFT 原来能稳稳识别为：
  `就有一零个，加级贫困县`

- 仅仅经过 10 步温和的 DPO（`lr=5e-6`），模型在静音切口处竟然突然吐出一串流利的英文：
  `就有一零个Human Rights Watch`（瞬间暴增 +14 个字编辑！）

- 除此之外，2,500 条验证集里还有几十条莫名其妙地多插逗号、或者前半句没变后半句突然级联跑偏（如 `纳入监管` → `纳入结款`）！

### 数学破案：揪出隐藏在整句序列和里的两大元凶！

![图 10-1 为什么朴素 DPO 会把中文语音训出英文幻觉？以及直通式差异掩码如何一键根治](assets/diagrams/17_dpo_mismatched_mask.svg)

逐个 Token 拆解 DPO 梯度后，两个深层数学漏洞暴露无遗：

#### 元凶一：无声学支撑的伪偏好对 & 不等长序列的 $\log Z$ 倾斜
1. 在 `dropout` 静音丢包样本里，音频中间 40%–55% 物理上是**彻底静音（全是 0）**！SFT 没听到声音所以漏写了那 2 个字（`rejected` 更短），可金标文本 `chosen` 里硬是有那 2 个字！用这种样本做 DPO，等于**逼迫模型在“完全没有声音的静音段里硬编汉字”**——模型只好疯狂调用语言脑补，直接催生出静音幻觉！
2. 当 `chosen` 比 `rejected` 长（比如多 2 个字，或 `rejected` 里多了标点逗号），整句序列和 $\sum_t \log \pi_\theta(y_t|c_t)$ 对长度不等的两边施加不对等的配分函数梯度 $\sum_t \nabla \log Z(c_t)$：压低 `rejected` 的过程会把所有常见中文词和逗号的概率一起往下按，**把挤压出来的概率质量推向低频离群词（如预训练语料里常见的英文新闻专名 `Human Rights Watch`）！**

#### 元凶二（全书最精彩的数学洞见！）：共享后缀位点（$t > t_0$）的“有毒负梯度”！
就算我们把所有 `dropout`、不等长、带标点对全部过滤掉，只留下 **100% 干净的等长单字替换对**（例如 `chosen = 纳入[监]管体系`，`rejected = 纳入[结]管体系`，仅在第 $t_0$ 个字“监 vs 结”不同），标准整句 DPO **依然会翻车**！为什么？

写出第 $t$ 个位置的梯度（令 $c_t^+$ 为好句前缀，$c_t^-$ 为坏句前缀）：

- **在共享前缀（$t < t_0$，“纳”、“入”）**：$c_t^+ \equiv c_t^-$ 且目标字相同，正负两项一减精确抵消为 **`0`**。
- **在唯一差异位点（$t = t_0$，“监” vs “结”）**：前面都是“纳入”，所以上下文 **$c_{t_0}^+ \equiv c_{t_0}^-$ 完全相同**！代入第 1 章的交叉熵公式 $\log \pi = z_y - \log Z(c)$：
  $$\log \pi_\theta(\text{"监"}|c_{t_0}) - \log \pi_\theta(\text{"结"}|c_{t_0}) = \left(z_{\text{监}} - \cancel{\log Z(c_{t_0})}\right) - \left(z_{\text{结}} - \cancel{\log Z(c_{t_0})}\right) = z_{\text{监}} - z_{\text{结}}$$
  **天呐！配分函数 $\log Z$ 在这里 100% 精确抵消！梯度纯粹只抬高“监”、压低“结”，对词表里其他 151,934 个词零扰动！**

- **但在共享后缀（$t > t_0$，“管”、“体”、“系”、“`<|im_end|>`”）**：**噩梦发生了！**
  - 尽管两边目标字**都是正确的“管、体、系”**，但好句的历史前缀里是“监”（$c_t^+$），坏句的历史前缀里是“结”（$c_t^-$）——**$c_t^+ \ne c_t^-$！**
  - 因此两个配分函数 $\log Z(c_t^+)$ 和 $\log Z(c_t^-)$ **不再抵消！** 统计发现，共享后缀平均占整句话 **58.9% 的 Token**，却吃掉了高达 **88% 的非零梯度**——整句 DPO 在拼命给“管、体、系”和句末 `<|im_end|>` 施加**负梯度惩罚 $-\nabla \log \pi_\theta(y_t^+ | c_t^-)$**，活活把正常续写和正常停顿给毁了！

### 外科手术三板斧：彻底根治并登顶真实 OOD 总冠军！
看清数学根因后，我们在 `train/train_dpo.py` 和 `train/dpo_contract.py` 祭出三板斧：

1. **第一斧：数据精准提纯（声学同音单字对）**
   - 训练集排除无声学支撑的 `dropout`、仅保留物理退化三场景 `[noise, reverb, far_field]` 的等长原位替换对（`aligned_substitution_only: true`, `max_edit_count: 1`, `exclude_clean_identical_rejected: true`），并过滤极端负参考边际（`ref_margin ∈ [-4.0, 0.0]`），让每一条偏好对都是货真价实的声学同音消歧！
2. **第二斧：直通式差异 Token 梯度掩码（`loss_token_scope: mismatched_only`）**
   - 前向依然算完整整句 `full_nll`（保证与离线缓存的 `ref_logp` 100% 标尺对齐），**反向只允许差异位点 $t_0$ 回传梯度**！核心实现只有 3 行（直通估计器 STE，见 `train/train_dpo.py` 第 217–219 行）：

```python
import torch

def dpo_mismatched_mask_nll(sup_nll: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    """前向保持整句 NLL（与离线 Ref 对齐），反向仅在 chosen != rejected 的差异位点求导。"""
    full_nll = sup_nll.sum()
    diff_nll = sup_nll[token_mask].sum()
    return diff_nll + (full_nll - diff_nll).detach()
```

   共享后缀与 `<|im_end|>` 梯度严格归零，$\log Z$ 100% 精确抵消！

3. **第三斧：父版本 Dropout 锚定（Parent Anchor）+ 余弦周期解耦（`scheduler_max_steps: 100`）**
   - 每步附加一个极轻的锚定正则（权重 `0.05`）：让模型在 `dropout` 静音样本上保持与父版本 SFT 预测分布一致（教师是父版本预测而非金标，绝不逼它静音编字！），彻底稳住未受训的静音场景！

### 战果辉煌：全流程跨两套基准登顶总冠军！
这套精准手术跑出的发布版本 **`dpo_06b_zh_20261125T005003_step000040`** 交出了无可挑剔的成绩单：

- **域内 `val`（2500 条）**：总字编辑从 Base `2347` → SFT `2230` → **DPO `2222`（相对 Base `-125`、相对 SFT `-8`）**，物理退化三场景净降 `-67 vs Base / -9 vs SFT`，偏好准确率 `66.25%`，标点 `100%` 完整，静音幻觉 **`0`**！
- **真实会议 OOD `AliMeeting Eval`（1000 条，全量 A100）**：**Base `1038` → SFT `1001` → DPO `998` 字编辑（`-40 vs Base`）、词编辑 `965 → 939 → 931`（`-34 vs Base` / `-8 vs SFT`，近场和远场零回退！）**，500 簇配对 Bootstrap 95% CI **`[-0.4746, -0.0243] pp` 严格为负**（胜率 $P_{\text{CER}} = 98.50\%$）！

::: note 本章小结
1. DPO 用闭式代数代换省掉了独立 Reward Model，本质是用参考模型当橡皮筋拉大好坏对数概率边际；第 1 步 Loss 必为 $\ln 2 \approx 0.693147$。
2. 永远先离线缓存 `ref_logp` 并做重算校验，训练时省一半显存与算力。
3. 在纠错/代码修改/ASR 等“局部差异”偏好任务中，**务必警惕不等长与共享后缀上的 $\log Z$ 不抵消问题**！直通式差异 Token 掩码 + 父版本 Anchor 正则，能把有毒梯度清零，让 DPO 稳如手术刀！
:::
