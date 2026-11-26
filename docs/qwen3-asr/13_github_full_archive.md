# GitHub 全量资产归档说明（代码、文档、数据、日志、权重与结论）

最后更新：2026-11-26

## 背景

项目已完成 Phase 0 至 Phase 6 Track 3B 全链路（`Base → SFT → DPO → RL → 封存 test.jsonl + AliMeeting Eval OOD`）并持久化在 Google Drive `MyDrive/qwen3-asr/`。为实现私有 GitHub 仓库（`https://github.com/cluster1900/qwen-asr-posttrain`，`isPrivate: true`）的一站式全量备份与可复现交付，在规避 Git 单文件 `100 MB` 与单包 `2 GB` 硬限制及不污染克隆体积的前提下，通过 `sub2` 服务器直连 Google Drive 并按 `< 1.55 GiB` 分卷，将全部数据、不可变发布权重、83 次训练与评测运行、服务器与本机日志、以及结构化结论全量同步至 GitHub。

## 范围

- **Git `main` 分支（无 GPG 签名提交）**：
  - 全部源码、配置、脚本、单测、Notebook 与 `docs/qwen3-asr/` 全部文档。
  - 新增 `conclusions/` 目录（37 份核心 JSON/CSV/Markdown 终局结论与配对 Bootstrap 数学期望文件，共 `252 KB`），便于在 GitHub 网页端直接审阅。
  - 严格遵守 `AGENTS.md`：`references/` 保持在 `.gitignore` 中，不提交其中任何文件；严禁提交任何 OAuth Token、私钥或会话凭证。
- **GitHub Releases（5 组主题 Release，共 30 个资产，`15.0 GiB` 归档总计覆盖 `4,408` 个源文件）**：
  1. `archive-data-v1`：Google Drive `data/` 下全部原始与派生数据集（OpenSLR 28、AISHELL-1 Phase 1 Pilot、AISHELL-1 SLR28 物理增强与标点恢复清单、只读 AliMeeting Eval OOD）。
  2. `archive-releases-and-dpo-v1`：全部已独立读回发布的不可变模型权重（`releases/{sft,dpo,rl}/`）与全部 DPO 偏好对及离线 reference logprob 缓存（`dpo/`）。
  3. `archive-runs-phase0-to-track2-v1`：Phase 0 至 Phase 6 Track 2 全部 19 次运行的完整 Checkpoint、预测、评分、门禁与报告。
  4. `archive-runs-track3-full-v1`：Phase 6 Track 3 & 3B 全部 64 次运行（Base/SFT/DPO/RL 及 Stage 4 封存 `test.jsonl` 与 AliMeeting Eval OOD 终局产物）。
  5. `archive-logs-and-local-results-v1`：`sub2` 全量脱敏编排/保活/训练日志与 Mac 本机非重复历史实验产物。

## 设计与完整资产清单

### 1. `archive-data-v1`（12 资产，10.90 GiB，38 源文件）

- 链接：`https://github.com/cluster1900/qwen-asr-posttrain/releases/tag/archive-data-v1`

| 资产文件名 | 字节数 | SHA-256 |
| --- | ---: | --- |
| `data_aishell1_phase1_pilot.tar.part_aa` | `1,625,292,800` | `c8312340381f0c1cd86d8a2cd29f960116c37e4bf4b4e155f7be70edc3d7f545` |
| `data_aishell1_phase1_pilot.tar.part_ab` | `1,625,292,800` | `514d7cd806193f037226dc278ad86a13c45e62f153ca0362778df62681046118` |
| `data_aishell1_phase1_pilot.tar.part_ac` | `1,625,292,800` | `1060ca9ad41c75f96f23cd7b6779f50bac299632ff35fb2aa41c4e97049a6298` |
| `data_aishell1_phase1_pilot.tar.part_ad` | `1,625,292,800` | `e90bbbfb10962aef936a6a68634e248ae7ee27cdaa97b62da6c9cf4dab4667e5` |
| `data_aishell1_phase1_pilot.tar.part_ae` | `22,691,840` | `88da24d9ea2dcb28f64e727e575553c549e9cb40a155960ff5f006851d3fb139` |
| `data_aishell1_slr28_20261124.tar.part_aa` | `1,625,292,800` | `b11602facd233b03cfc917799021d23748e88b0b3150ceed96a1f340d8fecc57` |
| `data_aishell1_slr28_20261124.tar.part_ab` | `1,625,292,800` | `24902b73f40b65d427253bc252e24ec45aa52f70ac960a6889a4b1cf4f741e10` |
| `data_aishell1_slr28_20261124.tar.part_ac` | `496,148,480` | `bda12c423604ba35bb63bdcc607e5e751481fbbdb8b01048c0f59238bb903a94` |
| `data_alimeeting_eval_v1_seed42.tar` | `118,917,120` | `41249fe5d6c8781f44c71ba3318b84d8f337bf28fa19d2b16dd0cb4d46c9476f` |
| `data_slr28_rirs_noises_v1.zip` | `1,311,166,223` | `3b50cfde915b3984738169b4beb341e9f6b8062ae4c2076146c5db71c2c05dc7` |
| `archive-data-v1_SHA256SUMS.txt` | `1,043` | `85b535ad3da7169a1c46915e41ae8bdea30f50f1afc192ac0433415324299988` |
| `archive-data-v1_FILE_MANIFEST.json` | `10,480` | `edc35b86ab0cd4143b0cfa7b1c7f3d7afab3c133750cb96839869d3e61826cb3` |

### 2. `archive-releases-and-dpo-v1`（4 资产，83.6 MiB，131 源文件）

- 链接：`https://github.com/cluster1900/qwen-asr-posttrain/releases/tag/archive-releases-and-dpo-v1`

| 资产文件名 | 字节数 | SHA-256 |
| --- | ---: | --- |
| `releases_all_sft_dpo_rl.tar.gz` | `80,891,361` | `31851b0c67b92d7ae0f7e70d0ca8e3fca2f223e3859bd1a0d3a2961633fcfeaf` |
| `dpo_pairs_and_ref_logprobs_all.tar.gz` | `6,704,240` | `86d822cc100e5dd0051809a321316f08a4508f0769c706267f723de6b1e17ca0` |
| `archive-releases-and-dpo-v1_SHA256SUMS.txt` | `201` | `1dc0f89855e1455ddbe0c4eeda11d859636ed91774e655341180b77022a4103a` |
| `archive-releases-and-dpo-v1_FILE_MANIFEST.json` | `26,533` | `50c36e1283081313b9d2436cf5eba4249f744e3621505ff91eff6bee70163343` |

### 3. `archive-runs-phase0-to-track2-v1`（3 资产，650.0 MiB，591 源文件，19 次运行）

- 链接：`https://github.com/cluster1900/qwen-asr-posttrain/releases/tag/archive-runs-phase0-to-track2-v1`

| 资产文件名 | 字节数 | SHA-256 |
| --- | ---: | --- |
| `runs_phase0_to_phase6_track2.tar.gz` | `681,452,983` | `1a29385c94f7a2cff0cd1acfccb000197f4cb8e6fec0f5ad5bb119c6ece0c8b3` |
| `archive-runs-phase0-to-track2-v1_SHA256SUMS.txt` | `102` | `f3276cc64d0b02abf23d4516899fef05706f92c475ff74b12c3fef9f3beb6921` |
| `archive-runs-phase0-to-track2-v1_FILE_MANIFEST.json` | `117,437` | `1d8fe638da5eb38d798c8e61b4cb01787450edcb531dd33441a3a0b5ab766b91` |

### 4. `archive-runs-track3-full-v1`（7 资产，2.94 GiB，2,973 源文件，64 次运行）

- 链接：`https://github.com/cluster1900/qwen-asr-posttrain/releases/tag/archive-runs-track3-full-v1`
- 说明：`runs_track3_base_and_sft.tar.gz` 合并后完整 SHA-256 为 `cafdb1a30773d5b810098c56e815c01035c6660de3cd3ef89f16183166c58731`。
- 2026-11-26 按归档清单核对，RL 第二轮 10 组中只有 `20261125T052009_rl_06b_zh_seed42` 的逐步文件进入 `runs_track3_rl.tar.gz`。十组发布判定均可在运行机的末次监控记录中核对。[第二轮逐组实测值](14_dpo_rl_diagnosis.md#第二轮监控读数)

| 资产文件名 | 字节数 | SHA-256 |
| --- | ---: | --- |
| `runs_track3_base_and_sft.tar.gz.part_aa` | `1,625,292,800` | `516cd335eb72b5792436180234032908ae498b13400c8e7b1e3352e3a7f4dae4` |
| `runs_track3_base_and_sft.tar.gz.part_ab` | `545,623,609` | `9ab3ab659411f6de51868ec75a4a580f6613ed287375a892088b135b4ba4b3f7` |
| `runs_track3_dpo.tar.gz` | `509,495,729` | `b4a5d4fe3b179057bc32349288a202239941a9336a3378b5139ba339b645f769` |
| `runs_track3_rl.tar.gz` | `475,908,972` | `72277718178481cca4202287da1b7a25be63cf716a62305cb472014a77eac09d` |
| `runs_track3_stage4_test_and_alimeeting_ood.tar.gz` | `939,678` | `224998da9cbbb0d2783ebce3bc69a442f468ef0cc77ff90701abbfd5467f3cdc` |
| `archive-runs-track3-full-v1_SHA256SUMS.txt` | `625` | `f54e4c2cfab0acbcbd60fb3a27a4511e885645b07023bef69a872462d39b9bd6` |
| `archive-runs-track3-full-v1_FILE_MANIFEST.json` | `580,138` | `b32b33a4b8b7cbf394f284ff784a6897e46d57ba29fc4380e9b4537a41ce0ca5` |

### 5. `archive-logs-and-local-results-v1`（4 资产，430.1 MiB，675 源文件）

- 链接：`https://github.com/cluster1900/qwen-asr-posttrain/releases/tag/archive-logs-and-local-results-v1`

| 资产文件名 | 字节数 | SHA-256 |
| --- | ---: | --- |
| `sub2_orchestration_and_keepalive_logs.tar.gz` | `1,755,485` | `240e0014c5816abc7477fae7ad54fb6eb4791e8715fae29dc05e956a39c67ec8` |
| `mac_unique_results_and_logs.tar.gz` | `449,221,089` | `8beac7b1b22bda4c8b6784bfd549a0942a2e2d02915522cf9e5291ba1c7e3ecf` |
| `archive-logs-and-local-results-v1_SHA256SUMS.txt` | `111` | `f4422a22ace8a0249333fac83a61391527666bf41048ed8d5abfa65b3ab6163b` |
| `archive-logs-and-local-results-v1_FILE_MANIFEST.json` | `49,663` | `b464d82c57627a9f3f55ecd8c9ef779ebf586ccac4d9dcac44f82845f01cc37d` |

### 6. `book-v1.0.0`（书籍成品 Release：4 资产，11.45 MiB，HTML / PDF / EPUB3）

- 链接：`https://github.com/cluster1900/qwen-asr-posttrain/releases/tag/book-v1.0.0`
- 说明：《从零到一：大模型底层原理、GPU 算子优化与任意模型后训练实战》三格式成品；配套 GitHub Pages 工作流 `.github/workflows/book-pages.yml` 与 `book/scripts/build_pages.py` 直接从该 Release 拉取并校验 HTML 成品部署在线阅读站。

| 资产文件名 | 字节数 | SHA-256 |
| --- | ---: | --- |
| `zero-to-one-llm-posttraining-v1.0.0.html` | `679,810` | `b73031d419cefcc52e6a3a132051ebb1be9f91f862225774c93b161c72ad2df8` |
| `zero-to-one-llm-posttraining-v1.0.0.pdf` | `10,442,936` | `b9f2a38d3f367e334e33a33e053efef2eaf46fbc9a5bdd7686e3ef1502d99814` |
| `zero-to-one-llm-posttraining-v1.0.0.epub` | `885,215` | `30734bed7553825d9a9ff44d4582491ab8b79aeaa97f846ccd8dd928d403b80f` |
| `book-v1.0.0_SHA256SUMS.txt` | `320` | `1e79f0fdba5a1f04b51a8b620e7c20fc1b1043905c292c60e17af5a4a3282ef9` |

### 7. 标准恢复命令示例

```bash
# 下载指定 Release 全量资产并核对校验和
gh release download archive-data-v1 -R cluster1900/qwen-asr-posttrain -D ./restore_data
cd ./restore_data
sha256sum -c archive-data-v1_SHA256SUMS.txt
cat data_aishell1_slr28_20261124.tar.part_* > data_aishell1_slr28_20261124.tar
tar -xf data_aishell1_slr28_20261124.tar

# 恢复 Track 3 全部 64 次运行与 Stage 4 终局产物
gh release download archive-runs-track3-full-v1 -R cluster1900/qwen-asr-posttrain -D ./restore_runs
cd ./restore_runs
sha256sum -c archive-runs-track3-full-v1_SHA256SUMS.txt
cat runs_track3_base_and_sft.tar.gz.part_* > runs_track3_base_and_sft.tar.gz
tar -xzf runs_track3_base_and_sft.tar.gz
tar -xzf runs_track3_dpo.tar.gz
tar -xzf runs_track3_rl.tar.gz
tar -xzf runs_track3_stage4_test_and_alimeeting_ood.tar.gz
```

## 测试

- Google Drive 下载阶段：全部 3,733 个云端文件逐文件核对远端 MD5 与字节数，零失配。
- GitHub Releases 上传阶段：全部 30 个资产上传后逐个校验远端 `state == "uploaded"` 与精确字节数一致，零截断。
- 安全与合规校验：确认仓库属性为 `isPrivate: true`；`sub2` 日志打包阶段自动跳过含凭证模式文件；上传后立即清理 `sub2` 全部 `/tmp/.gcp_tok*` 与 `/tmp/.gh_tok*`；`references/` 未入库。
- 代码与文档合同测试：运行 `.venv/bin/python -m unittest discover -s tests -v`，全量 163 项单测通过。

## 验收

- `main` 分支包含全部代码、配置、单测、文档及 `conclusions/` 核心结论且已推送至 `origin/main`（无 GPG 签名）。
- 全部 5 组 GitHub Releases 共 30 个资产在 `cluster1900/qwen-asr-posttrain` 上状态均为 `uploaded` 且字节数与 `SHA256SUMS.txt` 逐位匹配。
- Google Drive 与 GitHub Releases 形成双云端全量冗余备份，无残留临时 Token 或活跃 Colab GPU。

## 影响

- 不修改任何历史代码、训练配置、门禁阈值或冻结哈希；Git 克隆保持轻量（`conclusions/` 仅 `252 KB`），全量 15.0 GiB 大数据与模型权重通过 GitHub Releases 按需下载恢复。
