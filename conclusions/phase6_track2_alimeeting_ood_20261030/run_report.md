# Phase 6 Track 2: AliMeeting Eval Real-World OOD Four-Stage Benchmark Report

- **run_id**: `20261030T130500_phase6_alimeeting_ood_06b_zh_seed42`
- **dataset**: `alimeeting_eval_openslr119` (`license=CC-BY-SA-4.0`, `usage_policy=read_only_ood_evaluation`, `trainable=False`)
- **split**: `ood_eval` (`1000` rows = `500` paired utterances × 2 scenarios across `8` meetings, `23` speakers)
- **ood_manifest_sha256**: `0b73e82d1608d8a11db89cf7a039f9c57c207e5ed9647f2060cf360f5f7b07e2`
- **base_model**: `Qwen/Qwen3-ASR-0.6B` (`revision=5eb144179a02acc5e5ba31e748d22b0cf3e303b0`)
- **eval_decode_sha256**: `4a4fee2b67884438faa196909749fcc8dcc6126b175554ad19992bf9b4f2f642`
- **wer_tokenizer**: `jieba==0.42.1` (`hmm=False`)

## 1. Overall OOD Trajectory (CER + Locked `jieba` WER)

| Stage | Release ID | Char Edits | Δ vs Base | Micro CER | Word Edits | Δ Word vs Base | Micro WER | Far−Near CER Gap | Mean Greedy Reward |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `BASE` | `base_qwen3_asr_06b` | `1038` | `+0` | `0.062515` | `965` | `+0` | `0.096461` | `0.039268` | `0.919796` |
| `SFT` | `sft_06b_zh_20261015T234939_step000030` | `1051` | `+13` | `0.063298` | `974` | `+9` | `0.097361` | `0.040834` | `0.918880` |
| `DPO` | `dpo_06b_zh_20261022T133007_step000040` | `1046` | `+8` | `0.062997` | `966` | `+1` | `0.096561` | `0.040954` | `0.919439` |
| `RL` | `rl_06b_zh_20261023T042034_step000005` | `1047` | `+9` | `0.063057` | `970` | `+5` | `0.096961` | `0.041556` | `0.919279` |

## 2. Near-Field Headset vs Far-Field 8-Ch Array (`ch0`) Breakdown

| Scenario | Base Edits (CER / WER) | SFT Edits (CER / WER) | DPO Edits (CER / WER) | RL Edits (CER / WER) | DPO Δ vs Base | RL Δ vs Base |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `near_field` | `356` (`0.042881` / `0.070972`) | `356` (`0.042881` / `0.070772`) | `353` (`0.042520` / `0.069372`) | `351` (`0.042279` / `0.069372`) | `-3` | `-5` |
| `far_field` | `682` (`0.082149` / `0.121951`) | `695` (`0.083715` / `0.123950`) | `693` (`0.083474` / `0.123750`) | `696` (`0.083835` / `0.124550`) | `+11` | `+14` |

## 3. Paired Source-Cluster Bootstrap 95% CIs (`B=2000`)

| Comparison | Overall Δ CER [95% CI] | Near-Field Δ CER [95% CI] | Far-Field Δ CER [95% CI] | Overall Δ WER [95% CI] |
| --- | ---: | ---: | ---: | ---: |
| `sft_vs_base` | `+0.000783` [`-0.000355`, `+0.002039`] | `+0.000000` [`-0.001281`, `+0.001253`] | `+0.001566` [`-0.000238`, `+0.003438`] | `+0.000900` [`-0.000725`, `+0.002533`] |
| `dpo_vs_base` | `+0.000482` [`-0.000739`, `+0.001856`] | `-0.000361` [`-0.001762`, `+0.001095`] | `+0.001325` [`-0.000596`, `+0.003270`] | `+0.000100` [`-0.001837`, `+0.002059`] |
| `rl_vs_base` | `+0.000542` [`-0.000791`, `+0.001980`] | `-0.000602` [`-0.002001`, `+0.000858`] | `+0.001686` [`-0.000366`, `+0.003764`] | `+0.000500` [`-0.001490`, `+0.002484`] |
| `dpo_vs_sft` | `-0.000301` [`-0.000948`, `+0.000373`] | `-0.000361` [`-0.001351`, `+0.000622`] | `-0.000241` [`-0.001022`, `+0.000591`] | `-0.000800` [`-0.002027`, `+0.000399`] |
| `rl_vs_dpo` | `+0.000060` [`-0.000535`, `+0.000670`] | `-0.000241` [`-0.000967`, `+0.000372`] | `+0.000361` [`-0.000578`, `+0.001347`] | `+0.000400` [`-0.000643`, `+0.001587`] |
