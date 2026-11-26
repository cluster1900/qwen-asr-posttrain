#!/usr/bin/env python3
"""Local CPU rehearsal for DPO v2 per-step audio count (ga), step horizon, LR warmup, gate timing, and A100 runtime.

Calibrated directly from local Track 3B DPO Wave 1 A100 telemetry:
  - Step wall time: ~0.69s * ga + 0.37s (chosen+rejected forward/backward + 1 dropout anchor + AdamW)
  - Full 2,500-row val eval + preference scoring: ~29.5 min per eval checkpoint
  - Pre-clip median grad norm at beta=0.1: ~2.30 * sqrt(4 / ga)  (ga=4: 2.30, ga=8: 1.71, ga=16: 1.20)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.replay_dpo_gate import make_dpo_v2_gate  # noqa: E402
from train.dpo_contract import (  # noqa: E402
    judge_dpo_checkpoint,
    select_training_pairs,
    trailing_non_passing_evals,
)
from train.sft_contract import (  # noqa: E402
    SCENARIOS,
    iterate_updates,
    lr_multiplier,
)

DEFAULT_SEC_PER_PAIR = 0.69
DEFAULT_STEP_OVERHEAD_SEC = 0.37
DEFAULT_EVAL_MINUTES = 29.5
DEFAULT_BASE_GRAD_NORM_GA4_B01 = 2.30
DEFAULT_SMALL_POOL_PAIRS = 256
TRACK3B_V1_DIFF_TOKENS = 160


def choose_eval_steps(total_steps: int, target_evals: int = 5) -> int:
    """Choose a clean eval interval giving <= target_evals checkpoints without a tiny tail eval."""
    if total_steps < 1 or target_evals < 1:
        raise ValueError("total_steps and target_evals must be positive integers")
    return max(1, math.ceil(total_steps / target_evals))


def default_warmup_steps(total_steps: int, warmup_ratio: float = 0.10) -> int:
    """Compute a valid warmup horizon (~10% of total_steps, strictly < total_steps)."""
    if total_steps <= 1:
        return 0
    raw = max(1, int(round(total_steps * warmup_ratio)))
    return min(raw, total_steps - 1)


def _scaled_lr(base_lr: float, ga: int, *, ref_ga: int = 16, mode: str = "none") -> float:
    if mode == "none":
        return float(base_lr)
    if mode == "sqrt":
        return float(base_lr) * math.sqrt(ga / ref_ga)
    if mode == "linear":
        return float(base_lr) * (ga / ref_ga)
    raise ValueError(f"unsupported scale_lr_mode: {mode}")


def _simulate_earliest_patience_stop(
    eval_steps_list: list[int],
    *,
    min_stop_step: int,
    patience: int,
) -> int | None:
    """Run the production DPO v2 gate over a soft-miss trajectory to find earliest patience stop."""
    base_cers = {"clean": 0.020, "noise": 0.030, "reverb": 0.025, "far_field": 0.047, "dropout": 0.185}
    parent = {"by_scenario": {k: {"samples": 500, "micro_cer": v} for k, v in base_cers.items()}}
    base = {"by_scenario": {k: {"samples": 500, "micro_cer": v + 0.002} for k, v in base_cers.items()}}
    wobble = {
        "samples": 2500,
        "micro_cer": sum(base_cers.values()) / len(base_cers),
        "empty_rate": 0.0,
        "repeated_rate": 0.0,
        "long_rate": 0.0,
        "truncation_rate": 0.0,
        "valid_output_rate": 1.0,
        "by_scenario": {
            k: {
                "samples": 500,
                "micro_cer": v + (0.0005 if k in ("far_field", "noise", "reverb") else 0.0),
                "empty_rate": 0.0,
                "repeated_rate": 0.0,
                "long_rate": 0.0,
                "truncation_rate": 0.0,
                "valid_output_rate": 1.0,
            }
            for k, v in base_cers.items()
        },
    }
    pref = {
        "accuracy": 0.60,
        "pairs": 160,
        "ties": 0,
        "raw_policy_accuracy": 0.15,
        "length_buckets": {"equal_length": {"pairs": 160, "raw_policy_accuracy": 0.15}},
    }
    gate = make_dpo_v2_gate(min_stop_step=min_stop_step, patience=patience)
    history: list[dict[str, Any]] = []
    for step in eval_steps_list:
        dec = judge_dpo_checkpoint(
            wobble,
            parent,
            base,
            gate,
            pref,
            step=step,
            locate={"locate_gap_insertions": 0},
            prior_non_passing_evals=trailing_non_passing_evals(history, min_stop_step),
        )
        history.append({"step": step, **dec})
        if dec["risk_stop"]:
            return step
    return None


def rehearse_one(
    *,
    train_pairs: int,
    ga: int,
    max_epochs: int = 1,
    max_steps: int | None = None,
    drop_tiny_tail: bool = False,
    base_lr: float = 1e-5,
    ref_ga: int = 16,
    scale_lr_mode: str = "none",
    beta: float = 0.1,
    warmup_steps: int | None = None,
    warmup_ratio: float = 0.10,
    min_lr_ratio: float = 0.1,
    eval_steps: int | None = None,
    target_evals: int = 5,
    min_stop_step: int | None = None,
    patience: int = 2,
    avg_edits_per_pair: float = 1.35,
    small_pool_pairs: int = DEFAULT_SMALL_POOL_PAIRS,
    sec_per_pair: float = DEFAULT_SEC_PER_PAIR,
    step_overhead_sec: float = DEFAULT_STEP_OVERHEAD_SEC,
    eval_minutes: float = DEFAULT_EVAL_MINUTES,
    max_runtime_minutes: int = 360,
    shutdown_margin_minutes: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    """Rehearse one (train_pairs, ga) DPO v2 configuration using production sampler/LR/gate contracts."""
    if train_pairs < 1 or ga < 1 or max_epochs < 1:
        raise ValueError("train_pairs, ga, and max_epochs must be positive integers")

    full_steps_per_epoch = train_pairs // ga
    rem = train_pairs % ga
    if max_steps is None:
        if drop_tiny_tail and max_epochs == 1 and rem > 0 and rem < 0.5 * ga and full_steps_per_epoch >= 1:
            requested_max_steps = full_steps_per_epoch
        else:
            requested_max_steps = math.ceil(train_pairs / ga) * max_epochs
    else:
        requested_max_steps = int(max_steps)
        if requested_max_steps < 1:
            raise ValueError("max_steps must be >= 1")

    # Fast exact schedule count matching train.sft_contract.iterate_updates
    steps_per_epoch = math.ceil(train_pairs / ga)
    possible_steps = steps_per_epoch * max_epochs
    actual_steps = min(requested_max_steps, possible_steps)
    # Verify exact parity with iterate_updates on small/medium counts, or deterministic math for all counts
    if train_pairs <= 20000:
        synthetic_rows = [{"sample_id": f"p{i:06d}"} for i in range(train_pairs)]
        updates = list(
            iterate_updates(
                synthetic_rows,
                seed=seed,
                accumulation=ga,
                max_steps=requested_max_steps,
                max_epochs=max_epochs,
            )
        )
        actual_steps = len(updates)
        pairs_seen = sum(len(u["rows"]) for u in updates)
        last_step_pairs = len(updates[-1]["rows"]) if updates else 0
    else:
        full_epochs_done = actual_steps // steps_per_epoch
        rem_steps = actual_steps % steps_per_epoch
        pairs_seen = full_epochs_done * train_pairs
        if rem_steps == 0:
            last_step_pairs = rem if rem > 0 else ga
        else:
            pairs_seen += min(rem_steps * ga, train_pairs)
            last_step_pairs = (
                (train_pairs - (rem_steps - 1) * ga) if rem_steps == steps_per_epoch and rem > 0 else ga
            )

    tail_pairs = 0 if last_step_pairs == ga else last_step_pairs
    resolved_warmup = (
        default_warmup_steps(actual_steps, warmup_ratio)
        if warmup_steps is None
        else min(int(warmup_steps), max(0, actual_steps - 1))
    )
    resolved_eval_steps = (
        choose_eval_steps(actual_steps, target_evals) if eval_steps is None else max(1, int(eval_steps))
    )
    eval_step_list = [s for s in range(1, actual_steps + 1) if s % resolved_eval_steps == 0 or s == actual_steps]
    resolved_min_stop_step = (
        max(resolved_warmup, eval_step_list[0])
        if min_stop_step is None and eval_step_list
        else int(min_stop_step or 0)
    )
    keep_last_required = max(len(eval_step_list), requested_max_steps // resolved_eval_steps)

    eff_lr = _scaled_lr(base_lr, ga, ref_ga=ref_ga, mode=scale_lr_mode)
    eval_lrs = {
        step: eff_lr * lr_multiplier(step, resolved_warmup, max(requested_max_steps, actual_steps), min_lr_ratio)
        for step in eval_step_list
    }
    earliest_patience_stop = _simulate_earliest_patience_stop(
        eval_step_list,
        min_stop_step=resolved_min_stop_step,
        patience=patience,
    )

    est_grad_norm = DEFAULT_BASE_GRAD_NORM_GA4_B01 * (float(beta) / 0.1) * math.sqrt(4.0 / float(ga))
    noise_reduction_vs_ga4 = math.sqrt(float(ga) / 4.0)
    est_diff_tokens = int(round(pairs_seen * avg_edits_per_pair))
    signal_multiple_vs_v1 = est_diff_tokens / TRACK3B_V1_DIFF_TOKENS

    step_sec = sec_per_pair * ga + step_overhead_sec
    train_minutes = (actual_steps * step_sec) / 60.0
    eval_minutes_total = len(eval_step_list) * eval_minutes
    total_minutes = train_minutes + eval_minutes_total
    usable_minutes = float(max_runtime_minutes - shutdown_margin_minutes)
    eval_overhead_pct = (100.0 * eval_minutes_total / total_minutes) if total_minutes > 0 else 0.0
    fits_single_session = total_minutes <= usable_minutes

    small_updates_per_epoch = math.ceil(small_pool_pairs / ga) if small_pool_pairs > 0 else 0
    small_max_epochs = math.ceil(actual_steps / small_updates_per_epoch) if small_updates_per_epoch > 0 else 0
    small_tail_pairs = (small_pool_pairs % ga) if small_pool_pairs > 0 else 0

    reject_risks: list[str] = []
    warn_risks: list[str] = []
    if actual_steps < requested_max_steps:
        reject_risks.append(f"POOL_SHORTER_THAN_MAX_STEPS({actual_steps}<{requested_max_steps})")
    if actual_steps < 200:
        reject_risks.append(f"TOO_FEW_STEPS({actual_steps}<200)")
    elif actual_steps < 300:
        warn_risks.append(f"LOW_STEPS({actual_steps}<300)")
    if est_grad_norm > 1.8:
        warn_risks.append(f"HEAVY_GRAD_CLIP(est_gn={est_grad_norm:.2f}>1.80)")
    if 0 < tail_pairs < 0.5 * ga:
        warn_risks.append(f"TINY_TAIL_STEP({tail_pairs}/{ga})")
    if not fits_single_session:
        reject_risks.append(f"RUNTIME_OVERFLOW({total_minutes:.0f}m>{usable_minutes:.0f}m)")
    if eval_overhead_pct > 70.0:
        warn_risks.append(f"HIGH_EVAL_OVERHEAD({eval_overhead_pct:.0f}%)")
    if earliest_patience_stop is not None and earliest_patience_stop <= resolved_warmup:
        reject_risks.append(f"EARLY_STOP_BEFORE_WARMUP(stop={earliest_patience_stop}<=w={resolved_warmup})")

    verdict = "REJECT" if reject_risks else ("WARN" if warn_risks else "OK")
    return {
        "train_pairs": train_pairs,
        "ga": ga,
        "max_epochs": max_epochs,
        "requested_max_steps": requested_max_steps,
        "actual_steps": actual_steps,
        "pairs_seen": pairs_seen,
        "epoch_coverage": pairs_seen / train_pairs,
        "tail_pairs": tail_pairs,
        "warmup_steps": resolved_warmup,
        "eval_steps": resolved_eval_steps,
        "eval_count": len(eval_step_list),
        "eval_step_list": eval_step_list,
        "keep_last_required": keep_last_required,
        "min_stop_step": resolved_min_stop_step,
        "earliest_patience_stop_step": earliest_patience_stop,
        "effective_lr": eff_lr,
        "eval_lrs": eval_lrs,
        "est_grad_norm": est_grad_norm,
        "noise_reduction_vs_ga4": noise_reduction_vs_ga4,
        "est_diff_tokens": est_diff_tokens,
        "signal_multiple_vs_v1": signal_multiple_vs_v1,
        "step_sec": step_sec,
        "train_minutes": train_minutes,
        "eval_minutes": eval_minutes_total,
        "total_minutes": total_minutes,
        "usable_minutes": usable_minutes,
        "eval_overhead_pct": eval_overhead_pct,
        "fits_single_session": fits_single_session,
        "small_pool_pairs": small_pool_pairs,
        "small_updates_per_epoch": small_updates_per_epoch,
        "small_max_epochs": small_max_epochs,
        "small_tail_pairs": small_tail_pairs,
        "reject_risks": reject_risks,
        "warn_risks": warn_risks,
        "verdict": verdict,
    }


def rehearse_sweep(
    *,
    train_pairs: int,
    ga_list: list[int],
    **kwargs: Any,
) -> dict[str, Any]:
    """Rehearse a list of per-step audio counts (ga) and highlight the recommended sweet spot."""
    rows = [rehearse_one(train_pairs=train_pairs, ga=ga, **kwargs) for ga in ga_list]
    ok_rows = [r for r in rows if r["verdict"] == "OK"]
    candidates = ok_rows if ok_rows else [r for r in rows if r["verdict"] == "WARN"]
    recommended_ga: int | None = None
    if candidates:
        # Prefer steps in [400, 800], lower grad clipping, and clean small-pool division
        best = min(
            candidates,
            key=lambda r: (
                0 if 400 <= r["actual_steps"] <= 800 else 1,
                abs(r["actual_steps"] - 500) / 500.0 + max(0.0, r["est_grad_norm"] - 1.0),
                0 if r["small_tail_pairs"] == 0 else 1,
            ),
        )
        recommended_ga = int(best["ga"])
        best["verdict"] = "RECOMMENDED"
    return {
        "schema_version": "qwen3-asr-dpo-schedule-rehearsal-v1",
        "train_pairs": train_pairs,
        "recommended_ga": recommended_ga,
        "rows": rows,
    }


def format_rehearsal_table(report: dict[str, Any]) -> str:
    """Render a compact, readable CLI table comparing per-step audio counts (ga)."""
    lines = [
        f"=== DPO v2 每步音频数 (ga) × 步数/耗时/梯度本地预演 (train_pairs={report['train_pairs']}) ===",
        (
            f"{'ga':>3s} | {'steps':>5s} | {'seen':>6s} | {'tail':>4s} | {'warm':>4s} | "
            f"{'eval(n)':>9s} | {'minStop':>7s} | {'patStop':>7s} | {'estGN':>5s} | "
            f"{'diffTok':>7s} | {'tr/ev/tot(m)':>15s} | {'verdict':<11s} | risks"
        ),
        "-" * 126,
    ]
    for r in report["rows"]:
        risks = ", ".join(r["reject_risks"] + r["warn_risks"]) or "-"
        if r["small_tail_pairs"] != 0:
            risks = f"{risks}; small256_tail={r['small_tail_pairs']}" if risks != "-" else f"small256_tail={r['small_tail_pairs']}"
        pat = str(r["earliest_patience_stop_step"]) if r["earliest_patience_stop_step"] is not None else "none"
        timing = f"{r['train_minutes']:4.0f}/{r['eval_minutes']:4.0f}/{r['total_minutes']:4.0f}"
        eval_col = f"{r['eval_steps']:3d}({r['eval_count']}x)"
        lines.append(
            f"{r['ga']:>3d} | {r['actual_steps']:>5d} | {r['pairs_seen']:>6d} | {r['tail_pairs']:>4d} | "
            f"{r['warmup_steps']:>4d} | {eval_col:>9s} | {r['min_stop_step']:>7d} | {pat:>7s} | "
            f"{r['est_grad_norm']:>5.2f} | {r['est_diff_tokens']:>7d} | {timing:>15s} | "
            f"{r['verdict']:<11s} | {risks}"
        )
    if report["recommended_ga"] is not None:
        rec = next(r for r in report["rows"] if r["ga"] == report["recommended_ga"])
        lines.append("")
        lines.append(
            f"推荐档位: ga={rec['ga']} -> {rec['actual_steps']} 步 (seen={rec['pairs_seen']} 对, "
            f"~{rec['est_diff_tokens']} 有效差异 token = Track 3B v1 的 {rec['signal_multiple_vs_v1']:.1f}×, "
            f"warmup={rec['warmup_steps']}, eval_steps={rec['eval_steps']} [{rec['eval_count']}次], "
            f"min_stop_step={rec['min_stop_step']}, est_grad_norm={rec['est_grad_norm']:.2f}, "
            f"预计 A100 总耗时 {rec['total_minutes']:.0f}m / {rec['usable_minutes']:.0f}m)"
        )
    return "\n".join(lines)


def _count_pairs_from_jsonl(args: argparse.Namespace) -> int:
    rows = [json.loads(line) for line in args.pair_train.read_text(encoding="utf-8").splitlines() if line.strip()]
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()] if args.scenarios else None
    selected = select_training_pairs(
        rows,
        scenarios,
        aligned_substitution_only=args.aligned_only,
        allow_matched_punctuation=args.allow_matched_punctuation,
        max_edit_count=args.max_edit_count,
        ref_margin_min=args.ref_margin_min,
        ref_margin_max=args.ref_margin_max,
        exclude_clean_identical_rejected=args.exclude_clean_identical,
    )
    return len(selected)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DPO v2 每步音频数 (ga)、总步数、梯度裁剪与 A100 耗时本地预演工具")
    sub = parser.add_subparsers(dest="command", required=True)
    sweep = sub.add_parser("sweep", help="对比多档每步音频数 (ga) 的步数、梯度噪声、门禁时机与 A100 耗时")
    src = sweep.add_mutually_exclusive_group(required=True)
    src.add_argument("--train-pairs", type=int, help="有效训练偏好对总数 N")
    src.add_argument("--pair-train", type=Path, help="从已构建的 train_pairs.jsonl 自动按过滤条件统计有效对数")
    sweep.add_argument("--scenarios", default="noise,reverb,far_field")
    sweep.add_argument("--aligned-only", action="store_true", default=True)
    sweep.add_argument("--allow-matched-punctuation", action="store_true", default=True)
    sweep.add_argument("--exclude-clean-identical", action="store_true", default=True)
    sweep.add_argument("--max-edit-count", type=int, default=2)
    sweep.add_argument("--ref-margin-min", type=float, default=-8.0)
    sweep.add_argument("--ref-margin-max", type=float, default=0.0)
    sweep.add_argument("--ga-list", default="4,8,10,12,16,20,24,32,64", help="逗号分隔的每步音频/对数候选列表")
    sweep.add_argument("--max-epochs", type=int, default=1)
    sweep.add_argument("--max-steps", type=int, default=None, help="可选固定步数；缺省按 1 epoch 自动计算")
    sweep.add_argument("--drop-tiny-tail", action="store_true", help="1 epoch 尾步 < 0.5*ga 时自动裁掉极短尾步")
    sweep.add_argument("--base-lr", type=float, default=1e-5)
    sweep.add_argument("--ref-ga", type=int, default=16)
    sweep.add_argument("--scale-lr-mode", choices=("none", "sqrt", "linear"), default="none")
    sweep.add_argument("--beta", type=float, default=0.1)
    sweep.add_argument("--warmup-ratio", type=float, default=0.10)
    sweep.add_argument("--target-evals", type=int, default=5)
    sweep.add_argument("--eval-steps", type=int, default=None)
    sweep.add_argument("--min-stop-step", type=int, default=None)
    sweep.add_argument("--patience", type=int, default=2)
    sweep.add_argument("--avg-edits-per-pair", type=float, default=1.35)
    sweep.add_argument("--small-pool-pairs", type=int, default=DEFAULT_SMALL_POOL_PAIRS)
    sweep.add_argument("--sec-per-pair", type=float, default=DEFAULT_SEC_PER_PAIR)
    sweep.add_argument("--step-overhead-sec", type=float, default=DEFAULT_STEP_OVERHEAD_SEC)
    sweep.add_argument("--eval-minutes", type=float, default=DEFAULT_EVAL_MINUTES)
    sweep.add_argument("--max-runtime-minutes", type=int, default=360)
    sweep.add_argument("--shutdown-margin-minutes", type=int, default=10)
    sweep.add_argument("--output-json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        train_pairs = _count_pairs_from_jsonl(args) if args.pair_train is not None else int(args.train_pairs)
        ga_list = [int(item.strip()) for item in args.ga_list.split(",") if item.strip()]
        if not ga_list or any(g < 1 for g in ga_list):
            raise ValueError("--ga-list must contain positive integers")
        report = rehearse_sweep(
            train_pairs=train_pairs,
            ga_list=ga_list,
            max_epochs=args.max_epochs,
            max_steps=args.max_steps,
            drop_tiny_tail=args.drop_tiny_tail,
            base_lr=args.base_lr,
            ref_ga=args.ref_ga,
            scale_lr_mode=args.scale_lr_mode,
            beta=args.beta,
            warmup_ratio=args.warmup_ratio,
            eval_steps=args.eval_steps,
            target_evals=args.target_evals,
            min_stop_step=args.min_stop_step,
            patience=args.patience,
            avg_edits_per_pair=args.avg_edits_per_pair,
            small_pool_pairs=args.small_pool_pairs,
            sec_per_pair=args.sec_per_pair,
            step_overhead_sec=args.step_overhead_sec,
            eval_minutes=args.eval_minutes,
            max_runtime_minutes=args.max_runtime_minutes,
            shutdown_margin_minutes=args.shutdown_margin_minutes,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"DPO_REHEARSE_FAIL {exc}", file=sys.stderr)
        return 2
    print(format_rehearsal_table(report))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
