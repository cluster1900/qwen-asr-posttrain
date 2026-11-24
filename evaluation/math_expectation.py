#!/usr/bin/env python3
"""Paired source-cluster Bootstrap mathematical expectation & statistical gate evaluator.

Computes exact expected deltas E[ΔCER], E[ΔWER], E[ΔReward], 95% confidence intervals,
and posterior win probabilities P(ΔCER < 0) / P(ΔReward > 0) across independent
`source_utterance_id` clusters for any stage comparison (val, sealed test, or OOD).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.jsonl_io import read_jsonl  # noqa: E402

MATH_EXPECTATION_SCHEMA = "qwen3-asr-math-expectation-v1"
DEGRADED_4 = ("noise", "reverb", "far_field", "dropout")
PHYSICAL_DEGRADED_3 = ("noise", "reverb", "far_field")


def sample_reward_from_scored_row(
    row: dict[str, Any],
    *,
    empty_penalty: float = 0.5,
    repeated_penalty: float = 0.5,
    clip_low: float = -1.0,
    clip_high: float = 1.0,
) -> float:
    """Compute clipped per-sample reward consistent with 04_training_plan.md."""
    cer = float(row["sample_cer"])
    empty = 1.0 if bool(row.get("empty_output", False)) else 0.0
    rep = 1.0 if bool(row.get("repeated_output", False)) else 0.0
    raw = 1.0 - cer - empty_penalty * empty - repeated_penalty * rep
    return max(clip_low, min(clip_high, raw))


def _slice_defs_for_scenarios(present_scenarios: set[str]) -> list[tuple[str, set[str]]]:
    """Define evaluation slices automatically for 5-scenario AISHELL or 2-scenario AliMeeting OOD."""
    slices: list[tuple[str, set[str]]] = [("overall", set(present_scenarios))]
    if set(DEGRADED_4).issubset(present_scenarios):
        slices.append(("degraded_4", set(DEGRADED_4)))
    if set(PHYSICAL_DEGRADED_3).issubset(present_scenarios):
        slices.append(("physical_degraded_3", set(PHYSICAL_DEGRADED_3)))
    for sc in sorted(present_scenarios):
        slices.append((sc, {sc}))
    return slices


def evaluate_math_expectation(
    manifest_rows: list[dict[str, Any]],
    ref_scored: list[dict[str, Any]],
    cand_scored: list[dict[str, Any]],
    *,
    num_resamples: int = 2000,
    seed: int = 42,
    target_slice: str | None = None,
    clean_slice: str | None = None,
    min_win_prob: float = 0.90,
    clean_cer_increase_max: float = 0.01,
    require_ci95_negative: bool = False,
) -> dict[str, Any]:
    """Evaluate paired source-cluster bootstrap expectation of (cand - ref)."""
    if num_resamples < 100:
        raise ValueError("num_resamples must be >= 100")
    if not manifest_rows or len(ref_scored) != len(manifest_rows) or len(cand_scored) != len(manifest_rows):
        raise ValueError(
            f"row count mismatch: manifest={len(manifest_rows)}, ref={len(ref_scored)}, cand={len(cand_scored)}"
        )

    manifest_by_id = {str(r["sample_id"]): r for r in manifest_rows}
    ref_by_id = {str(r["sample_id"]): r for r in ref_scored}
    cand_by_id = {str(r["sample_id"]): r for r in cand_scored}
    if set(manifest_by_id) != set(ref_by_id) or set(manifest_by_id) != set(cand_by_id):
        raise ValueError("sample_id sets differ between manifest, ref_scored, and cand_scored")

    present_scenarios = {str(r["scenario"]) for r in manifest_rows}
    slices = _slice_defs_for_scenarios(present_scenarios)
    has_wer = all(
        "word_edit_count" in ref_by_id[sid]
        and "reference_word_count" in ref_by_id[sid]
        and "word_edit_count" in cand_by_id[sid]
        and "reference_word_count" in cand_by_id[sid]
        for sid in manifest_by_id
    )

    sources = sorted({str(r["source_utterance_id"]) for r in manifest_rows})
    n_src = len(sources)
    src_idx = {s: i for i, s in enumerate(sources)}
    n_slices = len(slices)

    cer_ed_ref = np.zeros((n_src, n_slices), dtype=np.float64)
    cer_ed_cand = np.zeros((n_src, n_slices), dtype=np.float64)
    cer_ref_chars = np.zeros((n_src, n_slices), dtype=np.float64)
    rew_ref_sum = np.zeros((n_src, n_slices), dtype=np.float64)
    rew_cand_sum = np.zeros((n_src, n_slices), dtype=np.float64)
    row_counts = np.zeros((n_src, n_slices), dtype=np.float64)
    wer_ed_ref = np.zeros((n_src, n_slices), dtype=np.float64)
    wer_ed_cand = np.zeros((n_src, n_slices), dtype=np.float64)
    wer_ref_words = np.zeros((n_src, n_slices), dtype=np.float64)

    for sid, mrow in manifest_by_id.items():
        idx = src_idx[str(mrow["source_utterance_id"])]
        sc = str(mrow["scenario"])
        ra = ref_by_id[sid]
        rb = cand_by_id[sid]
        rew_a = sample_reward_from_scored_row(ra)
        rew_b = sample_reward_from_scored_row(rb)
        for col, (_sname, sc_set) in enumerate(slices):
            if sc not in sc_set:
                continue
            cer_ed_ref[idx, col] += float(ra["edit_count"])
            cer_ed_cand[idx, col] += float(rb["edit_count"])
            cer_ref_chars[idx, col] += float(ra["reference_char_count"])
            rew_ref_sum[idx, col] += rew_a
            rew_cand_sum[idx, col] += rew_b
            row_counts[idx, col] += 1.0
            if has_wer:
                wer_ed_ref[idx, col] += float(ra["word_edit_count"])
                wer_ed_cand[idx, col] += float(rb["word_edit_count"])
                wer_ref_words[idx, col] += float(ra["reference_word_count"])

    rng = np.random.default_rng(int(seed))
    boot = rng.integers(0, n_src, size=(int(num_resamples), n_src))

    slice_results: dict[str, Any] = {}
    for col, (sname, _sc_set) in enumerate(slices):
        ref_chars_tot = float(cer_ref_chars[:, col].sum())
        point_cer_ref = float(cer_ed_ref[:, col].sum() / max(ref_chars_tot, 1.0))
        point_cer_cand = float(cer_ed_cand[:, col].sum() / max(ref_chars_tot, 1.0))
        point_cer_delta = point_cer_cand - point_cer_ref
        edit_delta = int(round(float(cer_ed_cand[:, col].sum() - cer_ed_ref[:, col].sum())))

        b_chars = np.maximum(cer_ref_chars[:, col][boot].sum(axis=1), 1.0)
        b_cer_delta = (cer_ed_cand[:, col][boot].sum(axis=1) - cer_ed_ref[:, col][boot].sum(axis=1)) / b_chars

        cnt_tot = float(row_counts[:, col].sum())
        point_rew_delta = float((rew_cand_sum[:, col].sum() - rew_ref_sum[:, col].sum()) / max(cnt_tot, 1.0))
        b_cnt = np.maximum(row_counts[:, col][boot].sum(axis=1), 1.0)
        b_rew_delta = (rew_cand_sum[:, col][boot].sum(axis=1) - rew_ref_sum[:, col][boot].sum(axis=1)) / b_cnt

        entry: dict[str, Any] = {
            "samples": int(round(cnt_tot)),
            "reference_char_count": int(round(ref_chars_tot)),
            "ref_edit_count": int(round(float(cer_ed_ref[:, col].sum()))),
            "cand_edit_count": int(round(float(cer_ed_cand[:, col].sum()))),
            "edit_delta": edit_delta,
            "ref_micro_cer": point_cer_ref,
            "cand_micro_cer": point_cer_cand,
            "point_cer_delta": point_cer_delta,
            "expected_cer_delta": float(np.mean(b_cer_delta)),
            "cer_delta_se": float(np.std(b_cer_delta, ddof=1)),
            "cer_delta_ci95": [float(np.percentile(b_cer_delta, 2.5)), float(np.percentile(b_cer_delta, 97.5))],
            "cer_win_prob": float(np.mean(b_cer_delta < 0.0)),
            "cer_one_sided_p": float(np.mean(b_cer_delta >= 0.0)),
            "point_reward_delta": point_rew_delta,
            "expected_reward_delta": float(np.mean(b_rew_delta)),
            "reward_delta_ci95": [float(np.percentile(b_rew_delta, 2.5)), float(np.percentile(b_rew_delta, 97.5))],
            "reward_win_prob": float(np.mean(b_rew_delta > 0.0)),
        }
        if has_wer:
            ref_words_tot = float(wer_ref_words[:, col].sum())
            point_wer_ref = float(wer_ed_ref[:, col].sum() / max(ref_words_tot, 1.0))
            point_wer_cand = float(wer_ed_cand[:, col].sum() / max(ref_words_tot, 1.0))
            b_words = np.maximum(wer_ref_words[:, col][boot].sum(axis=1), 1.0)
            b_wer_delta = (wer_ed_cand[:, col][boot].sum(axis=1) - wer_ed_ref[:, col][boot].sum(axis=1)) / b_words
            entry.update(
                {
                    "word_edit_delta": int(round(float(wer_ed_cand[:, col].sum() - wer_ed_ref[:, col].sum()))),
                    "ref_micro_wer": point_wer_ref,
                    "cand_micro_wer": point_wer_cand,
                    "point_wer_delta": point_wer_cand - point_wer_ref,
                    "expected_wer_delta": float(np.mean(b_wer_delta)),
                    "wer_delta_ci95": [float(np.percentile(b_wer_delta, 2.5)), float(np.percentile(b_wer_delta, 97.5))],
                    "wer_win_prob": float(np.mean(b_wer_delta < 0.0)),
                }
            )
        slice_results[sname] = entry

    # Resolve target and clean slices automatically if omitted
    if target_slice is None:
        if "physical_degraded_3" in slice_results:
            target_slice = "physical_degraded_3"
        elif "far_field" in slice_results:
            target_slice = "far_field"
        else:
            target_slice = "overall"
    if clean_slice is None:
        if "clean" in slice_results:
            clean_slice = "clean"
        elif "near_field" in slice_results:
            clean_slice = "near_field"

    target_block = slice_results[target_slice]
    expected_improvement = bool(
        target_block["expected_cer_delta"] < 0.0 and target_block["expected_reward_delta"] > 0.0
    )
    strong_positive = bool(target_block["cer_win_prob"] >= float(min_win_prob))
    significant_95 = bool(target_block["cer_delta_ci95"][1] < 0.0)
    clean_safe = True
    if clean_slice is not None and clean_slice in slice_results:
        clean_block = slice_results[clean_slice]
        clean_safe = bool(clean_block["cer_delta_ci95"][1] <= float(clean_cer_increase_max))

    passed = expected_improvement and strong_positive and clean_safe and (
        significant_95 if require_ci95_negative else True
    )
    verdict = {
        "target_slice": target_slice,
        "clean_slice": clean_slice,
        "min_win_prob": float(min_win_prob),
        "clean_cer_increase_max": float(clean_cer_increase_max),
        "require_ci95_negative": bool(require_ci95_negative),
        "expected_improvement": expected_improvement,
        "strong_positive_expectation": strong_positive,
        "statistically_significant_95": significant_95,
        "clean_safe": clean_safe,
        "passed_expectation": passed,
    }
    return {
        "schema_version": MATH_EXPECTATION_SCHEMA,
        "seed": int(seed),
        "resamples": int(num_resamples),
        "clusters": n_src,
        "samples": len(manifest_rows),
        "has_wer": has_wer,
        "slices": slice_results,
        "verdict": verdict,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="配对源簇 Bootstrap 数学期望与统计门禁评估")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--ref-scored", required=True, type=Path)
    parser.add_argument("--cand-scored", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-slice", default=None)
    parser.add_argument("--clean-slice", default=None)
    parser.add_argument("--min-win-prob", type=float, default=0.90)
    parser.add_argument("--clean-cer-increase-max", type=float, default=0.01)
    parser.add_argument("--require-ci95-negative", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest_rows = read_jsonl(args.manifest)
        ref_rows = read_jsonl(args.ref_scored)
        cand_rows = read_jsonl(args.cand_scored)
        report = evaluate_math_expectation(
            manifest_rows,
            ref_rows,
            cand_rows,
            num_resamples=args.resamples,
            seed=args.seed,
            target_slice=args.target_slice,
            clean_slice=args.clean_slice,
            min_win_prob=args.min_win_prob,
            clean_cer_increase_max=args.clean_cer_increase_max,
            require_ci95_negative=args.require_ci95_negative,
        )
    except (OSError, ValueError) as exc:
        print(f"MATH_EXPECTATION_ERROR: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    t = report["verdict"]["target_slice"]
    tb = report["slices"][t]
    print(
        json.dumps(
            {
                "target_slice": t,
                "edit_delta": tb["edit_delta"],
                "expected_cer_delta": tb["expected_cer_delta"],
                "cer_delta_ci95": tb["cer_delta_ci95"],
                "cer_win_prob": tb["cer_win_prob"],
                "passed_expectation": report["verdict"]["passed_expectation"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
