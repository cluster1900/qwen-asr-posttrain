#!/usr/bin/env python3
"""Offline local DPO v2 gate replay, synthetic boundary stress suite, and LR warmup verifier.

Runs completely on local CPU using the production gate contracts
(`train.dpo_contract.judge_dpo_checkpoint`, `train.sft_contract.select_sft_checkpoint`,
`train.sft_contract.lr_multiplier`) with zero model downloads or GPU requirements.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.dpo_contract import (  # noqa: E402
    judge_dpo_checkpoint,
    locate_dropout_edits,
    trailing_non_passing_evals,
)
from train.sft_contract import (  # noqa: E402
    SCENARIOS,
    lr_multiplier,
    select_sft_checkpoint,
)

DEFAULT_WAVE1_DIR = ROOT / "results/phase6_track3_slr28_20261124/dpo_wave1"
DEFAULT_TRACK3_PARENT = ROOT / "configs/train/aishell_slr28_val_sft_w1s240_20261124.json"
DEFAULT_TRACK3_BASE = ROOT / "configs/train/aishell_slr28_val_base_20261124.json"
DEFAULT_PHASE3_RUN = ROOT / "results/phase3_dpo_gap_strip_probe_20261018/runs/20261018T132840_dpo_06b_zh_seed42"
DEFAULT_PHASE3_PARENT = ROOT / "configs/train/aishell_val_sft_step30_20261016.json"
DEFAULT_PHASE3_BASE = ROOT / "configs/train/aishell_val_base_20261015.json"

BASE_LEGACY_GATE: dict[str, Any] = {
    "min_rows_per_scenario": 50,
    "clean_cer_increase_max": 0.01,
    "clean_cer_increase_vs_base_max": 0.02,
    "degraded_macro_increase_max": 0.0,
    "scenario_cer_increase_max": 0.01,
    "min_improved_degraded_scenarios": 1,
    "empty_rate_max": 0.01,
    "repeated_rate_max": 0.01,
    "long_rate_max": 0.01,
    "truncation_rate_max": 0.01,
    "valid_output_rate_min": 0.99,
    "min_preference_accuracy": 0.55,
    "min_val_pairs": 100,
    "min_reward_improvement": 0.002,
    "raw_kl_max": 0.001,
    "reward_drop_stop": 0.01,
}


def make_dpo_v2_gate(
    *,
    macro_scenarios: list[str] | None = None,
    macro_stop_max: float = 0.002,
    dropout_cap: float = 0.002,
    max_gap_insertions: int = 5,
    stop_on_low_preference: bool = False,
    min_stop_step: int = 40,
    patience: int = 2,
) -> dict[str, Any]:
    """Construct the calibrated DPO v2 decoupled anti-lockup gate dictionary."""
    gate = dict(BASE_LEGACY_GATE)
    gate["degraded_macro_scenarios"] = list(macro_scenarios or ["far_field", "noise", "reverb"])
    gate["degraded_macro_stop_increase_max"] = float(macro_stop_max)
    gate["dropout_cer_increase_max"] = float(dropout_cap)
    gate["max_gap_insertions"] = int(max_gap_insertions)
    if not stop_on_low_preference:
        gate["stop_on_low_preference"] = False
    if min_stop_step > 0:
        gate["min_stop_step"] = int(min_stop_step)
    gate["max_consecutive_non_passing_evals"] = int(patience)
    return gate


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def metrics_from_scored_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate ``scored.jsonl`` rows into the exact ``metrics.json`` structure consumed by gate contracts."""
    flags = {"empty_rate": "empty_output", "repeated_rate": "repeated_output", "long_rate": "long_output",
             "truncation_rate": "truncated", "valid_output_rate": "valid_output"}
    for row in rows:
        if row.get("scenario") not in SCENARIOS or any(type(row.get(flag)) is not bool for flag in flags.values()):
            raise ValueError(f"incomplete scenario or output health flags for {row.get('sample_id')}")

    def _agg(subset: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(subset)
        edits = sum(int(r["edit_count"]) for r in subset)
        chars = sum(int(r["reference_char_count"]) for r in subset)
        return {
            "samples": total,
            "edit_count": edits,
            "reference_char_count": chars,
            "micro_cer": (edits / chars) if chars else 0.0,
            **{key: sum(r[flag] for r in subset) / total if total else 0.0 for key, flag in flags.items()},
        }

    by_scenario: dict[str, Any] = {}
    for scenario in SCENARIOS:
        by_scenario[scenario] = _agg([r for r in rows if r.get("scenario") == scenario])
    overall = _agg(rows)
    overall["by_scenario"] = by_scenario
    return overall


def _load_step_inputs(
    run_dir: Path,
    recorded_decision: dict[str, Any],
    parent_metrics: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    step = int(recorded_decision["step"])
    step_dir = run_dir / "eval" / f"step_{step:06d}"
    metrics_path = step_dir / "metrics.json"
    scored_path = step_dir / "scored.jsonl"
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    elif scored_path.is_file():
        metrics = metrics_from_scored_rows(_read_jsonl(scored_path))
    else:
        raise FileNotFoundError(f"neither metrics.json nor scored.jsonl found in {step_dir}")
    if any(metrics["by_scenario"][name]["samples"] != parent_metrics["by_scenario"][name]["samples"]
           for name in SCENARIOS):
        raise ValueError(f"{step_dir}: incomplete validation metrics")

    pref_path = step_dir / "preference.json"
    if pref_path.is_file():
        preference = json.loads(pref_path.read_text(encoding="utf-8"))
    else:
        preference = {
            "accuracy": float(recorded_decision["preference_accuracy"]),
            "ties": int(recorded_decision["preference_ties"]),
            "raw_policy_accuracy": float(recorded_decision["raw_policy_accuracy"]),
        }
    buckets = preference.get("length_buckets", recorded_decision.get("length_buckets"))
    if not isinstance(buckets, dict) or not buckets or any(
        not isinstance(b, dict) or type(b.get("pairs")) is not int or b["pairs"] < 0 for b in buckets.values()
    ):
        raise ValueError(f"{step_dir}: missing or invalid preference pair counts")
    total_pairs = sum(b["pairs"] for b in buckets.values())
    if "pairs" in preference and (type(preference["pairs"]) is not int or preference["pairs"] != total_pairs):
        raise ValueError(f"{step_dir}: preference count differs from length buckets")
    preference.update(pairs=total_pairs, length_buckets=buckets)
    count = recorded_decision.get("locate_gap_insertions")
    if type(count) is int and count >= 0:
        locate = {"locate_gap_insertions": count}
    elif scored_path.is_file():
        scored = _read_jsonl(scored_path)
        dropout_ids = [r["sample_id"] for r in scored if r["scenario"] == "dropout"]
        if len(set(dropout_ids)) != len(dropout_ids) or len(dropout_ids) != metrics["by_scenario"]["dropout"]["samples"]:
            raise ValueError(f"{step_dir}: incomplete dropout scored rows")
        locate = locate_dropout_edits(scored, metrics, parent_metrics)
    else:
        raise ValueError(f"{step_dir}: missing dropout gap count and scored.jsonl")
    return metrics, preference, locate


def replay_run_directory(
    run_dir: Path,
    parent_metrics: dict[str, Any],
    base_metrics: dict[str, Any],
    legacy_gate: dict[str, Any],
    v2_gate: dict[str, Any],
) -> dict[str, Any]:
    """Replay every recorded eval step in ``run_dir`` under both Legacy and DPO v2 gates."""
    decision_path = run_dir / "eval_decisions.jsonl"
    recorded = _read_jsonl(decision_path)
    legacy_history: list[dict[str, Any]] = []
    v2_history: list[dict[str, Any]] = []
    step_reports: list[dict[str, Any]] = []
    legacy_parity_ok = True

    steps = [r["step"] for r in recorded]
    if not steps or any(type(s) is not int or s <= 0 for s in steps) or steps != sorted(set(steps)):
        raise ValueError(f"{decision_path}: empty, duplicate or out-of-order eval steps")
    for rec in recorded:
        step = int(rec["step"])
        metrics, preference, locate = _load_step_inputs(run_dir, rec, parent_metrics)
        leg_dec = judge_dpo_checkpoint(
            metrics,
            parent_metrics,
            base_metrics,
            legacy_gate,
            preference,
            step=step,
            locate=locate,
            prior_non_passing_evals=trailing_non_passing_evals(legacy_history),
        )
        leg_dec["step"] = step
        legacy_reachable = not legacy_history or not legacy_history[-1]["risk_stop"]
        if legacy_reachable:
            legacy_history.append(leg_dec)

        v2_dec = judge_dpo_checkpoint(
            metrics,
            parent_metrics,
            base_metrics,
            v2_gate,
            preference,
            step=step,
            locate=locate,
            prior_non_passing_evals=trailing_non_passing_evals(v2_history, v2_gate.get("min_stop_step", 0)),
        )
        v2_dec["step"] = step
        v2_reachable = not v2_history or not v2_history[-1]["risk_stop"]
        if v2_reachable:
            v2_history.append(v2_dec)

        parity = (
            bool(leg_dec["passed"]) == bool(rec["passed"])
            and bool(leg_dec["risk_stop"]) == bool(rec["risk_stop"])
        )
        if not parity:
            legacy_parity_ok = False

        step_reports.append(
            {
                "step": step,
                "legacy_parity": parity,
                "legacy_reachable": legacy_reachable,
                "v2_reachable": v2_reachable,
                "recorded_passed": bool(rec["passed"]),
                "recorded_risk_stop": bool(rec["risk_stop"]),
                "legacy_passed": bool(leg_dec["passed"]),
                "legacy_risk_stop": bool(leg_dec["risk_stop"]),
                "legacy_macro_delta": float(leg_dec["macro_delta"]),
                "v2_passed": bool(v2_dec["passed"]),
                "v2_risk_stop": bool(v2_dec["risk_stop"]),
                "v2_macro_delta": float(v2_dec["macro_delta"]),
                "v2_hard_risks": list(v2_dec["hard_risks"]),
                "v2_soft_risks": list(v2_dec["soft_risks"]),
                "v2_consecutive_non_passing": int(v2_dec["consecutive_non_passing_evals"]),
                "dropout_delta": float(v2_dec["degraded_deltas"]["dropout"]),
                "clean_delta": float(v2_dec["clean_delta_vs_parent"]),
                "gap_insertions": int(locate["locate_gap_insertions"]),
                "preference_accuracy": float(v2_dec["preference_accuracy"]),
            }
        )

    legacy_sel = select_sft_checkpoint(legacy_history)
    v2_sel = select_sft_checkpoint(v2_history)
    return {
        "run_name": run_dir.name,
        "steps": step_reports,
        "legacy_parity_ok": legacy_parity_ok,
        "legacy_selected_step": None if legacy_sel is None else int(legacy_sel["step"]),
        "v2_selected_step": None if v2_sel is None else int(v2_sel["step"]),
    }


def _synthetic_metrics(
    base_cers: dict[str, float],
    deltas: dict[str, float],
    *,
    empty_rate: float = 0.0,
    repeated_rate: float = 0.0,
) -> dict[str, Any]:
    by_scenario: dict[str, Any] = {}
    for name in SCENARIOS:
        cer = max(0.0, base_cers[name] + deltas.get(name, 0.0))
        by_scenario[name] = {
            "samples": 500,
            "micro_cer": cer,
            "empty_rate": empty_rate if name == "noise" else 0.0,
            "repeated_rate": repeated_rate if name == "noise" else 0.0,
            "long_rate": 0.0,
            "truncation_rate": 0.0,
            "valid_output_rate": 1.0 - (empty_rate if name == "noise" else 0.0),
        }
    return {
        "samples": 2500,
        "micro_cer": sum(by_scenario[n]["micro_cer"] for n in SCENARIOS) / len(SCENARIOS),
        "empty_rate": empty_rate / len(SCENARIOS),
        "repeated_rate": repeated_rate / len(SCENARIOS),
        "long_rate": 0.0,
        "truncation_rate": 0.0,
        "valid_output_rate": 1.0 - (empty_rate / len(SCENARIOS)),
        "by_scenario": by_scenario,
    }


def _synthetic_pref(accuracy: float, pairs: int = 200) -> dict[str, Any]:
    return {
        "accuracy": accuracy,
        "pairs": pairs,
        "ties": 0,
        "raw_policy_accuracy": 0.15,
        "length_buckets": {"equal_length": {"pairs": pairs, "raw_policy_accuracy": 0.15}},
    }


def run_synthetic_stress_suite(v2_gate: dict[str, Any]) -> list[dict[str, Any]]:
    """Run 7 deterministic boundary stress tests verifying safety stops and anti-lockup resilience."""
    base_cers = {
        "clean": 0.020,
        "noise": 0.030,
        "reverb": 0.025,
        "far_field": 0.047,
        "dropout": 0.185,
    }
    parent = {"by_scenario": {k: {"samples": 500, "micro_cer": v} for k, v in base_cers.items()}}
    base = {"by_scenario": {k: {"samples": 500, "micro_cer": v + 0.002} for k, v in base_cers.items()}}
    results: list[dict[str, Any]] = []

    def _record(name: str, ok: bool, detail: str) -> None:
        results.append({"case": name, "passed": ok, "detail": detail})

    # 1. Silence hallucination inside warmup MUST hard-stop immediately
    d1 = judge_dpo_checkpoint(
        _synthetic_metrics(base_cers, {"far_field": -0.001, "dropout": 0.001}),
        parent,
        base,
        v2_gate,
        _synthetic_pref(0.65),
        step=20,
        locate={"locate_gap_insertions": 10},
    )
    _record(
        "silence_hallucination_in_warmup",
        (d1["passed"] is False and d1["risk_stop"] is True),
        f"step=20 gap_ins=10 -> passed={d1['passed']} stop={d1['risk_stop']} hard={d1['hard_risks']}",
    )

    # 2. Dropout CER spike inside warmup MUST hard-stop immediately
    d2 = judge_dpo_checkpoint(
        _synthetic_metrics(base_cers, {"far_field": -0.001, "dropout": 0.005}),
        parent,
        base,
        v2_gate,
        _synthetic_pref(0.65),
        step=20,
        locate={"locate_gap_insertions": 0},
    )
    _record(
        "dropout_cer_spike_in_warmup",
        (d2["passed"] is False and d2["risk_stop"] is True),
        f"step=20 dropout=+0.005 -> passed={d2['passed']} stop={d2['risk_stop']} hard={d2['hard_risks']}",
    )

    # 3. Clean regression MUST hard-stop immediately
    d3 = judge_dpo_checkpoint(
        _synthetic_metrics(base_cers, {"clean": 0.015, "far_field": -0.002}),
        parent,
        base,
        v2_gate,
        _synthetic_pref(0.65),
        step=20,
        locate={"locate_gap_insertions": 0},
    )
    _record(
        "clean_regression_hard_stop",
        (d3["passed"] is False and d3["risk_stop"] is True),
        f"step=20 clean=+0.015 -> passed={d3['passed']} stop={d3['risk_stop']} hard={d3['hard_risks']}",
    )

    # 4. Health empty_rate spike MUST hard-stop immediately
    d4 = judge_dpo_checkpoint(
        _synthetic_metrics(base_cers, {"far_field": -0.002}, empty_rate=0.02),
        parent,
        base,
        v2_gate,
        _synthetic_pref(0.65),
        step=20,
        locate={"locate_gap_insertions": 0},
    )
    _record(
        "health_empty_rate_hard_stop",
        (d4["passed"] is False and d4["risk_stop"] is True),
        f"step=20 empty=0.02 -> passed={d4['passed']} stop={d4['risk_stop']} hard={d4['hard_risks']}",
    )

    # 5. Warmup early low preference blocks pass but does NOT kill run
    d5 = judge_dpo_checkpoint(
        _synthetic_metrics(base_cers, {"far_field": -0.001, "noise": -0.001, "dropout": 0.0005}),
        parent,
        base,
        v2_gate,
        _synthetic_pref(0.48),
        step=20,
        locate={"locate_gap_insertions": 0},
    )
    _record(
        "warmup_low_preference_survives",
        (d5["passed"] is False and d5["risk_stop"] is False and len(d5["soft_risks"]) == 1),
        f"step=20 pref=0.48 phys3<0 -> passed={d5['passed']} stop={d5['risk_stop']} soft={d5['soft_risks']}",
    )

    # 6. Soft wobble at step 40 followed by strong recovery at step 80 -> selects step 80
    history6: list[dict[str, Any]] = []
    d6a = judge_dpo_checkpoint(
        _synthetic_metrics(base_cers, {"far_field": 0.0005, "noise": 0.0005, "reverb": 0.0005}),
        parent,
        base,
        v2_gate,
        _synthetic_pref(0.60),
        step=40,
        locate={"locate_gap_insertions": 0},
        prior_non_passing_evals=trailing_non_passing_evals(history6),
    )
    d6a["step"] = 40
    history6.append(d6a)
    d6b = judge_dpo_checkpoint(
        _synthetic_metrics(base_cers, {"far_field": -0.002, "noise": -0.001, "reverb": -0.0006, "dropout": 0.0004}),
        parent,
        base,
        v2_gate,
        _synthetic_pref(0.64),
        step=80,
        locate={"locate_gap_insertions": 0},
        prior_non_passing_evals=trailing_non_passing_evals(history6),
    )
    d6b["step"] = 80
    history6.append(d6b)
    sel6 = select_sft_checkpoint(history6)
    _record(
        "soft_wobble_then_recovery",
        (
            d6a["passed"] is False
            and d6a["risk_stop"] is False
            and d6b["passed"] is True
            and d6b["risk_stop"] is False
            and d6b["consecutive_non_passing_evals"] == 0
            and sel6 is not None
            and int(sel6["step"]) == 80
        ),
        f"step40(pass={d6a['passed']},stop={d6a['risk_stop']}) -> step80(pass={d6b['passed']},stop={d6b['risk_stop']}) sel={None if sel6 is None else sel6['step']}",
    )

    # 7. Sustained non-passing evals after min_stop_step trigger clean patience stop
    history7 = []
    for step in (40, 80, 120):
        decision = judge_dpo_checkpoint(
            _synthetic_metrics(base_cers, {"far_field": 0.0005, "noise": 0.0005, "reverb": 0.0005}),
            parent, base, v2_gate, _synthetic_pref(0.60), step=step, locate={"locate_gap_insertions": 0},
            prior_non_passing_evals=trailing_non_passing_evals(history7, v2_gate.get("min_stop_step", 0)),
        )
        history7.append(dict(step=step, **decision))
    counts = [d["consecutive_non_passing_evals"] for d in history7]
    stops = [d["risk_stop"] for d in history7]
    _record("sustained_non_passing_patience_stop", counts == [0, 1, 2] and stops == [False, False, True],
            f"steps 40/80/120 counts={counts} stops={stops}")
    return results


def verify_warmup_schedules() -> list[dict[str, Any]]:
    """Verify linear warmup + cosine decay curves for planned DPO v2 recipes."""
    recipe = yaml.safe_load((ROOT / "configs/experiments/dpo_v2_sweep.yaml").read_text())
    optim = recipe["common"]["optim"]
    reports: list[dict[str, Any]] = []
    for base_lr in recipe["learning_rates"]:
        w = optim["warmup_steps"]
        total = recipe["common"]["train"]["max_steps"]
        min_r = optim["min_lr_ratio"]
        mults = [lr_multiplier(s, w, total, min_r) for s in range(1, total + 1)]
        warmup_monotonic = all(mults[i] < mults[i + 1] for i in range(w - 1))
        peak_ok = math.isclose(mults[w - 1], 1.0, rel_tol=1e-12, abs_tol=1e-12)
        decay_monotonic = all(mults[i] >= mults[i + 1] for i in range(w - 1, total - 1))
        end_ok = math.isclose(mults[-1], min_r, rel_tol=1e-12, abs_tol=1e-12)
        probes = sorted({1, max(1, w // 2), w, min(total, 40), total // 2, total})
        samples = [
            {"step": s, "multiplier": round(mults[s - 1], 6), "lr": base_lr * mults[s - 1]}
            for s in probes
        ]
        reports.append(
            {
                "name": f"v2_lr{base_lr:.0e}", "base_lr": base_lr, "warmup_steps": w,
                "max_steps": total, "min_lr_ratio": min_r,
                "passed": warmup_monotonic and peak_ok and decay_monotonic and end_ok,
                "samples": samples,
            }
        )
    return reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="离线本地重放 DPO v2 解耦抗锁死门禁、边界压测与 LR Warmup 核查")
    parser.add_argument("--wave1-dir", type=Path, default=DEFAULT_WAVE1_DIR)
    parser.add_argument("--track3-parent", type=Path, default=DEFAULT_TRACK3_PARENT)
    parser.add_argument("--track3-base", type=Path, default=DEFAULT_TRACK3_BASE)
    parser.add_argument("--phase3-run", type=Path, default=DEFAULT_PHASE3_RUN)
    parser.add_argument("--phase3-parent", type=Path, default=DEFAULT_PHASE3_PARENT)
    parser.add_argument("--phase3-base", type=Path, default=DEFAULT_PHASE3_BASE)
    parser.add_argument("--skip-local-runs", action="store_true", help="跳过本地历史运行目录，仅跑合成压测与 Warmup 校验")
    parser.add_argument("--macro-stop-max", type=float, default=0.002)
    parser.add_argument("--dropout-cap", type=float, default=0.002)
    parser.add_argument("--max-gap-insertions", type=int, default=5)
    parser.add_argument("--min-stop-step", type=int, default=40)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    v2_gate = make_dpo_v2_gate(
        macro_stop_max=args.macro_stop_max,
        dropout_cap=args.dropout_cap,
        max_gap_insertions=args.max_gap_insertions,
        min_stop_step=args.min_stop_step,
        patience=args.patience,
    )
    report: dict[str, Any] = {
        "schema_version": "qwen3-asr-dpo-v2-gate-replay-v1",
        "v2_gate": v2_gate,
        "wave1_runs": [],
        "phase3_runs": [],
        "stress_suite": [],
        "warmup_schedules": [],
    }
    all_ok = True

    report["history_requested"] = not args.skip_local_runs
    report["history_errors"] = []
    if not args.skip_local_runs:
        expected = {"t3-w0": [20], "t3-w1": [20], "t3-w2": [20, 40], "t3-w3": [20, 40, 60],
                    "t3-w4": [16], "t3-w5": [16], "t3-w6": [18], "t3-w8": [18], "t3-w9": [30]}
        try:
            groups = (
                (args.wave1_dir, args.track3_parent, args.track3_base, expected, "wave1_runs"),
                (args.phase3_run.parent, args.phase3_parent, args.phase3_base,
                 {args.phase3_run.name: [10]}, "phase3_runs"),
            )
            for directory, parent_file, base_file, expected_steps, key in groups:
                parent = json.loads(parent_file.read_text(encoding="utf-8"))
                base = json.loads(base_file.read_text(encoding="utf-8"))
                for prefix, steps in expected_steps.items():
                    if key == "phase3_runs":
                        run = args.phase3_run
                    else:
                        matches = [p for p in directory.glob(f"{prefix}_*") if p.is_dir()]
                        if len(matches) != 1:
                            raise ValueError(f"{directory}: expected one {prefix} run, found {len(matches)}")
                        run = matches[0]
                    rep = replay_run_directory(run, parent, base, BASE_LEGACY_GATE, v2_gate)
                    report[key].append(rep)
                    if [st["step"] for st in rep["steps"]] != steps or not rep["legacy_parity_ok"]:
                        raise ValueError(f"{run}: incomplete steps or legacy decision mismatch")
                    for st in rep["steps"]:
                        print(f"{prefix} step={st['step']} reachable={st['v2_reachable']} "
                              f"passed={st['v2_passed']} risk_stop={st['v2_risk_stop']}")
                    if key == "phase3_runs":
                        incident = rep["steps"][0]
                        risks = incident["v2_hard_risks"]
                        if (incident["gap_insertions"] != 10 or not incident["v2_risk_stop"]
                                or not any("dropout" in reason for reason in risks)
                                or (args.max_gap_insertions < 10 and not any("max_gap_insertions" in r for r in risks))):
                            raise ValueError(f"{run}: missing historical silence hallucination or its stop reason")
                    if key == "wave1_runs" and v2_gate == make_dpo_v2_gate():
                        expected_selection = {"t3-w3": 40, "t3-w8": 18, "t3-w5": None}
                        if prefix in expected_selection and rep["v2_selected_step"] != expected_selection[prefix]:
                            raise ValueError(f"{run}: unexpected default v2 selected step")
                        if prefix == "t3-w5" and rep["steps"][0]["v2_risk_stop"]:
                            raise ValueError(f"{run}: default v2 stopped during observation")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            report["history_errors"].append(str(exc))
            print(f"历史回放失败 {exc}", file=sys.stderr)
            all_ok = False

    stress = run_synthetic_stress_suite(v2_gate)
    report["stress_suite"] = stress
    print("=== 3. 合成边界压测 (7 项安全与抗锁死用例) ===")
    for item in stress:
        status = "PASS" if item["passed"] else "FAIL"
        if not item["passed"]:
            all_ok = False
        print(f"  [{status}] {item['case']}: {item['detail']}")
    print()

    warmups = verify_warmup_schedules()
    report["warmup_schedules"] = warmups
    print("=== 4. DPO v2 LR Warmup + Cosine 曲线核验 ===")
    for item in warmups:
        status = "PASS" if item["passed"] else "FAIL"
        if not item["passed"]:
            all_ok = False
        pts = ", ".join(f"s{s['step']}={s['lr']:.2e}" for s in item["samples"])
        print(f"  [{status}] {item['name']} (w={item['warmup_steps']}/{item['max_steps']}): {pts}")

    report["all_checks_passed"] = all_ok
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
