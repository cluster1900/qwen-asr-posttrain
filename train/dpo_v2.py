"""Data and comparison contracts for DPO runs using the relaxed checkpoint gate."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from evaluation.zh_cer import normalize_zh_cer_v1
from train.dpo_contract import preference_accuracy
from train.sft_schema import canonical_json, sha256_text

DEFAULT_VAL_COHORT = Path("configs/train/dpo_track3b_val_cohorts.json")
PHYSICAL = ("noise", "reverb", "far_field")
V2_GATE_FIELDS = {
    "degraded_macro_scenarios", "degraded_macro_stop_increase_max", "dropout_cer_increase_max",
    "max_gap_insertions", "stop_on_low_preference", "min_stop_step", "max_consecutive_non_passing_evals",
}


def uses_v2_gate(gate: dict[str, Any]) -> bool:
    return not V2_GATE_FIELDS.isdisjoint(gate)


def physical_clean_map(rows: list[dict[str, Any]]) -> dict[str, str]:
    controls = {}
    for row in rows:
        if row["scenario"] not in PHYSICAL:
            continue
        source = row.get("source_utterance_id")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"missing source for physical pair {row.get('pair_id')}")
        text = row.get("clean_prediction_normalized")
        if not isinstance(text, str) or not text or normalize_zh_cer_v1(text) != text:
            raise ValueError(f"{source} missing clean control; build pairs with --require-clean-controls")
        if controls.setdefault(source, text) != text:
            raise ValueError(f"inconsistent clean control for {source}")
    return controls


def cohort_digest(rows: list[dict[str, Any]]) -> str:
    fields = ("pair_id", "source_utterance_id", "scenario", "chosen_text", "rejected_text")
    return sha256_text(canonical_json([[r[k] for k in fields] for r in rows]))


def fixed_val_cohorts(rows: list[dict], spec: dict) -> tuple[list[dict], list[dict]]:
    if spec.get("schema_version") != "qwen3-asr-dpo-val-cohorts-v1":
        raise ValueError("invalid DPO val cohort schema")
    by_id = {r["pair_id"]: r for r in rows}
    seen: set[str] = set()
    if len(by_id) != len(rows) or any(r["split"] != "val" for r in rows):
        raise ValueError("val cohorts require unique validation pairs")

    def members(name: str) -> list[dict]:
        block = spec[name]
        ids = block["pair_ids"]
        if not isinstance(ids, list) or (name == "primary" and not ids):
            raise ValueError(f"invalid {name} pair IDs")
        if len(set(ids)) != len(ids) or seen.intersection(ids) or set(ids) - by_id.keys():
            raise ValueError(f"missing, duplicate or overlapping {name} val pair IDs")
        selected = [by_id[i] for i in ids]
        if cohort_digest(selected) != block["members_sha256"]:
            raise ValueError(f"{name} val pair content changed")
        seen.update(ids)
        return selected

    return members("primary"), members("supplemental")


def check_reference_context(report: dict, config: dict, gpu_name: str | None) -> None:
    dpo = config["dpo"]
    recorded = report.get("reference_context") or {}
    if not isinstance(recorded, dict) or not isinstance(recorded.get("gpu_name"), str) or not recorded["gpu_name"]:
        raise ValueError("reference cache has no GPU identity")
    expected = {
        "adapter_sha256": config["model"]["parent_release_sha256"],
        "lora_dtype": config["lora"].get("trainable_dtype") or "bfloat16",
        "decode_eval_sha256": config["decode_eval_sha256"],
        "gpu_name": gpu_name if gpu_name is not None else recorded["gpu_name"],
        "null_train_pairs_sha256": dpo["null_train_pairs_sha256"],
        "null_val_pairs_sha256": dpo["null_val_pairs_sha256"],
    }
    if report.get("reference_context") != expected or report.get("cache_sha256") != dpo["reference_cache_sha256"]:
        raise ValueError("reference cache inputs, parent, precision or GPU do not match this DPO run")
    for split in ("train", "val"):
        if report.get(f"{split}_pairs_sha256") != dpo[f"filled_{split}_pairs_sha256"]:
            raise ValueError(f"reference report does not match filled {split} pairs")
    if (report.get("release_id") != config["model"]["parent_release"]
            or type(report.get("rechecked")) is not int or report["rechecked"] < 1):
        raise ValueError("reference report is missing a matching parent or recheck")
    error = report.get("max_abs_error")
    if type(error) not in (float, int) or not 0 <= error <= 1e-4:
        raise ValueError("reference cache recheck exceeds 1e-4")


def check_step0_score(policy: float, reference: float, pair_id: str) -> float:
    delta = abs(policy - reference)
    if not math.isfinite(delta) or delta > 1e-4:
        raise ValueError(f"step-0 logprob differs from reference for {pair_id} by {delta}")
    return delta


def preference_evidence(zero: dict, candidate: dict, cohort: dict, cohort_sha: str) -> dict:
    """Recompute both preference measures from the same ordered pairs at both steps."""
    evidence = {}
    for name in ("primary", "supplemental"):
        ids = cohort[name]["pair_ids"]
        if not ids and name == "supplemental":
            continue
        baseline = zero if name == "primary" else zero["supplemental"]
        selected = candidate if name == "primary" else candidate["supplemental"]
        margins = []
        for block in (baseline, selected):
            rows = block["pair_margins"]
            if (block.get("val_cohort_sha256") != cohort_sha or block.get("pairs") != len(ids)
                    or [row["pair_id"] for row in rows] != ids):
                raise ValueError(f"{name} preference cohort differs from the fixed val pairs")
            values = [row["policy_margin"] for row in rows]
            if any(not math.isfinite(value) for value in values):
                raise ValueError(f"{name} has nonfinite policy margins")
            if block["raw_policy_accuracy"] != sum(value > 0 for value in values) / len(ids):
                raise ValueError(f"{name} raw policy accuracy differs from pair margins")
            margins.append(values)
        if (baseline["accuracy"] != 0.5 or baseline["ties"] != len(ids)
                or not 0 <= baseline["max_abs_error"] <= 1e-4):
            raise ValueError(f"{name} has invalid step-0 evidence")
        relative = preference_accuracy([end - start for start, end in zip(*margins)])
        if any(selected.get(key) != value for key, value in relative.items()):
            raise ValueError(f"{name} preference accuracy or counts differ from step-0 margin changes")
        evidence[name] = {
            **relative, "step0_raw_policy_accuracy": baseline["raw_policy_accuracy"],
            "raw_policy_accuracy": selected["raw_policy_accuracy"],
        }
    return evidence
