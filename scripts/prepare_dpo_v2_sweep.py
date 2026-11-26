#!/usr/bin/env python3
"""Validate materialized DPO v2 inputs, rehearse per-step audio count (ga), and write the four controlled recipes on CPU."""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rehearse_dpo_schedule import (  # noqa: E402
    choose_eval_steps,
    default_warmup_steps,
    format_rehearsal_table,
    rehearse_one,
)
from train.dpo_schema import load_dpo_config  # noqa: E402
from train.sft_schema import sha256_file  # noqa: E402
from train.train_dpo import check_input_metadata, check_null_pairs, prepare_pair_sets  # noqa: E402
from train.train_sft import RunStop  # noqa: E402


def portable(path: Path) -> str:
    path = path.resolve()
    return str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


def _apply_schedule_overrides(
    config: dict[str, Any],
    *,
    ga: int | None = None,
    max_steps: int | None = None,
    warmup_steps: int | None = None,
    eval_steps: int | None = None,
    min_stop_step: int | None = None,
    pool_size: int | None = None,
) -> None:
    """Apply optional per-step audio (ga) and horizon overrides while keeping warmup/eval/keep_last consistent."""
    schedule_changed = False
    if ga is not None:
        if ga < 1:
            raise ValueError("ga must be >= 1")
        config["batch"]["gradient_accumulation_steps"] = int(ga)
        schedule_changed = True
    eff_ga = int(config["batch"]["gradient_accumulation_steps"])
    if max_steps is not None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        config["train"]["max_steps"] = int(max_steps)
        schedule_changed = True
    elif ga is not None and pool_size is not None:
        auto_steps = pool_size // eff_ga
        if auto_steps < 1:
            raise ValueError(f"pool_size {pool_size} is smaller than ga {eff_ga}")
        config["train"]["max_steps"] = auto_steps
        schedule_changed = True
    eff_steps = int(config["train"]["max_steps"])
    if warmup_steps is not None:
        config["optim"]["warmup_steps"] = int(warmup_steps)
    elif schedule_changed:
        config["optim"]["warmup_steps"] = default_warmup_steps(eff_steps, 0.10)
    if eval_steps is not None:
        eff_eval = int(eval_steps)
        config["train"]["eval_steps"] = eff_eval
        config["train"]["save_steps"] = eff_eval
    elif schedule_changed:
        eff_eval = choose_eval_steps(eff_steps, target_evals=5)
        config["train"]["eval_steps"] = eff_eval
        config["train"]["save_steps"] = eff_eval
    eff_save = int(config["train"]["save_steps"])
    eff_eval = int(config["train"]["eval_steps"])
    config["train"]["keep_last"] = max(int(config["train"]["keep_last"]), math.ceil(eff_steps / eff_save))
    if min_stop_step is not None:
        config["gate"]["min_stop_step"] = int(min_stop_step)
    elif schedule_changed:
        config["gate"]["min_stop_step"] = max(int(config["optim"]["warmup_steps"]), eff_eval)


def prepare(
    manifest_dir: Path,
    pair_metadata: Path,
    pairs_dir: Path,
    output_dir: Path,
    *,
    ga: int | None = None,
    max_steps: int | None = None,
    warmup_steps: int | None = None,
    eval_steps: int | None = None,
    min_stop_step: int | None = None,
    rehearse_only: bool = False,
) -> dict[str, Any]:
    if not rehearse_only and output_dir.exists():
        raise ValueError(f"choose a new output directory: {output_dir}")
    recipe = yaml.safe_load((ROOT / "configs/experiments/dpo_v2_sweep.yaml").read_text())
    config = yaml.safe_load((ROOT / recipe["template"]).read_text())
    for section, overrides in recipe["common"].items():
        config[section].update(overrides)
    meta = json.loads(pair_metadata.read_text())
    dpo = config["dpo"]
    for split in ("train", "val"):
        manifest = manifest_dir / f"{split}.jsonl"
        config["data"][f"{split}_manifest"] = portable(manifest)
        pair_file = pairs_dir / f"{split}_pairs.jsonl"
        dpo[f"pair_{split}_manifest"] = portable(pair_file)
        dpo[f"null_{split}_pairs_sha256"] = meta[f"{split}_pairs_sha256"]
        dpo[f"filled_{split}_pairs_sha256"] = sha256_file(pair_file)
    config["data"]["manifest_metadata"] = portable(manifest_dir / "manifest.json")
    if not (manifest_dir / "manifest.json").is_file():
        raise ValueError("missing manifest.json")
    dpo["pair_metadata"] = portable(pair_metadata)
    dpo["pair_metadata_sha256"] = sha256_file(pair_metadata)
    check_input_metadata(pair_metadata, config, {"adapter_sha256": config["model"]["parent_release_sha256"]})
    check_null_pairs(pair_metadata.parent, config)
    for key, name in (("reference_report", "ref_report.json"), ("reference_cache", "ref_logprobs.jsonl")):
        dpo[key] = portable(pairs_dir / name)
        dpo[f"{key}_sha256"] = sha256_file(pairs_dir / name)

    # Apply explicit schedule overrides before first validation; if only --ga was given,
    # measure the validated large pool first and auto-fit max_steps to 1 exact epoch.
    _apply_schedule_overrides(
        config,
        ga=ga,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        eval_steps=eval_steps,
        min_stop_step=min_stop_step,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    rehearsal_summary: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(dir=output_dir.parent) as tmp:
        stage = Path(tmp) / "recipes"
        stage.mkdir()

        # Validate the large pool first to discover exact pool_size and fail closed on missing anchors/floors.
        probe_path = stage / "_probe.yaml"
        probe_cfg = copy.deepcopy(config)
        probe_cfg["optim"]["learning_rate"] = recipe["learning_rates"][0]
        probe_path.write_text(yaml.safe_dump(probe_cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        probe_resolved = load_dpo_config(probe_path)
        probe_path.unlink()
        if meta["val_manifest_sha256"] != probe_resolved["baseline"]["manifest_sha256"]:
            raise ValueError("validation manifest differs from the pinned SFT/base validation set")
        large_sets = prepare_pair_sets(probe_resolved)

        if ga is not None and max_steps is None:
            _apply_schedule_overrides(
                config,
                ga=ga,
                max_steps=None,
                warmup_steps=warmup_steps,
                eval_steps=eval_steps,
                min_stop_step=min_stop_step,
                pool_size=large_sets.pool_size,
            )

        eff_ga = int(config["batch"]["gradient_accumulation_steps"])
        eff_steps = int(config["train"]["max_steps"])
        required_pairs = eff_steps * eff_ga
        if large_sets.pool_size < required_pairs:
            raise ValueError(
                f"large pool has {large_sets.pool_size} pairs < max_steps({eff_steps}) * ga({eff_ga}) = "
                f"{required_pairs}; adjust --ga or --max-steps, or expand the training pool"
            )

        rehearsal_summary = rehearse_one(
            train_pairs=large_sets.pool_size,
            ga=eff_ga,
            max_epochs=int(config["train"]["max_epochs"]),
            max_steps=eff_steps,
            base_lr=float(recipe["learning_rates"][-1]),
            beta=float(config["dpo"]["beta"]),
            warmup_steps=int(config["optim"]["warmup_steps"]),
            min_lr_ratio=float(config["optim"]["min_lr_ratio"]),
            eval_steps=int(config["train"]["eval_steps"]),
            min_stop_step=int(config["gate"].get("min_stop_step", 0)),
            patience=int(config["gate"].get("max_consecutive_non_passing_evals", 2)),
            small_pool_pairs=int(recipe["small_pool_pairs"]),
            max_runtime_minutes=int(config["runtime"]["max_runtime_minutes"]),
            shutdown_margin_minutes=int(config["runtime"]["shutdown_margin_minutes"]),
        )
        print(format_rehearsal_table({"train_pairs": large_sets.pool_size, "recommended_ga": eff_ga, "rows": [rehearsal_summary]}))

        for pool in ("large", "small"):
            pool_cfg = copy.deepcopy(config)
            if pool == "small":
                limit = int(recipe["small_pool_pairs"])
                pool_cfg["dpo"]["train_pair_limit"] = limit
                updates = math.ceil(limit / pool_cfg["batch"]["gradient_accumulation_steps"])
                pool_cfg["train"]["max_epochs"] = math.ceil(pool_cfg["train"]["max_steps"] / updates)
            for index, lr in enumerate(recipe["learning_rates"]):
                pool_cfg["optim"]["learning_rate"] = lr
                path = stage / f"dpo_v2_{pool}_lr{lr:.0e}.yaml"
                path.write_text(yaml.safe_dump(pool_cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
                resolved = load_dpo_config(path)
                if index == 0 and pool == "small":
                    prepare_pair_sets(resolved)
        if not rehearse_only:
            stage.rename(output_dir)
    return rehearsal_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest-dir", "pair-metadata", "pairs-dir", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--ga", type=int, default=None, help="灵活覆盖每步音频/偏好对数 (gradient_accumulation_steps)")
    parser.add_argument("--max-steps", type=int, default=None, help="灵活覆盖总步数 (省略但设了 --ga 时按大池 1 epoch 自动推算)")
    parser.add_argument("--warmup-steps", type=int, default=None, help="可选覆盖 warmup 步数 (默认按 10%% 联动)")
    parser.add_argument("--eval-steps", type=int, default=None, help="可选覆盖评测/保存间隔 (默认联动为约 5 次全量评测)")
    parser.add_argument("--min-stop-step", type=int, default=None, help="可选覆盖门禁预热观察窗 (默认 >= max(warmup, eval))")
    parser.add_argument("--rehearse-only", action="store_true", help="仅校验输入并打印步数/耗时预演，不写出配置文件")
    args = parser.parse_args(argv)
    try:
        prepare(
            args.manifest_dir,
            args.pair_metadata,
            args.pairs_dir,
            args.output_dir,
            ga=args.ga,
            max_steps=args.max_steps,
            warmup_steps=args.warmup_steps,
            eval_steps=args.eval_steps,
            min_stop_step=args.min_stop_step,
            rehearse_only=args.rehearse_only,
        )
    except (OSError, ValueError, KeyError, RunStop) as exc:
        print(f"DPO_CONFIG_FAIL {exc}", file=sys.stderr)
        return 1
    if args.rehearse_only:
        print("DPO_REHEARSE_OK")
    else:
        print(f"DPO_CONFIG_READY {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
