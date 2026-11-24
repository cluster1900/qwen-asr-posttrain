#!/usr/bin/env python3
"""Deterministic sharded greedy decode and exact-order merge for multi-A100 parallel runs.

Uses the exact training/release greedy decode path (`decode_eval_rows` with
`tf32=False`, `deterministic_algorithms=True`, `eval_batch_size=1`, `do_sample=False`).
Each shard decodes a contiguous slice `[start:end]` of the manifest independently,
supporting prefix resume and exact 1:1 `sample_id` order verification upon merge.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.eval_cer import evaluate, write_scenario_csv  # noqa: E402
from evaluation.jsonl_io import read_jsonl  # noqa: E402
from inference.config import load_infer_config  # noqa: E402
from scripts.run_phase5_test_benchmark import zero_lora_parameters  # noqa: E402
from train.sft_checkpoint import atomic_write_json, sha256_file  # noqa: E402
from train.sft_contract import SCENARIOS  # noqa: E402
from train.sft_release import _match_training_runtime, completed_hypothesis_rows  # noqa: E402
from train.sft_schema import load_train_config  # noqa: E402


def shard_bounds(total_rows: int, shard_index: int, num_shards: int) -> tuple[int, int]:
    """Return contiguous `[start, end)` bounds partitioning `total_rows` across `num_shards`."""
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index {shard_index} out of range [0, {num_shards})")
    base = total_rows // num_shards
    rem = total_rows % num_shards
    start = shard_index * base + min(shard_index, rem)
    end = start + base + (1 if shard_index < rem else 0)
    return start, end


def merge_prediction_shards(
    manifest_rows: list[dict[str, Any]],
    shard_paths: list[Path],
    output_path: Path,
) -> dict[str, Any]:
    """Merge shard JSONLs and verify exact 1:1 `sample_id` order against `manifest_rows`."""
    expected_ids = [str(r["sample_id"]) for r in manifest_rows]
    seen_by_id: dict[str, dict[str, Any]] = {}
    for sp in shard_paths:
        if not sp.is_file():
            raise FileNotFoundError(f"missing shard file: {sp}")
        for row in read_jsonl(sp):
            sid = str(row.get("sample_id"))
            if sid in seen_by_id:
                raise ValueError(f"duplicate sample_id across shards: {sid}")
            seen_by_id[sid] = row
    missing = [sid for sid in expected_ids if sid not in seen_by_id]
    extra = sorted(set(seen_by_id) - set(expected_ids))
    if missing:
        raise ValueError(f"missing {len(missing)} sample_ids in merged shards, first={missing[:5]}")
    if extra:
        raise ValueError(f"unexpected {len(extra)} extra sample_ids in merged shards, first={extra[:5]}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(output_path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for sid in expected_ids:
            handle.write(json.dumps(seen_by_id[sid], ensure_ascii=False) + "\n")
    tmp.replace(output_path)
    return {
        "rows": len(expected_ids),
        "shards": len(shard_paths),
        "predictions_sha256": sha256_file(output_path),
    }


def write_baseline_json(
    metrics: dict[str, Any],
    *,
    manifest_sha256: str,
    predictions_sha256: str,
    decode_eval_sha256: str,
    output_path: Path,
    split: str = "val",
    parent: str = "base",
) -> dict[str, Any]:
    """Emit a pinned `qwen3-asr-baseline-v1` JSON compatible with `sft_schema.py`."""
    by_scenario = {}
    for sc in SCENARIOS:
        block = metrics["by_scenario"][sc]
        by_scenario[sc] = {
            "samples": int(block["samples"]),
            "micro_cer": float(block["micro_cer"]),
        }
    payload = {
        "schema_version": "qwen3-asr-baseline-v1",
        "split": split,
        "parent": parent,
        "manifest_sha256": manifest_sha256,
        "predictions_sha256": predictions_sha256,
        "decode_eval_sha256": decode_eval_sha256,
        "samples": int(metrics["samples"]),
        "by_scenario": by_scenario,
    }
    atomic_write_json(output_path, payload)
    return payload


def run_decode_shard(
    *,
    config_path: Path,
    manifest_path: Path,
    output_path: Path,
    shard_index: int,
    num_shards: int,
    adapter_path: Path | None = None,
    step_num: int = 0,
    seed: int = 42,
    model_loader: Callable[[dict[str, Any]], tuple[Any, Any, Any]] | None = None,
    adapter_loader: Callable[[Any, Path], None] | None = None,
    decode_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    config = load_train_config(config_path)
    if int(seed) != int(config["seed"]):
        raise ValueError(f"seed mismatch: {seed} != {config['seed']}")
    rows = read_jsonl(manifest_path)
    start, end = shard_bounds(len(rows), shard_index, num_shards)
    shard_rows = rows[start:end]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = completed_hypothesis_rows(output_path, shard_rows)
    print(
        f"SHARD_START shard={shard_index}/{num_shards} range=[{start},{end}) resume={done}/{len(shard_rows)}",
        flush=True,
    )
    if done < len(shard_rows):
        if model_loader is None or adapter_loader is None or decode_fn is None:
            from train.train_sft import decode_eval_rows, load_adapter, load_sft_model

            model_loader = model_loader or load_sft_model
            adapter_loader = adapter_loader or load_adapter
            decode_fn = decode_fn or decode_eval_rows

        _match_training_runtime(config)
        model, processor, _trainable = model_loader(config)
        if adapter_path is None:
            zero_lora_parameters(model)
            adapter_sha = "base_zero_lora"
        else:
            adapter_loader(model, adapter_path)
            adapter_sha = sha256_file(adapter_path)

        written = done
        with output_path.open("a" if done else "w", encoding="utf-8") as handle:

            def on_row(pred: dict[str, Any]) -> None:
                nonlocal written
                handle.write(json.dumps(pred, ensure_ascii=False) + "\n")
                handle.flush()
                written += 1

            def on_progress(finished: int, _total: int) -> None:
                absolute = done + finished
                if absolute % 100 == 0 or absolute == len(shard_rows):
                    print(f"SHARD_PROGRESS shard={shard_index}/{num_shards} done={absolute}/{len(shard_rows)}", flush=True)

            decode_fn(
                model,
                processor,
                shard_rows[done:],
                manifest_path,
                config,
                int(step_num),
                adapter_sha,
                on_progress=on_progress,
                on_row=on_row,
            )
        if written != len(shard_rows):
            raise ValueError(f"shard {shard_index} wrote {written} rows, expected {len(shard_rows)}")
    sha = sha256_file(output_path)
    print(f"SHARD_DONE shard={shard_index}/{num_shards} rows={len(shard_rows)} sha256={sha}", flush=True)
    return {"shard_index": shard_index, "num_shards": num_shards, "rows": len(shard_rows), "sha256": sha}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="多 A100 确定性分片 greedy 推理与严格顺序合并工具")
    sub = parser.add_subparsers(dest="command", required=True)

    p_dec = sub.add_parser("decode", help="解码指定分片")
    p_dec.add_argument("--config", required=True, type=Path)
    p_dec.add_argument("--manifest", required=True, type=Path)
    p_dec.add_argument("--output", required=True, type=Path)
    p_dec.add_argument("--shard-index", required=True, type=int)
    p_dec.add_argument("--num-shards", required=True, type=int)
    p_dec.add_argument("--adapter-path", type=Path, default=None)
    p_dec.add_argument("--step", type=int, default=0)
    p_dec.add_argument("--seed", type=int, default=42)

    p_mrg = sub.add_parser("merge", help="按 manifest sample_id 精确顺序合并分片并可选评分")
    p_mrg.add_argument("--manifest", required=True, type=Path)
    p_mrg.add_argument("--shards", required=True, nargs="+", type=Path)
    p_mrg.add_argument("--output", required=True, type=Path)
    p_mrg.add_argument("--eval-config", type=Path, default=None)
    p_mrg.add_argument("--eval-output-dir", type=Path, default=None)
    p_mrg.add_argument("--baseline-json-out", type=Path, default=None)

    args = parser.parse_args(argv)
    try:
        if args.command == "decode":
            run_decode_shard(
                config_path=args.config,
                manifest_path=args.manifest,
                output_path=args.output,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
                adapter_path=args.adapter_path,
                step_num=args.step,
                seed=args.seed,
            )
            return 0
        if args.command == "merge":
            manifest_rows = read_jsonl(args.manifest)
            info = merge_prediction_shards(manifest_rows, args.shards, args.output)
            if args.eval_config is not None:
                infer_cfg = load_infer_config(args.eval_config)
                preds = read_jsonl(args.output)
                manifest_sha = sha256_file(args.manifest)
                scored, metrics = evaluate(
                    manifest_rows,
                    preds,
                    manifest_sha256=manifest_sha,
                    predictions_sha256=info["predictions_sha256"],
                    decode_eval_sha256=infer_cfg["decode_eval_sha256"],
                    wer_tokenizer=infer_cfg["scoring"].get("wer_tokenizer"),
                    wer_tokenizer_version=infer_cfg["scoring"].get("wer_tokenizer_version"),
                    wer_dictionary_sha256=infer_cfg["scoring"].get("wer_dictionary_sha256"),
                    wer_hmm=bool(infer_cfg["scoring"].get("wer_hmm", False)),
                )
                if args.eval_output_dir is not None:
                    args.eval_output_dir.mkdir(parents=True, exist_ok=True)
                    with (args.eval_output_dir / "scored.jsonl").open("w", encoding="utf-8") as h:
                        for row in scored:
                            h.write(json.dumps(row, ensure_ascii=False) + "\n")
                    atomic_write_json(args.eval_output_dir / "metrics.json", metrics)
                    write_scenario_csv(args.eval_output_dir / "by_scenario.csv", metrics["by_scenario"])
                if args.baseline_json_out is not None:
                    write_baseline_json(
                        metrics,
                        manifest_sha256=manifest_sha,
                        predictions_sha256=info["predictions_sha256"],
                        decode_eval_sha256=infer_cfg["decode_eval_sha256"],
                        output_path=args.baseline_json_out,
                    )
                info["micro_cer"] = metrics["micro_cer"]
                info["micro_wer"] = metrics.get("micro_wer")
                info["edit_count"] = metrics["edit_count"]
            print(json.dumps(info, ensure_ascii=False, indent=2))
            return 0
    except (OSError, ValueError) as exc:
        print(f"SHARDED_DECODE_ERROR: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
