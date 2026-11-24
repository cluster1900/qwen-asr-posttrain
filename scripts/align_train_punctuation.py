#!/usr/bin/env python3
"""Phase 6 Track 3B: Train-only clean-audio self-punctuation restoration.

Projects frozen Base Qwen3-ASR prosodic punctuation from train `clean` audio
predictions back onto official AISHELL-1 `text` via exact Levenshtein character
alignment (`evaluation.zh_cer.align_edits`), asserting 100% character invariance:
`normalize_zh_cer_v1(text_punct) == normalize_zh_cer_v1(text)` on every single row.

Strictly refuses any non-`train` row (`val`/`test`/OOD are never touched).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.zh_cer import align_edits, normalize_zh_cer_v1  # noqa: E402

ALLOWED_PUNCT = frozenset("，。？！、；：")
TERMINAL_PUNCT = frozenset("。？！")
PUNCT_MAP = {
    ",": "，",
    ".": "。",
    "?": "？",
    "!": "！",
    ";": "；",
    ":": "：",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_hyp_slots(hyp_raw: str) -> list[tuple[str, str]]:
    """Parse raw hypothesis into [(norm_char, trailing_punct), ...]."""
    slots: list[tuple[str, str]] = []
    for ch in hyp_raw:
        norm = normalize_zh_cer_v1(ch)
        if norm:
            for nc in norm:
                slots.append((nc, ""))
            continue
        mapped = PUNCT_MAP.get(ch, ch)
        if mapped in ALLOWED_PUNCT and slots:
            prev_ch, prev_p = slots[-1]
            # Keep at most one prosodic mark per slot (avoid noisy double marks).
            if not prev_p:
                slots[-1] = (prev_ch, mapped)
    return slots


def align_punctuation(ref_text: str, hyp_raw: str | None) -> str:
    """Project prosodic punctuation from `hyp_raw` onto `ref_text` with zero char edits."""
    ref_norm = normalize_zh_cer_v1(ref_text)
    if not ref_norm:
        raise ValueError("ref_text has no normalized characters")
    if not hyp_raw or not hyp_raw.strip():
        return ref_text

    # Map each normalized character back to raw character positions in ref_text.
    raw_chunks: list[str] = []
    for ch in ref_text:
        nc = normalize_zh_cer_v1(ch)
        if nc:
            if len(nc) == 1:
                raw_chunks.append(ch)
            else:
                raw_chunks.extend(list(nc))
        elif raw_chunks:
            raw_chunks[-1] += ch
    if len(raw_chunks) != len(ref_norm):
        raw_chunks = list(ref_norm)

    hyp_slots = _extract_hyp_slots(hyp_raw)
    if not hyp_slots:
        return ref_text

    hyp_norm = [c for c, _ in hyp_slots]
    ops = align_edits(list(ref_norm), hyp_norm)

    trailing_for_ref: list[str] = [""] * len(ref_norm)
    for op in ops:
        kind = op["op"]
        if kind in ("match", "substitute"):
            r_idx = int(op["ref_index"])
            h_idx = int(op["hyp_index"])
            p = hyp_slots[h_idx][1]
            if p and not trailing_for_ref[r_idx]:
                # Never place sentence-terminal punctuation in the middle unless followed by another clause;
                # allow `。？！` mid-sentence only if hyp actually placed it after a matched char.
                trailing_for_ref[r_idx] = p

    # Preserve terminal sentence punctuation if hyp ended with one even when final char was an insertion.
    last_hyp_punct = hyp_slots[-1][1]
    if last_hyp_punct in TERMINAL_PUNCT and not trailing_for_ref[-1]:
        trailing_for_ref[-1] = last_hyp_punct

    # Replace any trailing comma/semicolon/colon at sentence end with a period if hyp had terminal punct,
    # or drop hanging non-terminal comma at very end of utterance.
    if trailing_for_ref[-1] and trailing_for_ref[-1] not in TERMINAL_PUNCT:
        trailing_for_ref[-1] = last_hyp_punct if last_hyp_punct in TERMINAL_PUNCT else ""

    out = "".join(raw_chunks[i] + trailing_for_ref[i] for i in range(len(ref_norm)))
    if normalize_zh_cer_v1(out) != ref_norm:
        raise AssertionError(
            f"punctuation alignment violated character invariance: {normalize_zh_cer_v1(out)!r} != {ref_norm!r}"
        )
    return out


def extract_clean_train_manifest(train_manifest: Path, output_path: Path) -> dict[str, Any]:
    with train_manifest.open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    clean_rows: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for row in rows:
        if row.get("split") != "train":
            raise ValueError(f"refusing non-train row: {row.get('sample_id')} split={row.get('split')}")
        if row.get("scenario") == "clean":
            src = str(row["source_utterance_id"])
            if src in seen_sources:
                raise ValueError(f"duplicate clean source_utterance_id: {src}")
            seen_sources.add(src)
            clean_rows.append(row)
    if not clean_rows:
        raise ValueError("no clean train rows found")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as h:
        for row in clean_rows:
            h.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "schema_version": "qwen3-asr-clean-train-extract-v1",
        "train_manifest_sha256": sha256_file(train_manifest),
        "clean_rows": len(clean_rows),
        "output_sha256": sha256_file(output_path),
    }


def build_punctuated_train_manifest(
    train_manifest: Path,
    clean_predictions: Path,
    output_manifest: Path,
    audit_path: Path,
) -> dict[str, Any]:
    with train_manifest.open(encoding="utf-8") as fh:
        train_rows = [json.loads(line) for line in fh if line.strip()]
    with clean_predictions.open(encoding="utf-8") as fh:
        pred_rows = [json.loads(line) for line in fh if line.strip()]

    # Map clean sample_id -> prediction_raw, and bind to source_utterance_id via train clean rows.
    pred_by_sample: dict[str, dict[str, Any]] = {}
    for prow in pred_rows:
        sid = str(prow["sample_id"])
        if sid in pred_by_sample:
            raise ValueError(f"duplicate prediction sample_id: {sid}")
        pred_by_sample[sid] = prow

    clean_by_source: dict[str, dict[str, Any]] = {}
    for row in train_rows:
        if row.get("split") != "train":
            raise ValueError(f"refusing non-train row: {row.get('sample_id')} split={row.get('split')}")
        if row.get("scenario") == "clean":
            clean_by_source[str(row["source_utterance_id"])] = row

    punct_by_source: dict[str, str] = {}
    sources_with_punct = 0
    total_punct_marks = 0
    fallback_sources = 0
    for src, crow in clean_by_source.items():
        sid = str(crow["sample_id"])
        if sid not in pred_by_sample:
            raise ValueError(f"missing Base clean prediction for {sid}")
        prow = pred_by_sample[sid]
        ref_text = str(crow["text"])
        if prow.get("error"):
            aligned = ref_text
            fallback_sources += 1
        else:
            aligned = align_punctuation(ref_text, str(prow.get("prediction_raw") or ""))
        if normalize_zh_cer_v1(aligned) != normalize_zh_cer_v1(ref_text):
            raise AssertionError(f"invariance violation for {src}")
        n_marks = sum(1 for ch in aligned if ch in ALLOWED_PUNCT)
        if n_marks > 0:
            sources_with_punct += 1
            total_punct_marks += n_marks
        punct_by_source[src] = aligned

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    rows_with_punct = 0
    with output_manifest.open("w", encoding="utf-8") as out_h:
        for row in train_rows:
            src = str(row["source_utterance_id"])
            if src not in punct_by_source:
                raise ValueError(f"missing clean source for row {row.get('sample_id')}")
            text_punct = punct_by_source[src]
            if normalize_zh_cer_v1(text_punct) != normalize_zh_cer_v1(str(row["text"])):
                raise AssertionError(f"row invariance violation for {row.get('sample_id')}")
            out_row = dict(row)
            out_row["text_punct"] = text_punct
            out_h.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            rows_written += 1
            if any(ch in ALLOWED_PUNCT for ch in text_punct):
                rows_with_punct += 1

    audit = {
        "schema_version": "qwen3-asr-train-punct-audit-v1",
        "split": "train",
        "train_manifest_sha256": sha256_file(train_manifest),
        "clean_predictions_sha256": sha256_file(clean_predictions),
        "output_manifest_sha256": sha256_file(output_manifest),
        "sources_total": len(clean_by_source),
        "sources_with_punct": sources_with_punct,
        "sources_punct_rate": sources_with_punct / max(len(clean_by_source), 1),
        "fallback_sources": fallback_sources,
        "mean_punct_per_source": total_punct_marks / max(len(clean_by_source), 1),
        "rows_written": rows_written,
        "rows_with_punct": rows_with_punct,
        "char_mismatches": 0,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Track 3B 训练集 Clean 自对齐标点恢复工具")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ext = sub.add_parser("extract-clean", help="抽取仅限 train split 的 clean 行用于 Base 解码")
    p_ext.add_argument("--train-manifest", required=True, type=Path)
    p_ext.add_argument("--output", required=True, type=Path)

    p_bld = sub.add_parser("build", help="将 Base clean 预测标点零字符改动投射回 train.jsonl")
    p_bld.add_argument("--train-manifest", required=True, type=Path)
    p_bld.add_argument("--clean-predictions", required=True, type=Path)
    p_bld.add_argument("--output", required=True, type=Path)
    p_bld.add_argument("--audit-out", required=True, type=Path)

    args = parser.parse_args(argv)
    try:
        if args.cmd == "extract-clean":
            res = extract_clean_train_manifest(args.train_manifest, args.output)
        else:
            res = build_punctuated_train_manifest(
                args.train_manifest, args.clean_predictions, args.output, args.audit_out
            )
    except (OSError, ValueError, AssertionError, json.JSONDecodeError) as exc:
        print(f"ALIGN_PUNCT_FAIL {exc}", file=sys.stderr)
        return 2
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
