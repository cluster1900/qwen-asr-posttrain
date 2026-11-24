#!/usr/bin/env python3
"""Build and independently validate a Phase 6 Track 3 SLR28 physical-augmentation manifest."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.jsonl_io import read_jsonl, write_jsonl  # noqa: E402
from scripts.build_manifest import (  # noqa: E402
    KNOWN_AUDIO_REASONS,
    _drop_cross_split_text,
    _drop_duplicate_audio,
    _drop_speaker_cross_split,
    _git_sha,
    _load_sources,
    _read_audio,
    _rejected,
    _reset_output,
    _write_sha_index,
    _write_splits,
)
from scripts.manifest_audio import load_mono_16k  # noqa: E402
from scripts.manifest_contract import (  # noqa: E402
    MANIFEST_ROW_SCHEMA,
    MANIFEST_SET_SCHEMA,
    SCENARIOS,
    SPLITS,
    STATUS_CODE,
    frame_bounds,
    sample_id_for,
    sha256_file,
    split_mode,
)
from scripts.manifest_validate import (  # noqa: E402
    _check_audio,
    _check_official_row,
    _check_speaker_identity,
    _counts,
    _duration_summary,
    _quality,
    _read_rejected,
    _read_rows,
    _shortages,
    _split_audit,
    _status,
    ROW_FIELDS,
)
from scripts.slr28_audio import (  # noqa: E402
    SLR28_PROCESSOR_VERSION,
    Slr28AssetCatalog,
    load_slr28_data_config,
    render_slr28_scenarios,
)

PRODUCTS_SLR28 = (
    "train.jsonl",
    "val.jsonl",
    "test.jsonl",
    "rejected.jsonl",
    "manifest.json",
    "slr28_split_audit.json",
    "sha256.json",
)

_WORKER_ZF: zipfile.ZipFile | None = None
_WORKER_CATALOG: Slr28AssetCatalog | None = None
_WORKER_CONFIG: dict[str, Any] | None = None
_WORKER_CACHE: dict[tuple[str, int], np.ndarray] | None = None


def _init_worker(archive_path: str, config: dict[str, Any]) -> None:
    global _WORKER_ZF, _WORKER_CATALOG, _WORKER_CONFIG, _WORKER_CACHE
    _WORKER_CONFIG = config
    _WORKER_CATALOG = Slr28AssetCatalog(
        Path(archive_path),
        seed=int(config["seed"]),
        val_fraction=float(config["slr28"]["split_fractions"]["val"]),
        test_fraction=float(config["slr28"]["split_fractions"]["test"]),
    )
    _WORKER_ZF = zipfile.ZipFile(archive_path, "r")
    _WORKER_CACHE = {}


def _render_chunk(
    items: list[dict[str, Any]],
    output_dir_str: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assert _WORKER_ZF is not None and _WORKER_CATALOG is not None and _WORKER_CONFIG is not None
    output_dir = Path(output_dir_str)
    config = _WORKER_CONFIG
    minimum, maximum = frame_bounds(config)
    rendered: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for item in items:
        try:
            clean = load_mono_16k(str(item["audio_path"]), _read_audio)
            if clean.size < minimum or clean.size > maximum:
                raise ValueError("duration_out_of_range")
            waves, aug_meta = render_slr28_scenarios(
                clean,
                seed=int(config["seed"]),
                source_id=item["source_id"],
                split=item["split"],
                config=config,
                catalog=_WORKER_CATALOG,
                zf=_WORKER_ZF,
                wav_cache=_WORKER_CACHE,
            )
            rows = []
            for scenario in SCENARIOS:
                sample_id = sample_id_for(item["source_id"], scenario)
                relative = f"wavs/{sample_id}.wav"
                destination = output_dir / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                sf.write(destination, np.clip(waves[scenario], -1.0, 1.0), 16000, format="WAV", subtype="PCM_16")
                row: dict[str, Any] = {
                    "schema_version": MANIFEST_ROW_SCHEMA,
                    "sample_id": sample_id,
                    "source_utterance_id": item["source_id"],
                    "audio": relative,
                    "text": item["text"],
                    "language": "zh",
                    "scenario": scenario,
                    "split": item["split"],
                }
                if "speaker_id" in item:
                    row["speaker_id"] = item["speaker_id"]
                    row["official_split"] = item["official_split"]
                if "source_package" in item:
                    row["source_package"] = item["source_package"]
                if scenario != "clean":
                    row["augmentation"] = aug_meta[scenario]
                rows.append(row)
        except Exception as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            reason = str(exc) if str(exc) in KNOWN_AUDIO_REASONS else f"audio_unreadable:{type(exc).__name__}"
            for sc in SCENARIOS:
                wav_p = output_dir / "wavs" / f"{sample_id_for(item['source_id'], sc)}.wav"
                if wav_p.exists():
                    wav_p.unlink()
            rejected.append(_rejected(config, item["source_id"], item["source_id"], reason, item["audio"]))
            continue
        rendered.append({**item, "rows": rows, "clean_sha256": sha256_file(output_dir / rows[0]["audio"])})
    return rendered, rejected


def inspect_slr28_manifest(manifest_dir: Path, config: dict[str, Any], check_audio: bool) -> dict[str, Any]:
    """Validate Track 3 SLR28 manifest rows, audio, speaker/text/source isolation, and SLR28 room/noise isolation."""
    from scripts.manifest_contract import safe_relative_audio, text_problem

    problems: list[str] = []
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        path = manifest_dir / f"{split}.jsonl"
        if not path.is_file():
            problems.append(f"missing_{split}_jsonl")
            continue
        rows.extend(_read_rows(path, split, problems))
    rejected = _read_rejected(manifest_dir / "rejected.jsonl", problems)

    seen_ids: set[str] = set()
    seen_audio: set[str] = set()
    source_splits: dict[str, set[str]] = defaultdict(set)
    texts: dict[str, set[str]] = {split: set() for split in SPLITS}
    rir_families_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    noise_groups_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}

    for row in rows:
        sample_id = row.get("sample_id")
        source_id = row.get("source_utterance_id")
        scenario = row.get("scenario")
        split = row.get("split")
        if row.get("schema_version") != MANIFEST_ROW_SCHEMA or set(row) - ROW_FIELDS:
            problems.append(f"bad_row_schema:{sample_id}")
        if not isinstance(sample_id, str) or sample_id in seen_ids:
            problems.append(f"duplicate_or_bad_sample_id:{sample_id}")
        else:
            seen_ids.add(sample_id)
        if scenario not in SCENARIOS or split not in SPLITS:
            problems.append(f"bad_enum:{sample_id}")
        elif isinstance(source_id, str) and sample_id != sample_id_for(source_id, scenario):
            problems.append(f"sample_id_mismatch:{sample_id}")
        if row.get("language") != "zh":
            problems.append(f"language:{sample_id}")
        tp = text_problem(row.get("text"))
        if tp:
            problems.append(f"{tp}:{sample_id}")
        audio = row.get("audio")
        if not safe_relative_audio(audio):
            problems.append(f"unsafe_audio_path:{sample_id}")
        elif audio in seen_audio:
            problems.append(f"duplicate_audio_path:{audio}")
        elif isinstance(audio, str):
            seen_audio.add(audio)

        aug = row.get("augmentation")
        if scenario == "clean":
            if "augmentation" in row:
                problems.append(f"clean_has_augmentation:{sample_id}")
        else:
            if not isinstance(aug, dict):
                problems.append(f"missing_augmentation:{sample_id}")
            else:
                if aug.get("processor_version") != SLR28_PROCESSOR_VERSION:
                    problems.append(f"processor_version:{sample_id}")
                if "audio" in aug or "text" in aug:
                    problems.append(f"augmentation_overwrites_row:{sample_id}")
                if isinstance(split, str) and split in SPLITS:
                    if isinstance(aug.get("rir_room_family"), str):
                        rir_families_by_split[split].add(aug["rir_room_family"])
                    if isinstance(aug.get("noise_group"), str):
                        noise_groups_by_split[split].add(aug["noise_group"])

        if isinstance(source_id, str) and isinstance(split, str):
            source_splits[source_id].add(split)
            if isinstance(row.get("text"), str) and split in texts:
                texts[split].add(row["text"])
        _check_official_row(row, problems)

    for source_id, splits in sorted(source_splits.items()):
        if len(splits) > 1:
            problems.append(f"source_leakage:{source_id}")
    _check_speaker_identity(rows, problems)
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if texts[left] & texts[right]:
            problems.append(f"text_leakage:{left}:{right}")
        if rir_families_by_split[left] & rir_families_by_split[right]:
            problems.append(f"slr28_rir_leakage:{left}:{right}")
        if noise_groups_by_split[left] & noise_groups_by_split[right]:
            problems.append(f"slr28_noise_leakage:{left}:{right}")

    durations = _check_audio(manifest_dir, rows, config, check_audio, problems) if check_audio else []
    quality = _quality(rows, problems)
    counts = _counts(rows, len(rejected))
    shortages = _shortages(counts, config)
    status = _status(rows, problems, shortages)
    return {
        "schema_version": MANIFEST_SET_SCHEMA,
        "status": status,
        "seed": config["seed"],
        "source_version": config["source_version"],
        "audited": config["audited"],
        "audio_processor_version": config["audio_processor_version"],
        "config_sha256": config["config_sha256"],
        "counts": counts,
        "quality": quality,
        "duration_seconds": _duration_summary(durations),
        "shortages": shortages,
        "problems": problems,
        "split_audit": _split_audit(rows, rejected),
        "slr28_usage_audit": {
            "rir_room_families_used": {sp: len(rir_families_by_split[sp]) for sp in SPLITS},
            "noise_groups_used": {sp: len(noise_groups_by_split[sp]) for sp in SPLITS},
            "rir_leakage": {
                f"{l}_{r}": sorted(rir_families_by_split[l] & rir_families_by_split[r])
                for l, r in (("train", "val"), ("train", "test"), ("val", "test"))
            },
            "noise_leakage": {
                f"{l}_{r}": sorted(noise_groups_by_split[l] & noise_groups_by_split[r])
                for l, r in (("train", "val"), ("train", "test"), ("val", "test"))
            },
        },
    }


def validate_slr28_manifest_dir(
    manifest_dir: Path,
    config: dict[str, Any],
    check_audio: bool,
) -> tuple[dict[str, Any], int]:
    if not check_audio:
        report = inspect_slr28_manifest(manifest_dir, config, False)
        report["status"] = "failed"
        report["problems"] = [*report["problems"], "check_audio_required"]
        return report, 2
    report = inspect_slr28_manifest(manifest_dir, config, True)
    stored_path = manifest_dir / "manifest.json"
    if not stored_path.is_file():
        report["status"] = "failed"
        report["problems"] = [*report["problems"], "missing_manifest_json"]
        return report, 1
    stored = json.loads(stored_path.read_text(encoding="utf-8"))
    for key in (
        "status",
        "counts",
        "quality",
        "shortages",
        "duration_seconds",
        "seed",
        "config_sha256",
        "split_audit",
        "slr28_usage_audit",
    ):
        if stored.get(key) != report[key]:
            report["status"] = "failed"
            report["problems"] = [*report["problems"], f"manifest_mismatch:{key}"]
    audit_path = manifest_dir / "slr28_split_audit.json"
    if not audit_path.is_file():
        report["status"] = "failed"
        report["problems"] = [*report["problems"], "missing_slr28_split_audit"]
    else:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("zero_leakage") is not True:
            report["status"] = "failed"
            report["problems"] = [*report["problems"], "slr28_split_audit_leakage"]
    return report, STATUS_CODE[report["status"]]


def build_slr28_manifest(
    config: dict[str, Any],
    sources_path: Path,
    slr28_archive: Path,
    output_dir: Path,
    git_sha: str | None,
    *,
    workers: int = 1,
    verify_archive_sha256: bool = True,
) -> dict[str, Any]:
    """Materialize Track 3 SLR28 WAVs and manifest directory, then independently validate."""
    if not slr28_archive.is_file():
        raise ValueError(f"missing SLR28 archive: {slr28_archive}")
    actual_slr_sha = sha256_file(slr28_archive)
    if verify_archive_sha256 and actual_slr_sha != config["slr28"]["archive_sha256"]:
        raise ValueError(
            f"SLR28 archive sha256 mismatch: {actual_slr_sha} != {config['slr28']['archive_sha256']}"
        )
    catalog = Slr28AssetCatalog(
        slr28_archive,
        seed=int(config["seed"]),
        val_fraction=float(config["slr28"]["split_fractions"]["val"]),
        test_fraction=float(config["slr28"]["split_fractions"]["test"]),
    )

    raw_rows = read_jsonl(sources_path)
    mode = split_mode(raw_rows)
    _reset_output(output_dir)
    for name in PRODUCTS_SLR28:
        p = output_dir / name
        if p.exists():
            p.unlink()

    accepted, rejected = _load_sources(raw_rows, sources_path.parent, config, mode)
    if mode == "official":
        accepted = _drop_speaker_cross_split(accepted, rejected, config)
    accepted = _drop_cross_split_text(accepted, rejected, config)

    # Render WAVs (parallel across deterministic chunks if workers > 1)
    rendered: list[dict[str, Any]] = []
    eff_workers = max(1, min(int(workers), max(1, len(accepted))))
    if eff_workers == 1:
        _init_worker(str(slr28_archive), config)
        try:
            rendered, rj = _render_chunk(accepted, str(output_dir))
            rejected.extend(rj)
        finally:
            if _WORKER_ZF is not None:
                _WORKER_ZF.close()
    else:
        chunk_size = max(1, (len(accepted) + eff_workers - 1) // eff_workers)
        chunks = [accepted[i : i + chunk_size] for i in range(0, len(accepted), chunk_size)]
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=eff_workers,
            initializer=_init_worker,
            initargs=(str(slr28_archive), config),
        ) as pool:
            futs = [pool.submit(_render_chunk, ch, str(output_dir)) for ch in chunks]
            for fut in futs:
                part_ren, part_rej = fut.result()
                rendered.extend(part_ren)
                rejected.extend(part_rej)

    rendered.sort(key=lambda item: item["source_id"])
    rendered = _drop_duplicate_audio(rendered, output_dir, rejected, config)
    _write_splits(output_dir, rendered)
    write_jsonl(output_dir / "rejected.jsonl", rejected)

    audit_payload = {
        **catalog.audit,
        "archive_name": config["slr28"]["archive_name"],
        "archive_sha256": actual_slr_sha,
        "license": config["slr28"]["license"],
    }
    (output_dir / "slr28_split_audit.json").write_text(
        json.dumps(audit_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = inspect_slr28_manifest(output_dir, config, True)
    report["generator_git_sha"] = git_sha
    report["source_sha256"] = sha256_file(sources_path)
    report["slr28_archive_sha256"] = actual_slr_sha
    (output_dir / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_sha_index(output_dir)
    confirmed, _code = validate_slr28_manifest_dir(output_dir, config, True)
    return confirmed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="构建并校验 Phase 6 Track 3 SLR28 物理增强 manifest")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--sources", type=Path, default=None)
    parser.add_argument("--slr28-archive", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--git-sha", default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--check-audio", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_slr28_data_config(args.config)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.seed != config["seed"]:
        print(f"seed {args.seed} does not match config seed {config['seed']}", file=sys.stderr)
        return 2
    if args.validate_only:
        report, code = validate_slr28_manifest_dir(args.output_dir, config, args.check_audio)
        print(json.dumps({"status": report["status"], "exit_code": code, "problems": report["problems"][:10]}, ensure_ascii=False))
        return code
    if args.sources is None or not args.sources.is_file():
        print(f"missing sources {args.sources}", file=sys.stderr)
        return 2
    if args.slr28_archive is None or not args.slr28_archive.is_file():
        print(f"missing slr28 archive {args.slr28_archive}", file=sys.stderr)
        return 2
    git_sha = args.git_sha if args.git_sha is not None else _git_sha(Path.cwd())
    try:
        report = build_slr28_manifest(
            config,
            args.sources,
            args.slr28_archive,
            args.output_dir,
            git_sha,
            workers=args.workers,
        )
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "rows": report["counts"]["rows"],
                "rejected": report["counts"]["rejected"],
                "slr28_usage": report["slr28_usage_audit"],
            },
            ensure_ascii=False,
        )
    )
    return STATUS_CODE[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
