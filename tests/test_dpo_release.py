"""DPO release contract tests. They do not download Qwen or use a GPU."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from evaluation.zh_cer import score_pair
from train import sft_release
from train.dpo_contract import preference_accuracy
from train.dpo_release import (
    compare_dpo_greedy_texts,
    load_dpo_release,
    load_published_dpo_release,
    mark_dpo_published,
    physical_release_evidence,
    prepare_dpo_release,
)
from train.dpo_schema import load_dpo_config, semantic_dpo_config_sha256
from train.sft_checkpoint import atomic_write_json, file_map, sha256_file, utc_now, write_checksums, write_latest
from train.sft_release import greedy_fields

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "train" / "dpo_pilot_edit1_anchor_step40_h100.yaml"
RUN_NAME = "20261022T133007_dpo_06b_zh_seed42"
RELEASE_ID = "dpo_06b_zh_20261022T133007_step000040"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def scored_rows() -> list[dict]:
    return [
        {"sample_id": "a__clean", "prediction_raw": "甲", "stop_reason": "eos", "error": None},
        {"sample_id": "a__dropout", "prediction_raw": "乙", "stop_reason": "eos", "error": None},
    ]


def build_dpo_run(
    root: Path,
    *,
    passed: bool = True,
    pref_acc: float = 0.6440677966101694,
    latest_step: int = 40,
    config_path: Path = CONFIG,
) -> Path:
    config = load_dpo_config(config_path)
    manifest_sha = {
        "train": "a" * 64,
        "val": config["baseline"]["manifest_sha256"],
        "metadata": "c" * 64,
        "pair_metadata": "d" * 64,
        "filled_train_pairs": config["dpo"]["filled_train_pairs_sha256"],
        "filled_val_pairs": config["dpo"]["filled_val_pairs_sha256"],
        "reference_cache": config["dpo"]["reference_cache_sha256"],
    }
    run = root / RUN_NAME
    checkpoint = run / "checkpoints" / "step_000040"
    (checkpoint / "adapter").mkdir(parents=True)
    (checkpoint / "adapter" / "adapter.safetensors").write_bytes(b"selected-dpo-adapter")
    (checkpoint / "adapter" / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    state = {
        "schema_version": "qwen3-asr-training-state-v1",
        "stage": "dpo",
        "global_step": 40,
        "eval_decode_sha256": config["decode_eval_sha256"],
        "parent_release_sha256": config["model"]["parent_release_sha256"],
        "config_sha256": semantic_dpo_config_sha256(config, manifest_sha),
        "manifest_sha256": manifest_sha,
    }
    (checkpoint / "training_state.json").write_text(json.dumps(state), encoding="utf-8")
    (checkpoint / "resolved_config.yaml").write_text("profile: pilot\n", encoding="utf-8")
    write_checksums(
        checkpoint,
        [
            "adapter/adapter.safetensors",
            "adapter/adapter_config.json",
            "training_state.json",
            "resolved_config.yaml",
        ],
        utc_now(),
    )
    decision = {
        "step": 40,
        "passed": passed,
        "risk_stop": not passed,
        "preference_accuracy": pref_acc,
        "preference_ties": 0,
        "raw_policy_accuracy": 0.025423728813559324,
        "length_buckets": {"equal_length": {"pairs": 118, "raw_policy_accuracy": 0.025423728813559324}},
    }
    gate = {
        "schema_version": "qwen3-asr-gate-v1",
        "status": "passed" if passed else "failed",
        "publishable": False,
        "audit_disposition": "unlistened",
        "selected_step": 40 if passed else None,
        "decisions": [decision],
    }
    (run / "gate.json").write_text(json.dumps(gate), encoding="utf-8")
    write_jsonl(run / "eval_decisions.jsonl", [decision])
    write_jsonl(run / "eval" / "step_000040" / "scored.jsonl", scored_rows())
    (run / "eval" / "step_000040" / "preference.json").write_text(
        json.dumps({"accuracy": pref_acc, "pairs": 118, "wins": 76, "ties": 0}),
        encoding="utf-8",
    )
    write_latest(run / "latest.json", step=latest_step, checkpoint_sha256="e" * 64)
    return run


def validation_rows(noise_edits: list[int]) -> list[dict]:
    reference = "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉"
    rows = []
    for i, edits in enumerate(noise_edits):
        for name in ("clean", "noise", "reverb", "far_field", "dropout"):
            count = edits if name == "noise" else 0
            prediction = "某" * count + reference[count:]
            rows.append({"sample_id": f"v{i}__{name}", "source_utterance_id": f"v{i}", "scenario": name,
                         "prediction_raw": prediction, "stop_reason": "eos", "error": None,
                         **score_pair(reference, prediction, inference_error=False, stop_reason="eos")})
    return rows


def build_v2_release_inputs(root: Path):
    template = ROOT / "configs/train/dpo_track3b_e1_excl_lr5e6_b02_ga4.yaml"
    raw = yaml.safe_load(template.read_text())
    recipe = yaml.safe_load((ROOT / "configs/experiments/dpo_v2_sweep.yaml").read_text())
    for section, overrides in recipe["common"].items():
        raw[section].update(overrides)
    candidate = validation_rows([0] * 500)
    parent_rows = validation_rows([1] * 500)
    sft = root / "sft"
    for relative in sft_release.PAYLOAD_FILES:
        path = sft / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    saved = sft / "source/scored.jsonl"
    write_jsonl(saved, parent_rows)
    payload = sft_release._write_payload_sha(sft)
    raw["model"]["parent_release_sha256"] = payload["files"]["adapter/adapter.safetensors"]
    config_path = root / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(raw))
    cfg = load_dpo_config(config_path)
    parent_id = raw["model"]["parent_release"]
    atomic_write_json(sft / "release.json", {"schema_version": sft_release.RELEASE_SCHEMA,
                      "status": "staged", "publishable": False, "release_id": parent_id,
                      "adapter_sha256": cfg["model"]["parent_release_sha256"], "scored_sha256": sha256_file(saved),
                      "val_manifest_sha256": cfg["baseline"]["manifest_sha256"], "eval_decode_sha256": cfg["decode_eval_sha256"],
                      "payload_sha256": payload["directory_sha256"]})
    fields = greedy_fields(parent_rows, "fixture SFT")
    sft_release.mark_published(sft, dict(sft_release.compare_greedy_texts(fields, fields), release_id=parent_id))

    run = build_dpo_run(root, pref_acc=1.0, config_path=config_path)
    step = run / "eval/step_000040"
    zero = run / "eval/step_000000"
    write_jsonl(step / "scored.jsonl", candidate)
    write_jsonl(zero / "scored.jsonl", parent_rows)
    atomic_write_json(zero / "metrics.json", {"samples": len(parent_rows)})
    spec = json.loads(Path(cfg["dpo"]["val_cohort"]).read_text())

    def preference_block(ids, trained):
        margin = 1 if trained else -1
        return dict(preference_accuracy([2 if trained else 0] * len(ids)), raw_policy_accuracy=float(trained),
                    val_cohort_sha256=cfg["dpo"]["val_cohort_sha256"], max_abs_error=0,
                    pair_margins=[{"pair_id": pair_id, "policy_margin": margin, "reference_margin": -1} for pair_id in ids])

    for directory, number in ((zero, 0), (step, 40)):
        pref = preference_block(spec["primary"]["pair_ids"], number > 0)
        pref["supplemental"] = preference_block(spec["supplemental"]["pair_ids"], number > 0)
        atomic_write_json(directory / "preference.json", dict(pref, step=number))
    checkpoint = run / "checkpoints/step_000040"
    state_path = checkpoint / "training_state.json"
    state = json.loads(state_path.read_text())
    state["step0_sha256"] = file_map(zero, ["metrics.json", "scored.jsonl", "preference.json"])
    atomic_write_json(state_path, state)
    write_checksums(checkpoint, ["adapter/adapter.safetensors", "adapter/adapter_config.json",
                                 "training_state.json", "resolved_config.yaml"], utc_now())
    gate = json.loads((run / "gate.json").read_text())
    decision = gate["decisions"][0]
    decision.update(adapter_sha256=sha256_file(checkpoint / "adapter/adapter.safetensors"),
                    preference_sha256=sha256_file(step / "preference.json"), scored_sha256=sha256_file(step / "scored.jsonl"),
                    raw_policy_accuracy=1.0)
    atomic_write_json(run / "gate.json", gate)
    write_jsonl(run / "eval_decisions.jsonl", [decision])
    return dict(config_path=config_path, run_dir=run, output_dir=root / "out", release_id=RELEASE_ID, parent_release_dir=sft)



class DpoReleaseTest(unittest.TestCase):
    def test_prepare_copies_selected_dpo_adapter_and_syncs_drive(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = build_dpo_run(root)
            output = root / "releases" / RELEASE_ID
            drive = root / "drive_releases" / RELEASE_ID
            release = prepare_dpo_release(
                config_path=CONFIG,
                run_dir=run,
                output_dir=output,
                release_id=RELEASE_ID,
                sync_dir=drive,
            )
            self.assertEqual(release["status"], "staged")
            self.assertFalse(release["publishable"])
            self.assertEqual(release["stage"], "dpo")
            self.assertEqual(release["source_step"], 40)
            self.assertEqual(release["latest_step"], 40)
            self.assertTrue(release["latest_is_release"])
            self.assertEqual(release["parent_release"], "sft_06b_zh_20261015T234939_step000030")
            self.assertEqual(release["audit_disposition"], "unlistened")
            self.assertEqual((output / "adapter" / "adapter.safetensors").read_bytes(), b"selected-dpo-adapter")
            self.assertFalse((output / "optimizer.pt").exists())
            self.assertEqual(load_dpo_release(output), load_dpo_release(drive))
            with self.assertRaises(ValueError):
                load_published_dpo_release(output)
            with self.assertRaises(ValueError):
                prepare_dpo_release(
                    config_path=CONFIG,
                    run_dir=run,
                    output_dir=output,
                    release_id=RELEASE_ID,
                )
            (run / "eval" / "step_000040" / "preference.json").unlink()
            output_fallback = root / "releases_fallback" / RELEASE_ID
            fallback_rel = prepare_dpo_release(
                config_path=CONFIG,
                run_dir=run,
                output_dir=output_fallback,
                release_id=RELEASE_ID,
            )
            self.assertAlmostEqual(fallback_rel["preference_accuracy"], 0.6440677966101694)
            self.assertTrue((output_fallback / "source" / "preference.json").is_file())

    def test_risk_stop_or_low_preference_is_not_releasable(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_failed = build_dpo_run(root / "failed", passed=False)
            with self.assertRaises(ValueError):
                prepare_dpo_release(
                    config_path=CONFIG,
                    run_dir=run_failed,
                    output_dir=root / "out1",
                    release_id=RELEASE_ID,
                )
            run_low_pref = build_dpo_run(root / "low_pref", passed=True, pref_acc=0.5)
            with self.assertRaises(ValueError):
                prepare_dpo_release(
                    config_path=CONFIG,
                    run_dir=run_low_pref,
                    output_dir=root / "out2",
                    release_id=RELEASE_ID,
                )
            stopped = build_dpo_run(root / "stopped")
            path = stopped / "eval_decisions.jsonl"
            selected = json.loads(path.read_text())
            write_jsonl(path, [dict(selected, step=20, passed=False, risk_stop=True), selected])
            with self.assertRaisesRegex(ValueError, "after a recorded hard stop"):
                prepare_dpo_release(config_path=CONFIG, run_dir=stopped, output_dir=root / "out3", release_id=RELEASE_ID)

    def test_physical_gate_requires_edit_reduction_and_negative_cluster_interval(self) -> None:
        best = validation_rows([0] * 50)
        evidence = physical_release_evidence(validation_rows([1] * 20 + [0] * 30), best, 42)
        self.assertEqual(evidence["physical"]["edit_delta"], -20)
        self.assertLess(evidence["physical"]["cer_delta_ci95"][1], 0)
        with self.assertRaisesRegex(ValueError, "edit_delta=-19"):
            physical_release_evidence(validation_rows([1] * 19 + [0] * 31), best, 42)
        with self.assertRaisesRegex(ValueError, "edit_delta=-20, CI95 upper=0"):
            physical_release_evidence(validation_rows([20] + [0] * 49), best, 42)

    def test_v2_prepare_verifies_step0_and_selected_preference_before_staging(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = build_v2_release_inputs(root)
            self.assertEqual(prepare_dpo_release(**args), load_dpo_release(args["output_dir"]))
            evidence = json.loads((args["output_dir"] / "source/physical_validation.json").read_text())
            self.assertEqual(evidence["resamples"], 2000)
            self.assertEqual(evidence["physical_edit_delta_vs_step0"], -500)
            self.assertEqual(evidence["preference"]["primary"]["pairs"], 160)
            self.assertEqual(evidence["preference"]["supplemental"]["pairs"], 82)
            self.assertEqual(evidence["preference"]["primary"]["step0_raw_policy_accuracy"], 0.0)
            self.assertEqual(evidence["preference"]["primary"]["raw_policy_accuracy"], 1.0)
            args["output_dir"] = root / "rejected"
            run = args["run_dir"]
            zero = run / "eval/step_000000/metrics.json"
            original = zero.read_bytes()
            zero.write_text("{}")
            with self.assertRaisesRegex(ValueError, "step-0 evaluation"):
                prepare_dpo_release(**args)
            zero.write_bytes(original)
            step_pref = run / "eval/step_000040/preference.json"
            doc = json.loads(step_pref.read_text())
            doc["pair_margins"].reverse()
            atomic_write_json(step_pref, doc)
            with self.assertRaisesRegex(ValueError, "saved checkpoint and evaluation"):
                prepare_dpo_release(**args)
            gate = json.loads((run / "gate.json").read_text())
            gate["decisions"][0]["preference_sha256"] = sha256_file(step_pref)
            write_jsonl(run / "eval_decisions.jsonl", gate["decisions"])
            atomic_write_json(run / "gate.json", gate)
            with self.assertRaisesRegex(ValueError, "fixed val pairs"):
                prepare_dpo_release(**args)
            self.assertFalse(args["output_dir"].exists())

    def test_text_mismatch_stays_unpublished_and_matching_publishes(self) -> None:
        saved = greedy_fields(scored_rows(), "saved")
        bad_fresh = greedy_fields(
            [
                {"sample_id": "a__clean", "prediction_raw": "甲", "stop_reason": "eos", "error": None},
                {"sample_id": "a__dropout", "prediction_raw": "丙", "stop_reason": "eos", "error": None},
            ],
            "fresh",
        )
        bad_report = compare_dpo_greedy_texts(saved, bad_fresh)
        self.assertFalse(bad_report["matched"])
        good_report = compare_dpo_greedy_texts(saved, saved)
        good_report["release_id"] = RELEASE_ID
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "releases" / RELEASE_ID
            prepare_dpo_release(
                config_path=CONFIG,
                run_dir=build_dpo_run(root),
                output_dir=output,
                release_id=RELEASE_ID,
            )
            with self.assertRaises(ValueError):
                mark_dpo_published(output, bad_report)
            before = (output / "adapter" / "adapter.safetensors").read_bytes()
            published = mark_dpo_published(output, good_report)
            self.assertEqual(published["status"], "published")
            self.assertTrue(published["publishable"])
            self.assertEqual(published["audit_disposition"], "unlistened")
            self.assertEqual((output / "adapter" / "adapter.safetensors").read_bytes(), before)
            loaded = load_published_dpo_release(output)
            self.assertEqual(loaded["release_id"], RELEASE_ID)


if __name__ == "__main__":
    unittest.main()
