"""Offline tests for the materialized DPO v2 input contracts and four-way sweep."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from evaluation.jsonl_io import write_jsonl
from scripts import prepare_dpo_v2_sweep as sweep
from scripts.replay_dpo_gate import make_dpo_v2_gate
from train.dpo_schema import load_dpo_config
from train.dpo_v2 import PHYSICAL, check_reference_context, check_step0_score, cohort_digest, fixed_val_cohorts, physical_clean_map
from train.reference_logprob import cache_reference_logprobs, publish_logprob_pairs
from train.sft_contract import iterate_updates
from train.sft_schema import sha256_file
from train.train_dpo import prepare_pair_sets
from train.train_sft import RunStop

ROOT = Path(__file__).resolve().parents[1]


def make_pair(split: str, index: int, scenario: str) -> dict:
    source = f"{split}-{index}"
    sid = f"{source}__{scenario}"
    return {
        "schema_version": "qwen3-asr-dpo-pair-v1", "pair_id": sid, "sample_id": sid,
        "split": split, "scenario": scenario, "source_utterance_id": source, "audio": f"{sid}.wav",
        "reference_text": "甲乙", "chosen_text": "甲乙", "rejected_text": "甲丙", "char_length_delta": 0,
        "chosen_token_count": 3, "rejected_token_count": 3, "ref_chosen_logp": None, "ref_rejected_logp": None,
        **({"clean_prediction_normalized": "甲乙"} if split == "train" and scenario in PHYSICAL else {}),
    }


def cohort_spec(primary: list, supplemental: list) -> dict:
    def members(rows: list) -> dict:
        return {"pair_ids": [r["pair_id"] for r in rows], "members_sha256": cohort_digest(rows)}
    return {"schema_version": "qwen3-asr-dpo-val-cohorts-v1", "primary": members(primary),
            "supplemental": members(supplemental)}


def sweep_inputs(root: Path, *, pool_size: int = 8000, anchor: bool = True):
    project = root / "recipe-project"
    manifest, null, filled = (root / name for name in ("manifest", "null", "filled"))
    for directory in (manifest, null, filled, project / "configs/experiments"):
        directory.mkdir(parents=True)
    train = [make_pair("train", i, PHYSICAL[i % 3]) for i in range(pool_size)]
    if anchor:
        train.append(make_pair("train", pool_size, "dropout"))
    val = [make_pair("val", i, ("clean", *PHYSICAL)[i % 4]) for i in range(242)]
    metadata = {"status": "pairs_built"}
    (manifest / "manifest.json").write_text("{}")
    for name, rows in (("train", train), ("val", val)):
        path = manifest / f"{name}.jsonl"
        write_jsonl(path, [{"sample_id": r["sample_id"], "text": r["reference_text"], "language": "zh",
                            **{k: r[k] for k in ("scenario", "source_utterance_id", "audio", "split")}} for r in rows])
        write_jsonl(null / f"{name}_pairs.jsonl", rows)
        metadata[f"{name}_manifest_sha256"] = sha256_file(path)
        metadata[f"{name}_pairs_sha256"] = sha256_file(null / f"{name}_pairs.jsonl")
    recipe = yaml.safe_load((ROOT / "configs/experiments/dpo_v2_sweep.yaml").read_text())
    template = yaml.safe_load((ROOT / recipe["template"]).read_text())
    baseline = json.loads((ROOT / template["baseline"]["metrics"]).read_text())
    baseline["manifest_sha256"] = metadata["val_manifest_sha256"]
    (root / "baseline.json").write_text(json.dumps(baseline))
    template["baseline"]["metrics"] = str(root / "baseline.json")
    (root / "template.yaml").write_text(yaml.safe_dump(template))
    recipe["template"] = str(root / "template.yaml")
    (root / "cohort.json").write_text(json.dumps(cohort_spec(val[:160], val[160:])))
    recipe["common"]["dpo"]["val_cohort"] = str(root / "cohort.json")
    recipe["common"]["dpo"]["val_cohort_sha256"] = sha256_file(root / "cohort.json")
    (project / "configs/experiments/dpo_v2_sweep.yaml").write_text(yaml.safe_dump(recipe))
    model = template["model"]
    metadata.update(adapter_sha256=model["parent_release_sha256"], release_id=model["parent_release"])
    meta = null / "pair_metadata.json"
    meta.write_text(json.dumps(metadata))
    cache = filled / "ref_logprobs.jsonl"
    def synthetic_score(row, text):
        return (-3.0 if text == row["chosen_text"] else -1.0), 3
    report = cache_reference_logprobs(train, val, synthetic_score, cache, seed=42)
    report.update(release_id=model["parent_release"], reference_context={
        "lora_dtype": "float32", "gpu_name": "fixture-GPU", "decode_eval_sha256": baseline["decode_eval_sha256"],
        "adapter_sha256": model["parent_release_sha256"], "null_train_pairs_sha256": metadata["train_pairs_sha256"],
        "null_val_pairs_sha256": metadata["val_pairs_sha256"],
    })
    publish_logprob_pairs(train, val, cache, filled, report)
    return project, manifest, meta, filled


class DpoV2InputTest(unittest.TestCase):
    def test_fixed_cohorts_survive_reordering_and_reject_missing_changed_or_overlapping_pairs(self):
        first, second = (make_pair("val", i, "noise") for i in range(2))
        spec = cohort_spec([first], [second])
        self.assertEqual(fixed_val_cohorts([second, first], spec), ([first], [second]))
        for rows, invalid in (([first], spec), ([first, dict(second, rejected_text="甲丁")], spec),
                              ([first, second], cohort_spec([first], [first]))):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                fixed_val_cohorts(rows, invalid)

    def test_clean_control_and_absolute_error(self):
        row = make_pair("train", 1, "noise")
        self.assertEqual(physical_clean_map([row]), {"train-1": "甲乙"})
        for invalid in (None, "", "甲乙。"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "missing clean"):
                physical_clean_map([dict(row, clean_prediction_normalized=invalid)])
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            physical_clean_map([row, dict(row, clean_prediction_normalized="甲丁")])
        self.assertEqual(check_step0_score(0.0001, 0.0, "chosen"), 0.0001)
        for invalid in (-1.0002, float("nan"), float("inf")):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "step-0"):
                check_step0_score(invalid, -1.0, "rejected")

    def test_materialized_sweep_generates_equal_budgets_and_refuses_cache_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project, manifest, meta, filled = sweep_inputs(root)
            out = root / "out"
            with patch.object(sweep, "ROOT", project):
                sweep.prepare(manifest, meta, filled, out)
            configs = [load_dpo_config(path) for path in sorted(out.glob("*.yaml"))]
            self.assertEqual(len(configs), 4)
            arms = set()
            small_ids = set()
            large_ids = set()
            for cfg in configs:
                sets = prepare_pair_sets(cfg)
                self.assertEqual(cfg["gate"], make_dpo_v2_gate(min_stop_step=100))
                arms.add((len(sets.train), cfg["optim"]["learning_rate"]))
                self.assertEqual((sets.pool_size, len(sets.primary), len(sets.supplemental), len(sets.anchors)),
                                 (8000, 160, 82, 1))
                self.assertEqual(cfg["dpo"]["pair_metadata_sha256"], sha256_file(meta))
                updates = list(iterate_updates(sets.train, seed=cfg["seed"],
                                               accumulation=cfg["batch"]["gradient_accumulation_steps"],
                                               max_steps=cfg["train"]["max_steps"], max_epochs=cfg["train"]["max_epochs"]))
                self.assertEqual((len(updates), sum(len(u["rows"]) for u in updates)), (500, 8000))
                ids = [r["pair_id"] for u in updates for r in u["rows"]]
                if len(sets.train) == 256:
                    small_ids.add(tuple(ids))
                    self.assertEqual(len(set(ids)), 256)
                else:
                    large_ids.add(tuple(ids))
                    self.assertEqual(len(set(ids)), 8000)
            self.assertEqual(len(small_ids), 1)
            self.assertEqual(len(large_ids), 1)
            self.assertEqual(arms, {(n, lr) for n in (256, 8000) for lr in (5e-6, 1e-5)})

            report_path = filled / "ref_report.json"
            report = json.loads(report_path.read_text())
            for field in report["reference_context"]:
                bad = copy.deepcopy(report)
                bad["reference_context"][field] = "wrong"
                with self.subTest(field=field), self.assertRaises(ValueError):
                    check_reference_context(bad, cfg, "fixture-GPU")
            with self.assertRaisesRegex(ValueError, "filled train"):
                check_reference_context(dict(report, train_pairs_sha256="0"*64), cfg, None)
            del report["reference_context"]
            report_path.write_text(json.dumps(report))
            rejected = root / "rejected"
            with patch.object(sweep, "ROOT", project), self.assertRaisesRegex(ValueError, "GPU identity"):
                sweep.prepare(manifest, meta, filled, rejected)
            self.assertFalse(rejected.exists())

    def test_incomplete_pool_and_missing_anchor_do_not_emit_recipes(self):
        # Smaller fixtures reach each failure by adjusting only the synthetic acceptance floor.
        for pool_size, anchor, expected in ((480, False, "anchor"), (479, True, "floor")):
            with self.subTest(case=expected), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                project, manifest, meta, filled = sweep_inputs(root, pool_size=pool_size, anchor=anchor)
                path = project / "configs/experiments/dpo_v2_sweep.yaml"
                recipe = yaml.safe_load(path.read_text())
                recipe["common"]["dpo"]["min_train_pairs"] = 480
                path.write_text(yaml.safe_dump(recipe))
                with patch.object(sweep, "ROOT", project), self.assertRaisesRegex(RunStop, expected):
                    sweep.prepare(manifest, meta, filled, root / "out")
                self.assertFalse((root / "out").exists())

    def test_prepare_sweep_supports_flexible_ga_rehearse_only_and_rejects_underfilled_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project, manifest, meta, filled = sweep_inputs(root, pool_size=8000)
            out_rehearse = root / "rehearse_only"
            with patch.object(sweep, "ROOT", project):
                summary = sweep.prepare(manifest, meta, filled, out_rehearse, ga=20, rehearse_only=True)
            self.assertFalse(out_rehearse.exists())
            self.assertEqual((summary["ga"], summary["actual_steps"], summary["pairs_seen"]), (20, 400, 8000))

            # Flexible --ga 10 auto-scales 1 epoch on 8000 pairs -> 800 steps, warmup 80, eval 160
            out_ga10 = root / "ga10"
            with patch.object(sweep, "ROOT", project):
                sweep.prepare(manifest, meta, filled, out_ga10, ga=10)
            cfgs = [load_dpo_config(p) for p in sorted(out_ga10.glob("*.yaml"))]
            self.assertEqual(len(cfgs), 4)
            for cfg in cfgs:
                self.assertEqual(cfg["batch"]["gradient_accumulation_steps"], 10)
                self.assertEqual(cfg["train"]["max_steps"], 800)
                self.assertEqual(cfg["optim"]["warmup_steps"], 80)
                self.assertEqual(cfg["train"]["eval_steps"], 160)
                self.assertEqual(cfg["gate"]["min_stop_step"], 160)

            # Requesting max_steps * ga > large pool_size must fail closed
            with patch.object(sweep, "ROOT", project), self.assertRaisesRegex(ValueError, "large pool has"):
                sweep.prepare(manifest, meta, filled, root / "too_big", ga=32, max_steps=500)
            self.assertFalse((root / "too_big").exists())


if __name__ == "__main__":
    unittest.main()
