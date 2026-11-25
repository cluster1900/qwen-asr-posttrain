from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from train.dpo_schema import load_dpo_config, semantic_dpo_config_sha256
from train.rl_schema import load_rl_config, semantic_rl_config_sha256
from train.sft_schema import load_train_config, resolve_environment_lock_path, semantic_config_sha256, sha256_file


class Track3ScaleContractsTest(unittest.TestCase):
    def test_sft_scale_and_train_scenarios_preserve_legacy_pilot_hash(self) -> None:
        legacy_path = Path("configs/train/sft_pilot_lr1e5.yaml")
        legacy = load_train_config(legacy_path)
        self.assertNotIn("train_scenarios", legacy["data"])
        msha = {"train": "a" * 64, "val": "b" * 64, "metadata": "c" * 64}
        legacy_hash = semantic_config_sha256(legacy, msha)

        raw = yaml.safe_load(legacy_path.read_text(encoding="utf-8"))
        raw["profile"] = "scale"
        raw["data"]["train_scenarios"] = ["clean", "noise", "reverb", "far_field"]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "sft_scale.yaml"
            p.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
            scaled = load_train_config(p)
            self.assertEqual(scaled["profile"], "scale")
            self.assertEqual(scaled["data"]["train_scenarios"], ["clean", "noise", "reverb", "far_field"])
            self.assertNotEqual(semantic_config_sha256(scaled, msha), legacy_hash)

    def test_dpo_scale_accepts_track3_sft_parent_release(self) -> None:
        raw = yaml.safe_load(Path("configs/train/dpo_pilot_edit1_anchor_step40_h100.yaml").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent_metrics = json.loads(Path("configs/train/aishell_val_sft_step30_20261016.json").read_text(encoding="utf-8"))
            parent_metrics["release_id"] = "sft_06b_zh_20261124T120000_step000080"
            parent_metrics["source_step"] = 80
            pm_path = root / "sft_parent.json"
            pm_path.write_text(json.dumps(parent_metrics, indent=2), encoding="utf-8")

            raw["profile"] = "scale"
            raw["model"]["parent_release"] = "sft_06b_zh_20261124T120000_step000080"
            raw["dpo"]["parent_metrics"] = str(pm_path)
            raw["optim"]["learning_rate"] = 0.000008
            raw["train"]["max_steps"] = 80
            raw["train"]["keep_last"] = 8
            raw["optim"]["scheduler_max_steps"] = 120
            cfg_p = root / "dpo_scale.yaml"
            cfg_p.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
            resolved = load_dpo_config(cfg_p)
            self.assertEqual(resolved["profile"], "scale")
            self.assertEqual(resolved["dpo"]["parent"]["source_step"], 80)
            msha = {"train": "a" * 64, "val": "b" * 64, "metadata": "c" * 64}
            self.assertEqual(len(semantic_dpo_config_sha256(resolved, msha)), 64)

    def test_rl_scale_accepts_track3_dpo_parent_release(self) -> None:
        raw = yaml.safe_load(Path("configs/train/rl_pilot_aligned.yaml").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent_metrics = json.loads(Path("configs/train/aishell_val_dpo_step40_20261022.json").read_text(encoding="utf-8"))
            parent_metrics["release_id"] = "dpo_06b_zh_20261124T150000_step000060"
            parent_metrics["source_step"] = 60
            pm_path = root / "dpo_parent.json"
            pm_path.write_text(json.dumps(parent_metrics, indent=2), encoding="utf-8")

            raw["profile"] = "scale"
            raw["model"]["parent_release"] = "dpo_06b_zh_20261124T150000_step000060"
            raw["rl"]["parent_metrics"] = str(pm_path)
            raw["optim"]["learning_rate"] = 0.000002
            raw["rl"]["kl_beta"] = 0.06
            raw["train"]["max_steps"] = 30
            raw["train"]["keep_last"] = 6
            cfg_p = root / "rl_scale.yaml"
            cfg_p.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
            resolved = load_rl_config(cfg_p)
            self.assertEqual(resolved["profile"], "scale")
            self.assertEqual(resolved["rl"]["parent"]["source_step"], 60)
            msha = {"train": "a" * 64, "val": "b" * 64, "metadata": "c" * 64}
            self.assertEqual(len(semantic_rl_config_sha256(resolved, msha)), 64)

    def test_environment_lock_file_optional_and_track3_smoke_matches(self) -> None:
        root = Path(__file__).resolve().parents[1]
        legacy = load_train_config(root / "configs/train/sft_pilot_lr1e5.yaml")
        self.assertNotIn("environment_lock_file", legacy["runtime"])
        self.assertEqual(resolve_environment_lock_path(legacy, root), (root / "requirements.lock").resolve())

        t3_smoke = load_train_config(root / "configs/train/sft_track3_smoke.yaml")
        self.assertEqual(t3_smoke["runtime"]["environment_lock_file"], "requirements.track3.lock")
        lock_path = resolve_environment_lock_path(t3_smoke, root)
        self.assertEqual(lock_path, (root / "requirements.track3.lock").resolve())
        self.assertEqual(sha256_file(lock_path), t3_smoke["runtime"]["environment_lock_sha256"])

        raw = yaml.safe_load((root / "configs/train/sft_track3_smoke.yaml").read_text(encoding="utf-8"))
        raw["runtime"]["environment_lock_file"] = "../escape.lock"
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.yaml"
            bad.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_train_config(bad)

    def test_build_dpo_pairs_align_punctuation_and_dpo_allow_matched_punctuation(self) -> None:
        from scripts.build_dpo_pairs import char_token_count, classify_row
        from train.dpo_contract import select_training_pairs, validate_pair_rows

        manifest_row = {
            "sample_id": "s1__far_field",
            "split": "train",
            "scenario": "far_field",
            "source_utterance_id": "s1",
            "audio": "wavs/s1.wav",
            "text": "在工业建筑领域建设新方案",
        }
        pred_row = {
            "sample_id": "s1__far_field",
            "reference_raw": "在工业建筑领域建设新方案",
            "prediction_raw": "在工业、建筑领域，借助新方案。",
            "stop_reason": "eos",
            "error": "",
        }
        legacy_rec = classify_row(
            manifest_row,
            pred_row,
            split="train",
            count_tokens=char_token_count,
            max_transcript_tokens=64,
        )
        self.assertEqual(legacy_rec["chosen_text"], "在工业建筑领域建设新方案")

        punct_rec = classify_row(
            manifest_row,
            pred_row,
            split="train",
            count_tokens=char_token_count,
            max_transcript_tokens=64,
            align_punctuation_from_rejected=True,
        )
        self.assertEqual(punct_rec["bucket"], "pair")
        self.assertEqual(punct_rec["reference_text"], "在工业建筑领域建设新方案")
        self.assertEqual(punct_rec["chosen_text"], "在工业、建筑领域，建设新方案。")
        self.assertEqual(punct_rec["chosen_token_count"], punct_rec["rejected_token_count"])
        punct_rec["ref_chosen_logp"] = -2.0
        punct_rec["ref_rejected_logp"] = -1.5

        with self.assertRaises(ValueError):
            validate_pair_rows([punct_rec], split="train", min_per_scenario=1, required_scenarios=("far_field",))
        counts = validate_pair_rows(
            [punct_rec],
            split="train",
            min_per_scenario=1,
            required_scenarios=("far_field",),
            allow_matched_punctuation=True,
        )
        self.assertEqual(counts["far_field"], 1)

        # Legacy strict filter rejects punctuated pairs; allow_matched_punctuation accepts identical punctuation.
        self.assertEqual(
            select_training_pairs([punct_rec], ["far_field"], aligned_substitution_only=True, max_edit_count=2),
            [],
        )
        kept = select_training_pairs(
            [punct_rec],
            ["far_field"],
            aligned_substitution_only=True,
            allow_matched_punctuation=True,
            max_edit_count=2,
        )
        self.assertEqual(len(kept), 1)

        # Mismatched punctuation character or position must still be rejected.
        bad_punct = dict(punct_rec, pair_id="s1__bad", chosen_text="在工业，建筑领域，建设新方案。")
        self.assertEqual(
            select_training_pairs(
                [bad_punct],
                ["far_field"],
                aligned_substitution_only=True,
                allow_matched_punctuation=True,
                max_edit_count=2,
            ),
            [],
        )

    def test_rl_allow_matched_punctuation_and_schema_hashes(self) -> None:
        from train.rl_contract import aligned_group_mismatch_mask

        cands = [
            {"prediction_raw": "你好，世界。", "reward": 1.0},
            {"prediction_raw": "你好，世界。", "reward": 1.0},
            {"prediction_raw": "你好，世借。", "reward": 0.75},
            {"prediction_raw": "你好。世借。", "reward": 0.75},
        ]
        tok_ids = [
            [1, 2, 10, 3, 4, 11, 99],
            [1, 2, 10, 3, 4, 11, 99],
            [1, 2, 10, 3, 5, 11, 99],
            [1, 2, 12, 3, 5, 11, 99],
        ]
        legacy = aligned_group_mismatch_mask(cands, tok_ids, max_edit_count=2)
        self.assertFalse(legacy["aligned_valid"])

        matched = aligned_group_mismatch_mask(
            cands, tok_ids, max_edit_count=2, allow_matched_punctuation=True
        )
        self.assertTrue(matched["aligned_valid"])
        self.assertEqual(matched["mismatch_index"], 4)
        # Candidate 3 has a punctuation mismatch ('。' vs '，') so it is excluded; only [0, 1, 2] align.
        self.assertEqual(matched["aligned_indices"], [0, 1, 2])
        self.assertEqual(matched["token_mask"], [False, False, False, False, True, False, False])

        # Verify legacy DPO & RL pilot config hashes remain bit-exact when allow_matched_punctuation is omitted.
        msha = {"train": "a" * 64, "val": "b" * 64, "metadata": "c" * 64}
        dpo_legacy = load_dpo_config(Path("configs/train/dpo_pilot_edit1_anchor_step40_h100.yaml"))
        self.assertFalse(dpo_legacy["dpo"]["allow_matched_punctuation"])
        dpo_h1 = semantic_dpo_config_sha256(dpo_legacy, msha)
        dpo_mod = copy.deepcopy(dpo_legacy)
        dpo_mod["dpo"]["allow_matched_punctuation"] = True
        self.assertNotEqual(semantic_dpo_config_sha256(dpo_mod, msha), dpo_h1)

        rl_legacy = load_rl_config(Path("configs/train/rl_pilot_aligned.yaml"))
        self.assertFalse(rl_legacy["rl"]["allow_matched_punctuation"])
        rl_h1 = semantic_rl_config_sha256(rl_legacy, msha)
        rl_mod = copy.deepcopy(rl_legacy)
        rl_mod["rl"]["allow_matched_punctuation"] = True
        self.assertNotEqual(semantic_rl_config_sha256(rl_mod, msha), rl_h1)

        # Track 3 RL smoke with explicit environment_lock_file accepts any published DPO parent and interrupt_after_step.
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            parent_p = tdir / "parent_dpo.json"
            parent_p.write_text(
                json.dumps(
                    {
                        "schema_version": "qwen3-asr-parent-metrics-v1",
                        "split": "val",
                        "parent": "dpo",
                        "release_id": "dpo_06b_zh_20261125T005000_step000040",
                        "source_step": 40,
                        "samples": 2500,
                        "micro_cer": 0.060,
                        "by_scenario": {
                            s: {"samples": 500, "micro_cer": 0.05}
                            for s in ("clean", "noise", "reverb", "far_field", "dropout")
                        },
                    }
                ),
                encoding="utf-8",
            )
            smoke_raw = yaml.safe_load(Path("configs/train/rl_pilot_aligned.yaml").read_text(encoding="utf-8"))
            smoke_raw["profile"] = "smoke"
            smoke_raw["model"]["parent_release"] = "dpo_06b_zh_20261125T005000_step000040"
            smoke_raw["runtime"]["environment_lock_file"] = "requirements.track3.lock"
            smoke_raw["runtime"]["environment_lock_sha256"] = "b4b5fc873108ea373787af85f2bd9ed0697bce0529a5d345b7a1761653b32432"
            smoke_raw["runtime"]["max_runtime_minutes"] = 30
            smoke_raw["runtime"]["shutdown_margin_minutes"] = 5
            smoke_raw["train"]["max_steps"] = 6
            smoke_raw["train"]["save_steps"] = 5
            smoke_raw["train"]["eval_steps"] = 6
            smoke_raw["train"]["interrupt_after_step"] = 5
            smoke_raw["optim"]["warmup_steps"] = 2
            smoke_raw["optim"]["scheduler_max_steps"] = None
            smoke_raw["rl"]["allow_matched_punctuation"] = True
            smoke_raw["rl"]["parent_metrics"] = str(parent_p)
            smoke_p = tdir / "rl_track3_smoke.yaml"
            smoke_p.write_text(yaml.safe_dump(smoke_raw, sort_keys=False), encoding="utf-8")
            parsed_smoke = load_rl_config(smoke_p)
            self.assertEqual(parsed_smoke["profile"], "smoke")
            self.assertEqual(parsed_smoke["train"]["interrupt_after_step"], 5)
            self.assertTrue(parsed_smoke["rl"]["allow_matched_punctuation"])
            h_smoke = semantic_rl_config_sha256(parsed_smoke, msha)
            parsed_smoke2 = copy.deepcopy(parsed_smoke)
            parsed_smoke2["rl"]["allow_matched_punctuation"] = False
            self.assertNotEqual(semantic_rl_config_sha256(parsed_smoke2, msha), h_smoke)

            # Verify optional lora.trainable_dtype changes hash only when set and rejects bad values.
            smoke_fp32_raw = copy.deepcopy(smoke_raw)
            smoke_fp32_raw["lora"]["trainable_dtype"] = "float32"
            smoke_fp32_p = tdir / "rl_track3_smoke_fp32.yaml"
            smoke_fp32_p.write_text(yaml.safe_dump(smoke_fp32_raw, sort_keys=False), encoding="utf-8")
            parsed_fp32 = load_rl_config(smoke_fp32_p)
            self.assertEqual(parsed_fp32["lora"]["trainable_dtype"], "float32")
            self.assertNotEqual(semantic_rl_config_sha256(parsed_fp32, msha), h_smoke)
            self.assertNotIn("trainable_dtype", rl_legacy["lora"])

            bad_dt_raw = copy.deepcopy(smoke_raw)
            bad_dt_raw["lora"]["trainable_dtype"] = "float16"
            bad_dt_p = tdir / "rl_track3_bad_dt.yaml"
            bad_dt_p.write_text(yaml.safe_dump(bad_dt_raw, sort_keys=False), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_rl_config(bad_dt_p)


if __name__ == "__main__":
    unittest.main()
