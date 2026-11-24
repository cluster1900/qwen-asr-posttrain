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


if __name__ == "__main__":
    unittest.main()
