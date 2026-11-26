"""Unit & contract tests for DPO v2 decoupled anti-lockup gate and offline replay CLI."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.replay_dpo_gate import (
    BASE_LEGACY_GATE,
    make_dpo_v2_gate,
    main as replay_main,
    metrics_from_scored_rows,
    replay_run_directory,
    run_synthetic_stress_suite,
    verify_warmup_schedules,
)
from train.dpo_contract import judge_dpo_checkpoint, resume_gate_action, trailing_non_passing_evals
from train.dpo_release import require_passed_dpo_decision
from train.dpo_schema import load_dpo_config, semantic_dpo_config_sha256
from train.rl_contract import judge_rl_checkpoint
from train.rl_schema import load_rl_config, semantic_rl_config_sha256
from train.sft_contract import DEGRADED_SCENARIOS, SCENARIOS, degraded_macro_cer, judge_sft_checkpoint
from train.sft_schema import load_train_config, semantic_config_sha256

ROOT = Path(__file__).resolve().parents[1]
FAKE_MANIFEST_SHA = {
    "train": "a" * 64,
    "val": "b" * 64,
    "metadata": "c" * 64,
}


def _make_metrics(
    cers: dict[str, float],
    *,
    empty_rate: float = 0.0,
    repeated_rate: float = 0.0,
) -> dict:
    by_scenario = {}
    for name in SCENARIOS:
        by_scenario[name] = {
            "samples": 500,
            "micro_cer": cers[name],
            "empty_rate": empty_rate if name == "noise" else 0.0,
            "repeated_rate": repeated_rate if name == "noise" else 0.0,
            "long_rate": 0.0,
            "truncation_rate": 0.0,
            "valid_output_rate": 1.0 - (empty_rate if name == "noise" else 0.0),
        }
    return {
        "samples": 2500,
        "micro_cer": sum(cers.values()) / len(cers),
        "empty_rate": empty_rate / len(SCENARIOS),
        "repeated_rate": repeated_rate / len(SCENARIOS),
        "long_rate": 0.0,
        "truncation_rate": 0.0,
        "valid_output_rate": 1.0 - (empty_rate / len(SCENARIOS)),
        "by_scenario": by_scenario,
    }


def _make_pref(accuracy: float, pairs: int = 160) -> dict:
    return {
        "accuracy": accuracy,
        "pairs": pairs,
        "ties": 0,
        "raw_policy_accuracy": 0.1,
        "length_buckets": {"equal_length": {"pairs": pairs, "raw_policy_accuracy": 0.1}},
    }


class TestDpoV2GateSchemaAndBackwardCompat(unittest.TestCase):
    def _write_yaml(self, tmp: Path, payload: dict) -> Path:
        path = tmp / "cfg.yaml"
        path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return path

    def test_legacy_configs_have_zero_gate_schema_or_hash_drift(self) -> None:
        legacy_keys = set(BASE_LEGACY_GATE.keys())
        sft_cfg = load_train_config(ROOT / "configs/train/sft_pilot_lr1e5.yaml")
        self.assertEqual(set(sft_cfg["gate"].keys()), legacy_keys)

        dpo_path = ROOT / "configs/train/dpo_track3b_e1_excl_lr5e6_b02_ga4.yaml"
        dpo_cfg = load_dpo_config(dpo_path)
        self.assertEqual(set(dpo_cfg["gate"].keys()), legacy_keys)
        base_dpo_hash = semantic_dpo_config_sha256(dpo_cfg, FAKE_MANIFEST_SHA)

        # Explicitly writing legacy default values must produce identical semantic hash
        raw = yaml.safe_load(dpo_path.read_text(encoding="utf-8"))
        raw["gate"]["stop_on_low_preference"] = True
        raw["gate"]["min_stop_step"] = 0
        with tempfile.TemporaryDirectory() as tmpdir:
            dup_path = self._write_yaml(Path(tmpdir), raw)
            dup_cfg = load_dpo_config(dup_path)
            self.assertEqual(set(dup_cfg["gate"].keys()), legacy_keys)
            self.assertEqual(semantic_dpo_config_sha256(dup_cfg, FAKE_MANIFEST_SHA), base_dpo_hash)

        rl_cfg = load_rl_config(ROOT / "configs/train/rl_pilot_aligned.yaml")
        self.assertEqual(set(rl_cfg["gate"].keys()), legacy_keys)
        self.assertEqual(len(semantic_rl_config_sha256(rl_cfg, FAKE_MANIFEST_SHA)), 64)

    def test_v2_optional_gate_fields_parse_and_change_semantic_hash(self) -> None:
        dpo_path = ROOT / "configs/train/dpo_track3b_e1_excl_lr5e6_b02_ga4.yaml"
        legacy_cfg = load_dpo_config(dpo_path)
        legacy_hash = semantic_dpo_config_sha256(legacy_cfg, FAKE_MANIFEST_SHA)

        raw = yaml.safe_load(dpo_path.read_text(encoding="utf-8"))
        raw["lora"]["trainable_dtype"] = "float32"
        raw["gate"]["degraded_macro_scenarios"] = ["far_field", "noise", "reverb"]
        raw["gate"]["degraded_macro_stop_increase_max"] = 0.002
        raw["gate"]["dropout_cer_increase_max"] = 0.002
        raw["gate"]["max_gap_insertions"] = 5
        raw["gate"]["stop_on_low_preference"] = False
        raw["gate"]["min_stop_step"] = 40
        raw["gate"]["max_consecutive_non_passing_evals"] = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            v2_path = self._write_yaml(Path(tmpdir), raw)
            v2_cfg = load_dpo_config(v2_path)
            self.assertEqual(v2_cfg["lora"]["trainable_dtype"], "float32")
            self.assertEqual(v2_cfg["gate"]["degraded_macro_scenarios"], ["far_field", "noise", "reverb"])
            self.assertEqual(v2_cfg["gate"]["degraded_macro_stop_increase_max"], 0.002)
            self.assertEqual(v2_cfg["gate"]["dropout_cer_increase_max"], 0.002)
            self.assertEqual(v2_cfg["gate"]["max_gap_insertions"], 5)
            self.assertIs(v2_cfg["gate"]["stop_on_low_preference"], False)
            self.assertEqual(v2_cfg["gate"]["min_stop_step"], 40)
            self.assertEqual(v2_cfg["gate"]["max_consecutive_non_passing_evals"], 2)
            self.assertNotEqual(semantic_dpo_config_sha256(v2_cfg, FAKE_MANIFEST_SHA), legacy_hash)

    def test_v2_gate_schema_rejects_invalid_values(self) -> None:
        base_raw = yaml.safe_load((ROOT / "configs/train/sft_pilot_lr1e5.yaml").read_text(encoding="utf-8"))
        invalid_mutations = [
            {"degraded_macro_scenarios": []},
            {"degraded_macro_scenarios": ["clean", "noise"]},
            {"degraded_macro_scenarios": ["noise", "noise"]},
            {"degraded_macro_scenarios": ["noise"], "min_improved_degraded_scenarios": 2},
            {"degraded_macro_stop_increase_max": -0.001},
            {"dropout_cer_increase_max": -0.001},
            {"max_gap_insertions": -1},
            {"stop_on_low_preference": "no"},
            {"min_stop_step": -5},
            {"max_consecutive_non_passing_evals": 0},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            for idx, mutation in enumerate(invalid_mutations):
                raw = copy.deepcopy(base_raw)
                raw["gate"].update(mutation)
                path = Path(tmpdir) / f"bad_{idx}.yaml"
                path.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
                with self.assertRaises(ValueError, msg=f"mutation {mutation} should fail"):
                    load_train_config(path)
                # Ensure semantic_config_sha256 remains clean on base
            self.assertEqual(len(semantic_config_sha256(load_train_config(ROOT / "configs/train/sft_pilot_lr1e5.yaml"), FAKE_MANIFEST_SHA)), 64)


class TestDpoV2GateJudgment(unittest.TestCase):
    def setUp(self) -> None:
        self.parent_cers = {
            "clean": 0.020,
            "noise": 0.030,
            "reverb": 0.025,
            "far_field": 0.047,
            "dropout": 0.185,
        }
        self.base_cers = {k: v + 0.002 for k, v in self.parent_cers.items()}
        self.parent = {"by_scenario": {k: {"samples": 500, "micro_cer": v} for k, v in self.parent_cers.items()}}
        self.base = {"by_scenario": {k: {"samples": 500, "micro_cer": v} for k, v in self.base_cers.items()}}
        self.v2_gate = make_dpo_v2_gate()

    def test_degraded_macro_cer_subset(self) -> None:
        all4 = degraded_macro_cer(self.parent["by_scenario"])
        phys3 = degraded_macro_cer(self.parent["by_scenario"], ("far_field", "noise", "reverb"))
        self.assertAlmostEqual(all4, (0.185 + 0.047 + 0.030 + 0.025) / 4.0)
        self.assertAlmostEqual(phys3, (0.047 + 0.030 + 0.025) / 3.0)
        with self.assertRaises(ValueError):
            degraded_macro_cer(self.parent["by_scenario"], ())

    def test_decoupled_gate_passes_phys3_improvement_with_small_dropout_drift(self) -> None:
        cand_cers = dict(self.parent_cers)
        cand_cers["far_field"] -= 0.0004
        cand_cers["noise"] -= 0.0003
        cand_cers["reverb"] += 0.0003
        cand_cers["dropout"] += 0.0007  # pushes 4-scenario macro positive, but within dropout cap 0.002
        metrics = _make_metrics(cand_cers)
        pref = _make_pref(0.575)
        locate = {"locate_gap_insertions": 0}

        legacy = judge_dpo_checkpoint(metrics, self.parent, self.base, BASE_LEGACY_GATE, pref, step=18, locate=locate)
        self.assertFalse(legacy["passed"])
        self.assertTrue(legacy["risk_stop"])

        v2 = judge_dpo_checkpoint(metrics, self.parent, self.base, self.v2_gate, pref, step=18, locate=locate)
        self.assertTrue(v2["passed"])
        self.assertFalse(v2["risk_stop"])
        self.assertEqual(v2["consecutive_non_passing_evals"], 0)
        self.assertEqual(v2["degraded_macro_scenarios"], ["far_field", "noise", "reverb"])
        # Release gate must accept the passed v2 decision cleanly
        require_passed_dpo_decision(v2, 18, float(self.v2_gate["min_preference_accuracy"]))

    def test_hard_safety_caps_stop_immediately_even_during_warmup_grace(self) -> None:
        cand_cers = dict(self.parent_cers)
        cand_cers["far_field"] -= 0.001
        metrics = _make_metrics(cand_cers)
        # Gap insertion hallucination
        dec_gap = judge_dpo_checkpoint(
            metrics, self.parent, self.base, self.v2_gate, _make_pref(0.70), step=10, locate={"locate_gap_insertions": 10}
        )
        self.assertFalse(dec_gap["passed"])
        self.assertTrue(dec_gap["risk_stop"])
        self.assertTrue(any("max_gap_insertions" in r for r in dec_gap["hard_risks"]))

        # Dropout CER spike
        drop_cers = dict(cand_cers)
        drop_cers["dropout"] += 0.003
        dec_drop = judge_dpo_checkpoint(
            _make_metrics(drop_cers), self.parent, self.base, self.v2_gate, _make_pref(0.70), step=10, locate={"locate_gap_insertions": 0}
        )
        self.assertFalse(dec_drop["passed"])
        self.assertTrue(dec_drop["risk_stop"])
        self.assertTrue(any("dropout micro CER" in r for r in dec_drop["hard_risks"]))

        # Macro delta exceeding hard stop fuse (> 0.002)
        bad_phys = dict(self.parent_cers)
        for s in ("far_field", "noise", "reverb"):
            bad_phys[s] += 0.003
        dec_macro = judge_dpo_checkpoint(
            _make_metrics(bad_phys), self.parent, self.base, self.v2_gate, _make_pref(0.70), step=10, locate={"locate_gap_insertions": 0}
        )
        self.assertFalse(dec_macro["passed"])
        self.assertTrue(dec_macro["risk_stop"])

    def test_trailing_non_passing_and_patience_stop(self) -> None:
        self.assertEqual(trailing_non_passing_evals([]), 0)
        self.assertEqual(trailing_non_passing_evals([{"passed": False}, {"passed": True}]), 0)
        self.assertEqual(trailing_non_passing_evals([{"passed": True}, {"passed": False}, {"passed": False}]), 2)

        wobble_cers = dict(self.parent_cers)
        for s in ("far_field", "noise", "reverb"):
            wobble_cers[s] += 0.0005  # inside soft band (0, 0.002]
        metrics = _make_metrics(wobble_cers)

        for count in (None, -1, True, 0.5):
            with self.subTest(count=count):
                missing = judge_dpo_checkpoint(metrics, self.parent, self.base, self.v2_gate, _make_pref(0.6),
                                              step=20, locate={"locate_gap_insertions": count})
                self.assertTrue(missing["risk_stop"])
                self.assertIn("missing or invalid locate_gap_insertions", missing["hard_risks"])

        history = []
        for step, count in ((20, 0), (40, 0), (80, 1), (120, 2)):
            decision = judge_dpo_checkpoint(
                metrics, self.parent, self.base, self.v2_gate, _make_pref(0.60), step=step,
                locate={"locate_gap_insertions": 0}, prior_non_passing_evals=trailing_non_passing_evals(history, 40),
            )
            self.assertFalse(decision["passed"])
            self.assertEqual(decision["risk_stop"], step == 120)
            self.assertEqual(decision["consecutive_non_passing_evals"], count)
            history.append(dict(step=step, **decision))

    def test_resume_honors_hard_stop_and_requires_prior_decisions(self) -> None:
        soft = {"step": 40, "passed": False, "risk_stop": False}
        stop = {"step": 80, "passed": False, "risk_stop": True}
        self.assertEqual(resume_gate_action([soft, stop], 80, 40, 40, 240), "gate_risk")
        self.assertEqual(resume_gate_action([soft], 80, 40, 40, 240), "evaluate")
        self.assertEqual(resume_gate_action([soft], 40, 40, 40, 240), "continue")
        self.assertEqual(resume_gate_action([soft], 54, 40, 40, 54), "evaluate")
        for history, step in (([stop], 80), ([soft, stop], 120), ([stop, soft], 80), ([soft, soft], 40)):
            with self.subTest(history=history, step=step), self.assertRaises(ValueError):
                resume_gate_action(history, step, 40, 40, 240)

    def test_sft_and_rl_legacy_judgments_unchanged(self) -> None:
        improved = dict(self.parent_cers)
        improved["noise"] -= 0.002
        sft_dec = judge_sft_checkpoint(_make_metrics(improved), self.parent, BASE_LEGACY_GATE)
        self.assertTrue(sft_dec["passed"])
        self.assertFalse(sft_dec["risk_stop"])
        self.assertEqual(sft_dec["degraded_macro_scenarios"], list(DEGRADED_SCENARIOS))

        rl_dec = judge_rl_checkpoint(
            _make_metrics(improved),
            self.parent,
            self.base,
            BASE_LEGACY_GATE,
            step0_reward=0.90,
            current_reward=0.91,
            raw_kl=0.0002,
            reference_matched=True,
        )
        self.assertTrue(rl_dec["passed"])
        self.assertFalse(rl_dec["risk_stop"])


class TestReplayDpoGateCli(unittest.TestCase):
    def test_synthetic_stress_suite_and_warmup_verifier_pass(self) -> None:
        v2_gate = make_dpo_v2_gate()
        stress = run_synthetic_stress_suite(v2_gate)
        self.assertEqual(len(stress), 7)
        for item in stress:
            self.assertTrue(item["passed"], msg=f"stress case failed: {item}")

        warmups = verify_warmup_schedules()
        self.assertEqual({(w['base_lr'], w['warmup_steps'], w['max_steps']) for w in warmups},
                         {(5e-6, 50, 500), (1e-5, 50, 500)})
        for item in warmups:
            self.assertTrue(item["passed"], msg=f"warmup schedule failed: {item}")

    def test_replay_run_directory_from_scored_jsonl_and_cli_json(self) -> None:
        parent_cers = {"clean": 0.02, "noise": 0.03, "reverb": 0.025, "far_field": 0.047, "dropout": 0.185}
        parent = {"by_scenario": {k: {"samples": 50, "micro_cer": v} for k, v in parent_cers.items()}}
        base = {"by_scenario": {k: {"samples": 50, "micro_cer": v + 0.002} for k, v in parent_cers.items()}}
        rows = []
        for sc in SCENARIOS:
            for idx in range(50):
                # Make dropout have a huge spike (0.25 vs 0.185) and far_field slightly improve
                edits = 25 if sc == "dropout" else (4 if sc == "far_field" else int(round(parent_cers[sc] * 100)))
                rows.append(
                    {
                        "sample_id": f"u{idx}__{sc}",
                        "scenario": sc,
                        "edit_count": edits,
                        "reference_char_count": 100,
                        "empty_output": False,
                        "repeated_output": False,
                        "long_output": False,
                        "truncated": False,
                        "valid_output": True,
                    }
                )
        agg = metrics_from_scored_rows(rows)
        self.assertAlmostEqual(agg["by_scenario"]["dropout"]["micro_cer"], 0.25)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            run_dir = tmp / "fake_dpo_run"
            step_dir = run_dir / "eval" / "step_000010"
            step_dir.mkdir(parents=True)
            (step_dir / "scored.jsonl").write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
            )
            (run_dir / "eval_decisions.jsonl").write_text(
                json.dumps(
                    {
                        "step": 10,
                        "passed": False,
                        "risk_stop": True,
                        "preference_accuracy": 0.80,
                        "preference_ties": 0,
                        "raw_policy_accuracy": 0.2,
                        "length_buckets": {"equal_length": {"pairs": 150, "raw_policy_accuracy": 0.2}},
                        "locate_signature": "OTHER",
                        "locate_gap_insertions": 10,
                        "locate_dropout_length_ratio": 0.86,
                        "locate_dropout_delta": 0.065,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rep = replay_run_directory(run_dir, parent, base, BASE_LEGACY_GATE, make_dpo_v2_gate())
            self.assertTrue(rep["legacy_parity_ok"])
            self.assertFalse(rep["steps"][0]["v2_passed"])
            self.assertTrue(rep["steps"][0]["v2_risk_stop"])

            out_json = tmp / "report.json"
            rc = replay_main(["--skip-local-runs", "--output-json", str(out_json)])
            self.assertEqual(rc, 0)
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertTrue(payload["all_checks_passed"])
            self.assertEqual(len(payload["stress_suite"]), 7)

    def test_missing_history_and_incomplete_scores_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            report = directory / "report.json"
            self.assertEqual(replay_main(["--wave1-dir", str(directory), "--output-json", str(report)]), 1)
            self.assertTrue(json.loads(report.read_text())["history_errors"])
            with self.assertRaisesRegex(ValueError, "health flags"):
                metrics_from_scored_rows([{"sample_id": "v", "scenario": "noise", "valid_output": True}])

    def test_replay_can_only_select_reachable_steps_and_requires_counts(self) -> None:
        cers = dict(clean=0.02, dropout=0.185, far_field=0.047, reverb=0.025, noise=0.03)
        parent = _make_metrics(cers)
        candidate = _make_metrics(dict(cers, noise=0.028))
        pref = _make_pref(0.75)
        with tempfile.TemporaryDirectory() as tmpdir:
            run = Path(tmpdir)
            decisions = []
            for step, gap in ((20, 8), (40, 0)):
                directory = run / "eval" / f"step_{step:06d}"
                directory.mkdir(parents=True)
                (directory / "metrics.json").write_text(json.dumps(candidate))
                (directory / "preference.json").write_text(json.dumps({k:v for k,v in pref.items() if k != "pairs"}))
                decisions.append(dict(step=step, passed=True, risk_stop=False, locate_gap_insertions=gap))
            path = run / "eval_decisions.jsonl"
            path.write_text("".join(json.dumps(d) + "\n" for d in decisions))
            replay = replay_run_directory(run, parent, parent, BASE_LEGACY_GATE, make_dpo_v2_gate())
            self.assertIsNone(replay["v2_selected_step"])
            self.assertFalse(replay["steps"][1]["v2_reachable"])
            self.assertTrue(replay["steps"][1]["v2_passed"])
            (directory / "preference.json").write_text(json.dumps({"accuracy": 0.75, "pairs": 160}))
            with self.assertRaisesRegex(ValueError, "pair counts"):
                replay_run_directory(run, parent, parent, BASE_LEGACY_GATE, make_dpo_v2_gate())
            (directory / "preference.json").write_text(json.dumps(pref))
            del decisions[-1]["locate_gap_insertions"]
            path.write_text("".join(json.dumps(d) + "\n" for d in decisions))
            with self.assertRaisesRegex(ValueError, "gap count"):
                replay_run_directory(run, parent, parent, BASE_LEGACY_GATE, make_dpo_v2_gate())


class TestRehearseDpoSchedule(unittest.TestCase):
    def test_rehearse_sweep_flags_too_few_steps_heavy_clip_and_recommends_sweet_spot(self) -> None:
        from scripts.rehearse_dpo_schedule import main as rehearse_main, rehearse_one, rehearse_sweep

        rep = rehearse_sweep(train_pairs=8000, ga_list=[4, 8, 12, 16, 24, 32, 64], drop_tiny_tail=True)
        by_ga = {row["ga"]: row for row in rep["rows"]}
        # ga=4 clips heavily (est_gn=2.30)
        self.assertTrue(any("HEAVY_GRAD_CLIP" in r for r in by_ga[4]["warn_risks"]))
        # ga=64 has only 125 steps (< 200 -> REJECT TOO_FEW_STEPS)
        self.assertEqual(by_ga[64]["verdict"], "REJECT")
        self.assertTrue(any("TOO_FEW_STEPS" in r for r in by_ga[64]["reject_risks"]))
        # ga=16 on 8000 pairs is 500 steps, 5 evals, fits single 360m session -> RECOMMENDED
        self.assertEqual(rep["recommended_ga"], 16)
        self.assertEqual(by_ga[16]["actual_steps"], 500)
        self.assertEqual(by_ga[16]["eval_count"], 5)
        self.assertTrue(by_ga[16]["fits_single_session"])
        self.assertEqual(by_ga[16]["earliest_patience_stop_step"], 300)

        # Tiny tail detection without --drop-tiny-tail
        tiny = rehearse_one(train_pairs=8002, ga=16, drop_tiny_tail=False)
        self.assertTrue(any("TINY_TAIL_STEP" in r for r in tiny["warn_risks"]))
        trimmed = rehearse_one(train_pairs=8002, ga=16, drop_tiny_tail=True)
        self.assertEqual(trimmed["tail_pairs"], 0)
        self.assertEqual(trimmed["actual_steps"], 500)

        # Runtime overflow when too many evals are requested
        overflow = rehearse_one(train_pairs=8000, ga=16, eval_steps=40, max_runtime_minutes=360)
        self.assertEqual(overflow["verdict"], "REJECT")
        self.assertTrue(any("RUNTIME_OVERFLOW" in r for r in overflow["reject_risks"]))

        # CLI writes JSON report cleanly
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "rehearse.json"
            rc = rehearse_main(["sweep", "--train-pairs", "8000", "--ga-list", "8,16,32", "--drop-tiny-tail", "--output-json", str(out)])
            self.assertEqual(rc, 0)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["recommended_ga"], 16)


if __name__ == "__main__":
    unittest.main()
