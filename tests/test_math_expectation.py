from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.math_expectation import evaluate_math_expectation, main as math_exp_main


def _make_pair(n_sources: int = 80, *, improvement: int = 1, clean_regress: int = 0):
    manifest = []
    ref_scored = []
    cand_scored = []
    scenarios = ("clean", "noise", "reverb", "far_field", "dropout")
    for i in range(n_sources):
        src = f"SRC{i:04d}"
        for sc in scenarios:
            sid = f"{src}__{sc}"
            manifest.append({"sample_id": sid, "source_utterance_id": src, "scenario": sc})
            ref_ed = 1 if sc == "clean" else 4
            if sc == "clean":
                cand_ed = ref_ed + clean_regress
            elif sc in ("noise", "reverb", "far_field"):
                cand_ed = max(0, ref_ed - improvement)
            else:
                cand_ed = ref_ed
            ref_chars = 20
            ref_words = 10
            ref_scored.append(
                {
                    "sample_id": sid,
                    "scenario": sc,
                    "edit_count": ref_ed,
                    "reference_char_count": ref_chars,
                    "sample_cer": ref_ed / ref_chars,
                    "empty_output": False,
                    "repeated_output": False,
                    "word_edit_count": ref_ed,
                    "reference_word_count": ref_words,
                }
            )
            cand_scored.append(
                {
                    "sample_id": sid,
                    "scenario": sc,
                    "edit_count": cand_ed,
                    "reference_char_count": ref_chars,
                    "sample_cer": cand_ed / ref_chars,
                    "empty_output": False,
                    "repeated_output": False,
                    "word_edit_count": cand_ed,
                    "reference_word_count": ref_words,
                }
            )
    return manifest, ref_scored, cand_scored


class MathExpectationTest(unittest.TestCase):
    def test_significant_improvement_passes_expectation_and_is_reproducible(self) -> None:
        manifest, ref, cand = _make_pair(80, improvement=1, clean_regress=0)
        r1 = evaluate_math_expectation(manifest, ref, cand, num_resamples=500, seed=42, require_ci95_negative=True)
        r2 = evaluate_math_expectation(manifest, ref, cand, num_resamples=500, seed=42, require_ci95_negative=True)
        self.assertEqual(r1, r2)
        self.assertTrue(r1["verdict"]["passed_expectation"])
        self.assertTrue(r1["verdict"]["statistically_significant_95"])
        self.assertTrue(r1["verdict"]["clean_safe"])
        self.assertEqual(r1["verdict"]["target_slice"], "physical_degraded_3")
        self.assertLess(r1["slices"]["physical_degraded_3"]["expected_cer_delta"], 0.0)
        self.assertEqual(r1["slices"]["physical_degraded_3"]["cer_win_prob"], 1.0)
        self.assertGreater(r1["slices"]["physical_degraded_3"]["expected_reward_delta"], 0.0)
        self.assertTrue(r1["has_wer"])
        self.assertLess(r1["slices"]["physical_degraded_3"]["expected_wer_delta"], 0.0)

    def test_clean_regression_fails_clean_safe_verdict(self) -> None:
        manifest, ref, cand = _make_pair(60, improvement=1, clean_regress=2)
        rep = evaluate_math_expectation(
            manifest,
            ref,
            cand,
            num_resamples=500,
            seed=42,
            clean_cer_increase_max=0.01,
        )
        self.assertFalse(rep["verdict"]["clean_safe"])
        self.assertFalse(rep["verdict"]["passed_expectation"])

    def test_zero_change_fails_expectation(self) -> None:
        manifest, ref, cand = _make_pair(50, improvement=0, clean_regress=0)
        rep = evaluate_math_expectation(manifest, ref, cand, num_resamples=500, seed=42)
        self.assertFalse(rep["verdict"]["expected_improvement"])
        self.assertFalse(rep["verdict"]["passed_expectation"])

    def test_cli_outputs_json_and_handles_ood_slices(self) -> None:
        manifest = []
        ref = []
        cand = []
        for i in range(40):
            src = f"MTG{i:03d}"
            for sc in ("near_field", "far_field"):
                sid = f"{src}__{sc}"
                manifest.append({"sample_id": sid, "source_utterance_id": src, "scenario": sc})
                re = 1 if sc == "near_field" else 3
                ce = 1 if sc == "near_field" else 2
                ref.append({"sample_id": sid, "scenario": sc, "edit_count": re, "reference_char_count": 15, "sample_cer": re / 15})
                cand.append({"sample_id": sid, "scenario": sc, "edit_count": ce, "reference_char_count": 15, "sample_cer": ce / 15})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m_p, r_p, c_p, out_p = root / "m.jsonl", root / "r.jsonl", root / "c.jsonl", root / "out.json"
            for p, rows in ((m_p, manifest), (r_p, ref), (c_p, cand)):
                p.write_text("\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8")
            rc = math_exp_main(["--manifest", str(m_p), "--ref-scored", str(r_p), "--cand-scored", str(c_p), "--output", str(out_p), "--resamples", "400"])
            self.assertEqual(rc, 0)
            payload = json.loads(out_p.read_text(encoding="utf-8"))
            self.assertEqual(payload["verdict"]["target_slice"], "far_field")
            self.assertEqual(payload["verdict"]["clean_slice"], "near_field")
            self.assertTrue(payload["verdict"]["passed_expectation"])


if __name__ == "__main__":
    unittest.main()
