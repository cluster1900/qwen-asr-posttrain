from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from evaluation.zh_cer import normalize_zh_cer_v1
from scripts.align_train_punctuation import (
    align_punctuation,
    build_punctuated_train_manifest,
    extract_clean_train_manifest,
)
from train.sft_schema import load_train_config, semantic_config_sha256


class AlignTrainPunctuationTest(unittest.TestCase):
    def test_align_punctuation_preserves_characters_across_edits(self) -> None:
        cases = [
            (
                "在工业建筑交通三大高能耗领域中",
                "在工业、建筑、交通三大高能耗领域中。",
                "在工业、建筑、交通三大高能耗领域中。",
            ),
            (
                "张文秀说清者自清",
                "张文秀说：“清者自清。”",
                "张文秀说：清者自清。",
            ),
            (
                "新募集的钱既会投入到战略投资中也会投入财务投资",
                "私募基金的钱，既会投到战略投资中，也会投入财务投资。",
                "新募集的钱，既会投入到战略投资中，也会投入财务投资。",
            ),
            (
                "这是什么原因",
                "这是什么原因？",
                "这是什么原因？",
            ),
            (
                "今天天气真好",
                "",
                "今天天气真好",
            ),
        ]
        for ref, hyp, expected in cases:
            got = align_punctuation(ref, hyp)
            self.assertEqual(normalize_zh_cer_v1(got), normalize_zh_cer_v1(ref))
            self.assertEqual(got, expected)

    def test_extract_and_build_rejects_non_train_and_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_rows = [
                {"sample_id": "u1__clean", "source_utterance_id": "u1", "split": "train", "scenario": "clean", "language": "zh", "audio": "a.wav", "text": "今天天气真好我们出去玩"},
                {"sample_id": "u1__far_field", "source_utterance_id": "u1", "split": "train", "scenario": "far_field", "language": "zh", "audio": "b.wav", "text": "今天天气真好我们出去玩"},
                {"sample_id": "u2__clean", "source_utterance_id": "u2", "split": "train", "scenario": "clean", "language": "zh", "audio": "c.wav", "text": "这是一条测试句子"},
            ]
            train_p = root / "train.jsonl"
            train_p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in train_rows) + "\n", encoding="utf-8")

            clean_p = root / "train_clean.jsonl"
            ext = extract_clean_train_manifest(train_p, clean_p)
            self.assertEqual(ext["clean_rows"], 2)

            preds_p = root / "clean_preds.jsonl"
            preds = [
                {"sample_id": "u1__clean", "prediction_raw": "今天天气真好，我们出去玩。", "error": None},
                {"sample_id": "u2__clean", "prediction_raw": "", "error": "decode_err"},
            ]
            preds_p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in preds) + "\n", encoding="utf-8")

            out1 = root / "train_punct_1.jsonl"
            aud1 = root / "audit_1.json"
            a1 = build_punctuated_train_manifest(train_p, preds_p, out1, aud1)
            self.assertEqual(a1["char_mismatches"], 0)
            self.assertEqual(a1["sources_with_punct"], 1)
            self.assertEqual(a1["fallback_sources"], 1)
            self.assertEqual(a1["rows_written"], 3)
            self.assertEqual(a1["rows_with_punct"], 2)

            built = [json.loads(l) for l in out1.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(built[0]["text_punct"], "今天天气真好，我们出去玩。")
            self.assertEqual(built[1]["text_punct"], "今天天气真好，我们出去玩。")
            self.assertEqual(built[2]["text_punct"], "这是一条测试句子")
            self.assertEqual(built[0]["text"], "今天天气真好我们出去玩")

            # Reject non-train split
            bad_p = root / "bad.jsonl"
            bad_p.write_text(json.dumps({**train_rows[0], "split": "val"}, ensure_ascii=False) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                extract_clean_train_manifest(bad_p, root / "bad_out.jsonl")

    def test_sft_schema_target_text_field_optional_preserves_legacy_hash(self) -> None:
        legacy_p = Path("configs/train/sft_pilot_lr1e5.yaml")
        legacy = load_train_config(legacy_p)
        self.assertNotIn("target_text_field", legacy["data"])
        msha = {"train": "a" * 64, "val": "b" * 64, "metadata": "c" * 64}
        legacy_hash = semantic_config_sha256(legacy, msha)

        raw = yaml.safe_load(legacy_p.read_text(encoding="utf-8"))
        raw["profile"] = "scale"
        raw["data"]["target_text_field"] = "text_punct"
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "punct.yaml"
            p.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
            resolved = load_train_config(p)
            self.assertEqual(resolved["data"]["target_text_field"], "text_punct")
            self.assertNotEqual(semantic_config_sha256(resolved, msha), legacy_hash)

            raw["data"]["target_text_field"] = "invalid_col"
            p2 = Path(tmp) / "bad.yaml"
            p2.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_train_config(p2)


if __name__ == "__main__":
    unittest.main()
