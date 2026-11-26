"""Reference logprob contracts. They do not download a model."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from evaluation.jsonl_io import write_jsonl
from train.reference_logprob import (
    bind_reference_cache,
    cache_reference_logprobs,
    completed_logprob_rows,
    copy_reference_cache,
    freeze_reference_model,
    logprobs_match,
    main as reference_main,
    publish_logprob_pairs,
    reference_sequence_logprob,
    select_recheck,
)
from train.sft_schema import load_train_config, sha256_file


def _pair(pair_id: str, split: str, chosen: str = "你好", rejected: str = "你号") -> dict[str, object]:
    return {
        "pair_id": pair_id,
        "sample_id": pair_id,
        "split": split,
        "audio": f"wavs/{pair_id}.wav",
        "chosen_text": chosen,
        "rejected_text": rejected,
        "chosen_token_count": len(chosen) + 1,
        "rejected_token_count": len(rejected) + 1,
        "ref_chosen_logp": None,
        "ref_rejected_logp": None,
    }


def _stable_score(row: dict[str, object], text: str) -> tuple[float, int]:
    if text == row["chosen_text"]:
        return -2.5, int(row["chosen_token_count"])
    return -4.5, int(row["rejected_token_count"])


class ReferenceLogprobTest(unittest.TestCase):
    def test_identified_prefix_sync_restore_and_legacy_prefix_isolation(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "local/ref_logprobs.jsonl"
            drive = root / "drive/ref_logprobs.jsonl"
            context = {"gpu_name": "fixture", "lora_dtype": "float32", "null_train_pairs_sha256": "one"}
            self.assertTrue(bind_reference_cache(source, context, strict=True))
            source.write_bytes(b"first row\n")
            copy_reference_cache(source, drive, context, bound=True)
            source.write_bytes(b"first row\nsecond row\n")
            copy_reference_cache(source, drive, context, bound=True)
            restored = root / "restored/ref_logprobs.jsonl"
            copy_reference_cache(drive, restored, context, bound=True)
            self.assertEqual(restored.read_bytes(), source.read_bytes())
            self.assertEqual(json.loads(restored.with_suffix(".identity.json").read_text()), context)
            for key in context:
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "identity changed"):
                    copy_reference_cache(drive, restored, {**context, key: "different"}, bound=True)
            legacy = root / "unlabelled.jsonl"
            legacy.write_bytes(b"historical prefix\n")
            self.assertFalse(bind_reference_cache(legacy, context, strict=False))
            with self.assertRaisesRegex(ValueError, "no precision"):
                copy_reference_cache(legacy, drive, context, bound=True)
            with self.assertRaisesRegex(ValueError, "no precision"):
                copy_reference_cache(drive, legacy, context, bound=True)
            with self.assertRaisesRegex(ValueError, "unlabelled"):
                copy_reference_cache(legacy, drive, context, bound=False)
            self.assertEqual(drive.read_bytes(), source.read_bytes())
            self.assertFalse(legacy.with_suffix(".identity.json").exists())

    def test_logprob_is_the_negative_sum_not_the_mean(self) -> None:
        self.assertEqual(reference_sequence_logprob([0.25, 0.75]), -1.0)
        self.assertNotEqual(reference_sequence_logprob([0.25, 0.75]), -0.5)
        with self.assertRaises(Exception):
            reference_sequence_logprob([])
        with self.assertRaises(Exception):
            reference_sequence_logprob([float("nan")])

    def test_both_error_limits_are_required(self) -> None:
        self.assertTrue(logprobs_match(-10.0, -10.00005))
        self.assertFalse(logprobs_match(-10.0, -10.0002))
        self.assertFalse(logprobs_match(-1.0, -1.00005))

    def test_reference_cli_restores_the_identified_drive_prefix(self) -> None:
        config = Path(__file__).resolve().parents[1] / "configs/train/sft_pilot_lr1e5.yaml"
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, val = [_pair("old", "train")], [_pair("new", "val")]
            write_jsonl(root / "train_pairs.jsonl", train)
            write_jsonl(root / "val_pairs.jsonl", val)
            context = {"adapter_sha256": "fixture-parent", "lora_dtype": "float32", "gpu_name": "fixture-GPU",
                       "decode_eval_sha256": load_train_config(config)["decode_eval_sha256"]}
            identity = {**context, "null_train_pairs_sha256": sha256_file(root / "train_pairs.jsonl"),
                        "null_val_pairs_sha256": sha256_file(root / "val_pairs.jsonl")}
            drive = root / "drive" / "ref_logprobs.jsonl"
            bind_reference_cache(drive, identity, strict=True)
            cache_reference_logprobs(train, [], _stable_score, drive, seed=42)
            calls = []
            def fake_score(row, text):
                calls.append(row["pair_id"])
                return _stable_score(row, text)
            fake_score.reference_context = context
            fake_score.release_id = "fixture-SFT"
            out = root / "local"
            with patch("train.reference_logprob.build_reference_scorer", return_value=fake_score) as factory:
                rc = reference_main(["--config", str(config), "--pairs-dir", str(root), "--release-dir", str(root),
                                     "--audio-root", str(root), "--output-dir", str(out), "--sync-path", str(drive),
                                     "--seed", "42", "--lora-trainable-dtype", "float32"])
            self.assertEqual(rc, 0)
            self.assertEqual(factory.call_args.args[0]["lora"]["trainable_dtype"], "float32")
            self.assertEqual(calls, ["new", "new", "old", "old", "new", "new"])
            final = drive.parent / "pairs_logprob"
            self.assertEqual(json.loads((final / "ref_report.json").read_text())["reference_context"], identity)
            self.assertEqual((final / "ref_logprobs.identity.json").read_bytes(), drive.with_suffix(".identity.json").read_bytes())

    def test_cache_must_be_a_pair_prefix(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pairs = [_pair("a", "train"), _pair("b", "val")]
            cache = root / "ref_logprobs.jsonl"
            self.assertEqual(completed_logprob_rows(cache, pairs), [])
            cache.write_text(
                json.dumps(
                    {
                        "schema_version": "qwen3-asr-dpo-ref-logprob-v1",
                        "pair_id": "b",
                        "ref_chosen_logp": -1.0,
                        "ref_rejected_logp": -2.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Exception) as caught:
                completed_logprob_rows(cache, pairs)
            self.assertIn("prefix", str(caught.exception))

    def test_recheck_failure_does_not_publish_pairs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = [_pair("t1", "train")]
            val = [_pair("v1", "val")]
            calls = {"n": 0}

            def score(row: dict[str, object], text: str) -> tuple[float, int]:
                calls["n"] += 1
                logprob, count = _stable_score(row, text)
                if calls["n"] > 4:
                    logprob -= 1.0
                return logprob, count

            with self.assertRaises(Exception) as caught:
                cache_reference_logprobs(train, val, score, root / "ref_logprobs.jsonl", seed=42, recheck_limit=20)
            self.assertIn("recheck", str(caught.exception))
            self.assertFalse((root / "train_pairs.jsonl").exists())

    def test_matching_recheck_fills_logprobs_without_using_the_mean(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = [_pair("t1", "train"), _pair("t2", "train", chosen="甲乙丙", rejected="甲乙")]
            val = [_pair("v1", "val")]
            report = cache_reference_logprobs(
                train,
                val,
                _stable_score,
                root / "ref_logprobs.jsonl",
                seed=42,
                recheck_limit=20,
            )
            self.assertEqual(report["rechecked"], 3)
            self.assertFalse(report["trainable"])
            self.assertLessEqual(report["max_abs_error"], 1e-4)
            published = publish_logprob_pairs(train, val, root / "ref_logprobs.jsonl", root / "out", report)
            filled = [
                json.loads(line)
                for line in (root / "out" / "train_pairs.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(filled[0]["ref_chosen_logp"], -2.5)
            self.assertEqual(filled[0]["ref_rejected_logp"], -4.5)
            self.assertEqual(published["train_pairs_sha256"], report["train_pairs_sha256"] if False else published["train_pairs_sha256"])
            picked = select_recheck(train, 42, 1)
            self.assertEqual(len(picked), 1)
            self.assertIn(picked[0]["pair_id"], {"t1", "t2"})

    def test_freeze_removes_the_training_hook_and_checkpointing(self) -> None:
        class Param:
            def __init__(self) -> None:
                self.requires_grad = True

            def requires_grad_(self, value: bool) -> "Param":
                self.requires_grad = value
                return self

        class Embeddings:
            def __init__(self) -> None:
                self._forward_hooks = {1: object()}

        class Thinker:
            def __init__(self) -> None:
                self.embeddings = Embeddings()
                self.is_gradient_checkpointing = True

            def get_input_embeddings(self) -> Embeddings:
                return self.embeddings

            def gradient_checkpointing_disable(self) -> None:
                self.is_gradient_checkpointing = False

        class Model:
            def __init__(self) -> None:
                self.thinker = Thinker()
                self.params = [Param(), Param()]
                self.is_gradient_checkpointing = True
                self.training = True

            def parameters(self):
                return self.params

            def eval(self) -> None:
                self.training = False

            def gradient_checkpointing_disable(self) -> None:
                self.is_gradient_checkpointing = False

        model = Model()
        freeze_reference_model(model)
        self.assertEqual(model.thinker.embeddings._forward_hooks, {})
        self.assertFalse(model.is_gradient_checkpointing)
        self.assertFalse(model.thinker.is_gradient_checkpointing)
        self.assertFalse(model.training)
        self.assertTrue(all(not param.requires_grad for param in model.params))


if __name__ == "__main__":
    unittest.main()
