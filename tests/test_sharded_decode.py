from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_sharded_decode import merge_prediction_shards, shard_bounds, write_baseline_json


class ShardedDecodeTest(unittest.TestCase):
    def test_shard_bounds_partition_exactly(self) -> None:
        covered: list[int] = []
        for idx in range(5):
            s, e = shard_bounds(2503, idx, 5)
            covered.extend(range(s, e))
        self.assertEqual(covered, list(range(2503)))

    def test_merge_verifies_exact_order_and_rejects_missing_or_dup(self) -> None:
        manifest = [{"sample_id": f"S{i:03d}__clean"} for i in range(6)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            s0 = root / "s0.jsonl"
            s1 = root / "s1.jsonl"
            s0.write_text("\n".join(json.dumps({"sample_id": f"S{i:03d}__clean", "prediction_raw": f"t{i}"}) for i in (3, 4, 5)) + "\n", encoding="utf-8")
            s1.write_text("\n".join(json.dumps({"sample_id": f"S{i:03d}__clean", "prediction_raw": f"t{i}"}) for i in (0, 1, 2)) + "\n", encoding="utf-8")
            out = root / "merged.jsonl"
            info = merge_prediction_shards(manifest, [s0, s1], out)
            self.assertEqual(info["rows"], 6)
            lines = [json.loads(line)["sample_id"] for line in out.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(lines, [f"S{i:03d}__clean" for i in range(6)])

            with self.assertRaises(ValueError):
                merge_prediction_shards(manifest, [s0], out)
            with self.assertRaises(ValueError):
                merge_prediction_shards(manifest, [s0, s0], out)

    def test_write_baseline_json_schema(self) -> None:
        metrics = {
            "samples": 2500,
            "by_scenario": {
                sc: {"samples": 500, "micro_cer": 0.05}
                for sc in ("clean", "dropout", "far_field", "noise", "reverb")
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "base.json"
            payload = write_baseline_json(
                metrics,
                manifest_sha256="a" * 64,
                predictions_sha256="b" * 64,
                decode_eval_sha256="c" * 64,
                output_path=out,
            )
            self.assertEqual(payload["schema_version"], "qwen3-asr-baseline-v1")
            self.assertEqual(payload["samples"], 2500)


if __name__ == "__main__":
    unittest.main()
