from __future__ import annotations

import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from scripts.build_slr28_manifest import build_slr28_manifest, validate_slr28_manifest_dir
from scripts.manifest_contract import sha256_file
from scripts.slr28_audio import (
    SLR28_DATA_SCHEMA,
    SLR28_PROCESSOR_VERSION,
    Slr28AssetCatalog,
    canonical_real_room_family,
    convolve_rir_float64,
    load_slr28_data_config,
    mix_real_noise_float64,
)


def _wav_bytes(samples: np.ndarray, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _make_synthetic_slr28_zip(path: Path) -> str:
    rng = np.random.default_rng(123)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        real_rir_lines = []
        iso_noise_lines = []
        # 6 real room families so train/val/test all get >=1 room + linked isotropic noise
        for ridx in range(1, 7):
            room = f"RVB2014_room{ridx}"
            # 2-channel real RIR with leading delay
            rir = np.zeros((4000, 2), dtype=np.float64)
            rir[200, 0] = 0.9
            rir[450, 0] = 0.35
            rir[900, 0] = 0.15
            rir[220, 1] = 0.85
            rir[500, 1] = 0.30
            m_rir = f"RIRS_NOISES/real_rirs_isotropic_noises/{room}_rir.wav"
            zf.writestr(m_rir, _wav_bytes(rir))
            real_rir_lines.append(f"--rir-id {ridx:05d} --room-id {room} {m_rir}")

            noi = rng.normal(0.0, 0.05, size=(16000, 2))
            m_noi = f"RIRS_NOISES/real_rirs_isotropic_noises/{room}_noise.wav"
            zf.writestr(m_noi, _wav_bytes(noi))
            iso_noise_lines.append(f"--noise-id {ridx:05d} --noise-type isotropic --room-linkage {room} {m_noi}")

        # Add multi-position air room to verify room_family grouping
        for pos in ("0", "1"):
            rir = np.zeros((3000,), dtype=np.float64)
            rir[100] = 0.8
            rir[300] = 0.3
            m = f"RIRS_NOISES/real_rirs_isotropic_noises/air_meeting_{pos}.wav"
            zf.writestr(m, _wav_bytes(rir))
            real_rir_lines.append(f"--rir-id 9000{pos} --room-id air_binaural_meeting_{pos} {m}")

        zf.writestr("RIRS_NOISES/real_rirs_isotropic_noises/rir_list", "\n".join(real_rir_lines) + "\n")
        zf.writestr("RIRS_NOISES/real_rirs_isotropic_noises/noise_list", "\n".join(iso_noise_lines) + "\n")

        for sz in ("smallroom", "mediumroom", "largeroom"):
            sim_lines = []
            for ridx in range(1, 11):
                room = f"{sz[:-4]}-Room{ridx:03d}"
                rir = np.zeros((4000,), dtype=np.float64)
                rir[150] = 0.9
                rir[600] = 0.4
                rir[1200] = 0.2
                m = f"RIRS_NOISES/simulated_rirs/{sz}/Room{ridx:03d}/Room{ridx:03d}-00001.wav"
                zf.writestr(m, _wav_bytes(rir))
                sim_lines.append(f"--rir-id {room}-00001 --room-id {room} {m}")
            zf.writestr(f"RIRS_NOISES/simulated_rirs/{sz}/rir_list", "\n".join(sim_lines) + "\n")

        ps_lines = []
        for nidx in range(1, 11):
            noi = rng.normal(0.0, 0.1, size=(12000,))
            m = f"RIRS_NOISES/pointsource_noises/noise-free-sound-{nidx:04d}.wav"
            zf.writestr(m, _wav_bytes(noi))
            ps_lines.append(f"--noise-id noise-free-sound-{nidx:04d} --noise-type point-source --bg-fg-type foreground {m}")
        zf.writestr("RIRS_NOISES/pointsource_noises/noise_list", "\n".join(ps_lines) + "\n")
    return sha256_file(path)


class Slr28ManifestTest(unittest.TestCase):
    def test_canonical_real_room_family_groups_same_physical_room(self) -> None:
        self.assertEqual(canonical_real_room_family("air_binaural_meeting_0"), "air_binaural_meeting")
        self.assertEqual(canonical_real_room_family("air_binaural_meeting_1"), "air_binaural_meeting")
        self.assertEqual(canonical_real_room_family("air_binaural_aula_carolina_1_1_90"), "air_binaural_aula_carolina")
        self.assertEqual(canonical_real_room_family("air_phone_stairway2"), "air_phone_stairway")
        self.assertEqual(canonical_real_room_family("RWCP_m30l.wav"), "RWCP_directional_anechoic")

    def test_float64_convolve_and_snr_mixing_are_deterministic(self) -> None:
        t = np.arange(16000, dtype=np.float64) / 16000.0
        clean = 0.25 * np.sin(2.0 * np.pi * 440.0 * t)
        rir = np.zeros(4000, dtype=np.float64)
        rir[500] = 0.8
        rir[1000] = 0.3
        out1 = convolve_rir_float64(clean, rir)
        out2 = convolve_rir_float64(clean, rir)
        np.testing.assert_array_equal(out1, out2)
        self.assertEqual(out1.shape, clean.shape)
        self.assertLessEqual(float(np.max(np.abs(out1))), 0.990001)

        noise = 0.1 * np.cos(2.0 * np.pi * 170.0 * t[:8000])
        m1 = mix_real_noise_float64(clean, noise, snr_db=5.0, offset_seed=77)
        m2 = mix_real_noise_float64(clean, noise, snr_db=5.0, offset_seed=77)
        np.testing.assert_array_equal(m1, m2)
        # Actual added noise power matches target 5 dB SNR closely
        added = m1 - clean
        snr_est = 10.0 * np.log10(np.mean(clean**2) / np.mean(added**2))
        self.assertAlmostEqual(float(snr_est), 5.0, places=2)

    def test_zero_leakage_split_and_reproducible_manifest_build(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "rirs_noises.zip"
            archive_sha = _make_synthetic_slr28_zip(archive)

            catalog = Slr28AssetCatalog(archive, seed=42, val_fraction=0.2, test_fraction=0.2)
            self.assertTrue(catalog.audit["zero_leakage"])
            for sp in ("train", "val", "test"):
                self.assertGreater(len(catalog.real_rirs[sp]), 0)
                self.assertGreater(len(catalog.isotropic_noises[sp]), 0)
                self.assertGreater(len(catalog.pointsource_noises[sp]), 0)
                self.assertGreater(len(catalog.sim_rirs[sp]["mediumroom"]), 0)

            cfg_dict = yaml.safe_load(Path("configs/data/manifest_slr28_track3.yaml").read_text(encoding="utf-8"))
            cfg_dict["slr28"]["archive_sha256"] = archive_sha
            cfg_dict["slr28"]["split_fractions"] = {"val": 0.2, "test": 0.2}
            cfg_dict["quota"]["min_val_rows_per_scenario"] = 2
            cfg_path = root / "cfg.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg_dict, sort_keys=False), encoding="utf-8")
            config = load_slr28_data_config(cfg_path)
            self.assertEqual(config["schema_version"], SLR28_DATA_SCHEMA)
            self.assertEqual(config["audio_processor_version"], SLR28_PROCESSOR_VERSION)

            src_dir = root / "src"
            (src_dir / "audio").mkdir(parents=True)
            src_rows = []
            specs = [
                ("UTT001", "SPK01", "train", "今天天气晴朗适合出门散步"),
                ("UTT002", "SPK01", "train", "语音识别系统需要在噪声环境下保持稳健"),
                ("UTT003", "SPK02", "dev", "远场会议室混响会显著拉高识别错误率"),
                ("UTT004", "SPK03", "dev", "真实房间脉冲响应可以缩小域外泛化差距"),
                ("UTT005", "SPK04", "test", "封存测试集只允许在发布前评测一次"),
                ("UTT006", "SPK05", "test", "配对源簇重抽样用于检验统计数学期望"),
            ]
            t = np.arange(16000, dtype=np.float64) / 16000.0
            for idx, (uid, spk, split, text) in enumerate(specs, start=1):
                wav = 0.25 * np.sin(2.0 * np.pi * (220.0 + 40.0 * idx) * t)
                rel = f"audio/{uid}.wav"
                sf.write(src_dir / rel, wav, 16000, format="WAV", subtype="PCM_16")
                src_rows.append(
                    {
                        "source_utterance_id": uid,
                        "audio": rel,
                        "text": text,
                        "language": "zh",
                        "speaker_id": spk,
                        "official_split": split,
                        "source_package": "synthetic-slr28-fixture",
                    }
                )
            sources_path = src_dir / "source.jsonl"
            sources_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in src_rows) + "\n", encoding="utf-8")

            out1 = root / "out1"
            out2 = root / "out2"
            r1 = build_slr28_manifest(config, sources_path, archive, out1, "deadbeef", workers=1)
            r2 = build_slr28_manifest(config, sources_path, archive, out2, "deadbeef", workers=2)
            self.assertEqual(r1["status"], "passed")
            self.assertEqual(r2["status"], "passed")
            for fname in ("train.jsonl", "val.jsonl", "test.jsonl", "manifest.json", "slr28_split_audit.json"):
                self.assertEqual(sha256_file(out1 / fname), sha256_file(out2 / fname), f"mismatch on {fname}")

            confirmed, code = validate_slr28_manifest_dir(out1, config, True)
            self.assertEqual(code, 0)
            self.assertEqual(confirmed["status"], "passed")
            for k, v in confirmed["slr28_usage_audit"]["rir_leakage"].items():
                self.assertEqual(v, [], f"rir leakage {k}")
            for k, v in confirmed["slr28_usage_audit"]["noise_leakage"].items():
                self.assertEqual(v, [], f"noise leakage {k}")

    def test_real_track3_config_parses_cleanly(self) -> None:
        cfg = load_slr28_data_config(Path("configs/data/manifest_slr28_track3.yaml"))
        self.assertEqual(cfg["seed"], 42)
        self.assertTrue(cfg["audited"])
        self.assertEqual(
            cfg["slr28"]["archive_sha256"],
            "3b50cfde915b3984738169b4beb341e9f6b8062ae4c2076146c5db71c2c05dc7",
        )


if __name__ == "__main__":
    unittest.main()
