"""Phase 6 Track 3 OpenSLR 28 physical RIR & real-noise zero-leakage audio engine.

All DSP operations (channel selection, direct-path alignment, FFT convolution,
RMS matching, and SNR noise mixing) run strictly in float64 to guarantee
cross-CPU determinism before single-pass PCM_16 quantization.
"""

from __future__ import annotations

import hashlib
import io
import math
import re
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import yaml

from scripts.manifest_contract import SCENARIOS, SPLITS, sha256_file
from scripts.prepare_phase0_audio import to_mono_16k

SLR28_DATA_SCHEMA = "qwen3-asr-data-v2"
SLR28_PROCESSOR_VERSION = "phase6-slr28-phys-v1"
SLR28_SPLIT_AUDIT_SCHEMA = "qwen3-asr-slr28-split-audit-v1"
VALID_SIM_ROOM_SIZES = ("smallroom", "mediumroom", "largeroom")

_TRAIL_NUM_RE = re.compile(r"(?:_\d+)+$")
_RWCP_DIR_RE = re.compile(r"^RWCP_[mp0]\d+[lr]\.wav$")
_AIR_PHONE_NUM_RE = re.compile(r"^(air_phone_[a-z_]+?)\d+$")


@dataclass(frozen=True)
class Slr28RirEntry:
    rir_id: str
    room_id: str
    room_family: str
    member_path: str
    kind: str  # "real" or "simulated"
    room_size: str  # "real" | "smallroom" | "mediumroom" | "largeroom"
    split: str


@dataclass(frozen=True)
class Slr28NoiseEntry:
    noise_id: str
    noise_group: str
    member_path: str
    noise_type: str  # "isotropic" or "point-source"
    split: str


def canonical_real_room_family(room_id: str) -> str:
    """Group multi-position measurements of the same physical room into one leakage unit."""
    if _RWCP_DIR_RE.match(room_id):
        return "RWCP_directional_anechoic"
    if room_id.startswith("air_binaural_"):
        return _TRAIL_NUM_RE.sub("", room_id)
    m = _AIR_PHONE_NUM_RE.match(room_id)
    if m:
        return m.group(1)
    if room_id.startswith("RWCP_cirline_ofc"):
        return "RWCP_cirline_ofc"
    return room_id


def _hash_unit(seed: int, key: str, domain: str) -> float:
    digest = hashlib.sha256(f"{seed}:{key}:{domain}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _deterministic_int(seed: int, *parts: str) -> int:
    joined = ":".join([str(seed), *parts])
    digest = hashlib.sha256(joined.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _partition_groups(
    groups: list[str],
    *,
    seed: int,
    domain: str,
    val_fraction: float,
    test_fraction: float,
) -> dict[str, str]:
    """Deterministically partition unique group keys into non-empty train/val/test splits."""
    unique = sorted(set(groups))
    if len(unique) < 3:
        raise ValueError(f"SLR28 domain {domain!r} requires at least 3 distinct groups, found {len(unique)}")
    ranked = sorted(unique, key=lambda g: hashlib.sha256(f"{seed}:{g}:{domain}".encode("utf-8")).hexdigest())
    n = len(ranked)
    n_test = max(1, int(round(n * test_fraction)))
    n_val = max(1, int(round(n * val_fraction)))
    if n_test + n_val >= n:
        n_test = 1
        n_val = 1
    mapping: dict[str, str] = {}
    for idx, grp in enumerate(ranked):
        if idx < n_test:
            mapping[grp] = "test"
        elif idx < n_test + n_val:
            mapping[grp] = "val"
        else:
            mapping[grp] = "train"
    return mapping


def _parse_flag_value(tokens: list[str], flag: str) -> str:
    idx = tokens.index(flag)
    if idx + 1 >= len(tokens):
        raise ValueError(f"missing value after {flag}")
    return tokens[idx + 1]


def load_slr28_data_config(path: Path) -> dict[str, Any]:
    """Parse and strictly validate qwen3-asr-data-v2 Track 3 configuration."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a mapping")
    allowed = {
        "schema_version",
        "seed",
        "source_version",
        "audited",
        "audio_processor_version",
        "language",
        "sample_rate",
        "duration_seconds",
        "splits",
        "quota",
        "scenarios",
        "slr28",
        "augmentation",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown config fields: {unknown}")
    if raw.get("schema_version") != SLR28_DATA_SCHEMA:
        raise ValueError(f"schema_version must be {SLR28_DATA_SCHEMA}")
    seed = raw.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    source_version = raw.get("source_version")
    if not isinstance(source_version, str) or not source_version.strip():
        raise ValueError("source_version must be a non-empty string")
    audited = raw.get("audited")
    if not isinstance(audited, bool):
        raise ValueError("audited must be a boolean")
    if raw.get("audio_processor_version") != SLR28_PROCESSOR_VERSION:
        raise ValueError(f"audio_processor_version must be {SLR28_PROCESSOR_VERSION}")
    if raw.get("language") != "zh":
        raise ValueError("language must be zh")
    if raw.get("sample_rate") != 16000:
        raise ValueError("sample_rate must be 16000")
    if list(raw.get("scenarios") or []) != list(SCENARIOS):
        raise ValueError(f"scenarios must be {list(SCENARIOS)}")

    dur = raw.get("duration_seconds")
    if not isinstance(dur, dict) or set(dur) != {"min", "max"}:
        raise ValueError("duration_seconds must have min and max")
    dmin, dmax = float(dur["min"]), float(dur["max"])
    if not (0.0 < dmin < dmax):
        raise ValueError("duration bounds must satisfy 0 < min < max")

    splits = raw.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"val", "test"}:
        raise ValueError("splits must have val and test")
    sval, stest = float(splits["val"]), float(splits["test"])
    if sval < 0 or stest < 0 or sval + stest >= 1.0:
        raise ValueError("invalid split fractions")

    quota = raw.get("quota")
    if not isinstance(quota, dict) or set(quota) != {"min_val_rows_per_scenario"}:
        raise ValueError("quota must have min_val_rows_per_scenario")
    min_val = int(quota["min_val_rows_per_scenario"])
    if min_val < 1:
        raise ValueError("quota.min_val_rows_per_scenario must be >= 1")

    slr28 = raw.get("slr28")
    if not isinstance(slr28, dict):
        raise ValueError("slr28 must be a mapping")
    slr_allowed = {"archive_name", "archive_sha256", "license", "split_fractions", "max_rir_seconds", "direct_peak_pre_samples"}
    if set(slr28) != slr_allowed:
        raise ValueError(f"slr28 keys must be {sorted(slr_allowed)}")
    archive_sha = str(slr28["archive_sha256"])
    if len(archive_sha) != 64 or any(c not in "0123456789abcdef" for c in archive_sha):
        raise ValueError("slr28.archive_sha256 must be 64 lowercase hex characters")
    sf_block = slr28["split_fractions"]
    if not isinstance(sf_block, dict) or set(sf_block) != {"val", "test"}:
        raise ValueError("slr28.split_fractions must have val and test")
    slr_val, slr_test = float(sf_block["val"]), float(sf_block["test"])
    if not (0.0 < slr_val < 0.5 and 0.0 < slr_test < 0.5 and slr_val + slr_test < 0.8):
        raise ValueError("slr28.split_fractions out of range")
    max_rir_sec = float(slr28["max_rir_seconds"])
    pre_samples = int(slr28["direct_peak_pre_samples"])
    if max_rir_sec <= 0.05 or pre_samples < 0:
        raise ValueError("invalid slr28 RIR trimming settings")

    aug = raw.get("augmentation")
    if not isinstance(aug, dict) or set(aug) != {"noise", "reverb", "far_field", "dropout"}:
        raise ValueError("augmentation must contain noise, reverb, far_field, dropout")

    noise_cfg = aug["noise"]
    if not isinstance(noise_cfg, dict) or set(noise_cfg) != {"snr_db_min", "snr_db_max", "isotropic_prob"}:
        raise ValueError("invalid augmentation.noise fields")
    if not (float(noise_cfg["snr_db_min"]) < float(noise_cfg["snr_db_max"])):
        raise ValueError("noise snr_db_min must be < snr_db_max")
    if not (0.0 <= float(noise_cfg["isotropic_prob"]) <= 1.0):
        raise ValueError("noise isotropic_prob must be in [0, 1]")

    reverb_cfg = aug["reverb"]
    if not isinstance(reverb_cfg, dict) or set(reverb_cfg) != {"real_rir_prob", "sim_room_sizes"}:
        raise ValueError("invalid augmentation.reverb fields")
    if not (0.0 <= float(reverb_cfg["real_rir_prob"]) <= 1.0):
        raise ValueError("reverb real_rir_prob must be in [0, 1]")
    if not isinstance(reverb_cfg["sim_room_sizes"], list) or not reverb_cfg["sim_room_sizes"]:
        raise ValueError("reverb sim_room_sizes must be a non-empty list")
    for sz in reverb_cfg["sim_room_sizes"]:
        if sz not in VALID_SIM_ROOM_SIZES:
            raise ValueError(f"unknown sim_room_size: {sz}")

    ff_cfg = aug["far_field"]
    ff_allowed = {"real_rir_prob", "sim_room_sizes", "snr_db_min", "snr_db_max", "isotropic_prob", "gain_min", "gain_max"}
    if not isinstance(ff_cfg, dict) or set(ff_cfg) != ff_allowed:
        raise ValueError("invalid augmentation.far_field fields")
    if not (0.0 <= float(ff_cfg["real_rir_prob"]) <= 1.0):
        raise ValueError("far_field real_rir_prob must be in [0, 1]")
    if not (float(ff_cfg["snr_db_min"]) < float(ff_cfg["snr_db_max"])):
        raise ValueError("far_field snr_db_min must be < snr_db_max")
    if not (0.0 <= float(ff_cfg["isotropic_prob"]) <= 1.0):
        raise ValueError("far_field isotropic_prob must be in [0, 1]")
    if not (0.0 < float(ff_cfg["gain_min"]) < float(ff_cfg["gain_max"]) <= 1.0):
        raise ValueError("far_field gain bounds must satisfy 0 < gain_min < gain_max <= 1")
    for sz in ff_cfg["sim_room_sizes"]:
        if sz not in VALID_SIM_ROOM_SIZES:
            raise ValueError(f"unknown far_field sim_room_size: {sz}")

    drop_cfg = aug["dropout"]
    if not isinstance(drop_cfg, dict) or set(drop_cfg) != {"start_ratio", "width_ratio"}:
        raise ValueError("invalid augmentation.dropout fields")
    sr, wr = float(drop_cfg["start_ratio"]), float(drop_cfg["width_ratio"])
    if sr < 0.0 or wr <= 0.0 or sr + wr >= 1.0:
        raise ValueError("dropout span must leave audio")

    return {
        "schema_version": SLR28_DATA_SCHEMA,
        "seed": seed,
        "source_version": source_version,
        "audited": audited,
        "audio_processor_version": SLR28_PROCESSOR_VERSION,
        "language": "zh",
        "sample_rate": 16000,
        "duration_seconds": {"min": dmin, "max": dmax},
        "min_frames": round(dmin * 16000),
        "max_frames": round(dmax * 16000),
        "splits": {"val": sval, "test": stest},
        "min_val_rows_per_scenario": min_val,
        "slr28": {
            "archive_name": str(slr28["archive_name"]),
            "archive_sha256": archive_sha,
            "license": str(slr28["license"]),
            "split_fractions": {"val": slr_val, "test": slr_test},
            "max_rir_seconds": max_rir_sec,
            "direct_peak_pre_samples": pre_samples,
        },
        "augmentation": {
            "noise": {
                "snr_db_min": float(noise_cfg["snr_db_min"]),
                "snr_db_max": float(noise_cfg["snr_db_max"]),
                "isotropic_prob": float(noise_cfg["isotropic_prob"]),
            },
            "reverb": {
                "real_rir_prob": float(reverb_cfg["real_rir_prob"]),
                "sim_room_sizes": list(reverb_cfg["sim_room_sizes"]),
            },
            "far_field": {
                "real_rir_prob": float(ff_cfg["real_rir_prob"]),
                "sim_room_sizes": list(ff_cfg["sim_room_sizes"]),
                "snr_db_min": float(ff_cfg["snr_db_min"]),
                "snr_db_max": float(ff_cfg["snr_db_max"]),
                "isotropic_prob": float(ff_cfg["isotropic_prob"]),
                "gain_min": float(ff_cfg["gain_min"]),
                "gain_max": float(ff_cfg["gain_max"]),
            },
            "dropout": {"start_ratio": sr, "width_ratio": wr},
        },
        "config_sha256": sha256_file(path),
    }


class Slr28AssetCatalog:
    """Zero-leakage split index of OpenSLR 28 RIRs and noises backed by rirs_noises.zip."""

    def __init__(
        self,
        archive_path: Path,
        *,
        seed: int,
        val_fraction: float,
        test_fraction: float,
    ) -> None:
        self.archive_path = Path(archive_path)
        self.seed = int(seed)
        self.val_fraction = float(val_fraction)
        self.test_fraction = float(test_fraction)
        self.real_rirs: dict[str, list[Slr28RirEntry]] = {s: [] for s in SPLITS}
        self.sim_rirs: dict[str, dict[str, list[Slr28RirEntry]]] = {
            s: {sz: [] for sz in VALID_SIM_ROOM_SIZES} for s in SPLITS
        }
        self.isotropic_noises: dict[str, list[Slr28NoiseEntry]] = {s: [] for s in SPLITS}
        self.pointsource_noises: dict[str, list[Slr28NoiseEntry]] = {s: [] for s in SPLITS}
        self.audit: dict[str, Any] = {}
        self._build_index()

    def _build_index(self) -> None:
        with zipfile.ZipFile(self.archive_path, "r") as zf:
            members = set(zf.namelist())
            real_rir_raw: list[tuple[str, str, str, str]] = []
            for line in zf.read("RIRS_NOISES/real_rirs_isotropic_noises/rir_list").decode("utf-8").splitlines():
                tokens = line.strip().split()
                if not tokens:
                    continue
                rir_id = _parse_flag_value(tokens, "--rir-id")
                room_id = _parse_flag_value(tokens, "--room-id")
                member = tokens[-1]
                if member not in members:
                    raise ValueError(f"missing SLR28 real RIR member: {member}")
                real_rir_raw.append((rir_id, room_id, canonical_real_room_family(room_id), member))

            iso_raw: list[tuple[str, str, str]] = []
            for line in zf.read("RIRS_NOISES/real_rirs_isotropic_noises/noise_list").decode("utf-8").splitlines():
                tokens = line.strip().split()
                if not tokens:
                    continue
                noise_id = _parse_flag_value(tokens, "--noise-id")
                room_link = _parse_flag_value(tokens, "--room-linkage")
                member = tokens[-1]
                if member not in members:
                    raise ValueError(f"missing SLR28 isotropic noise member: {member}")
                iso_raw.append((noise_id, canonical_real_room_family(room_link), member))

            # Partition isotropic-linked real rooms first so every split gets >=1 linked room
            iso_room_families = sorted({fam for _, fam, _ in iso_raw})
            iso_split_map = _partition_groups(
                iso_room_families,
                seed=self.seed,
                domain="slr28-real-iso-room-v1",
                val_fraction=self.val_fraction,
                test_fraction=self.test_fraction,
            )
            other_real_families = sorted({fam for _, _, fam, _ in real_rir_raw if fam not in iso_split_map})
            other_split_map: dict[str, str] = {}
            if len(other_real_families) >= 3:
                other_split_map = _partition_groups(
                    other_real_families,
                    seed=self.seed,
                    domain="slr28-real-other-room-v1",
                    val_fraction=self.val_fraction,
                    test_fraction=self.test_fraction,
                )
            else:
                for fam in other_real_families:
                    other_split_map[fam] = "train"
            real_family_split = {**iso_split_map, **other_split_map}

            for rir_id, room_id, fam, member in real_rir_raw:
                sp = real_family_split[fam]
                self.real_rirs[sp].append(
                    Slr28RirEntry(
                        rir_id=f"real-{rir_id}",
                        room_id=room_id,
                        room_family=fam,
                        member_path=member,
                        kind="real",
                        room_size="real",
                        split=sp,
                    )
                )
            for noise_id, fam, member in iso_raw:
                sp = real_family_split[fam]
                self.isotropic_noises[sp].append(
                    Slr28NoiseEntry(
                        noise_id=f"iso-{noise_id}",
                        noise_group=fam,
                        member_path=member,
                        noise_type="isotropic",
                        split=sp,
                    )
                )

            # Simulated RIRs partitioned per room_size by unique room_id
            sim_room_sets: dict[str, set[str]] = {s: set() for s in SPLITS}
            for size in VALID_SIM_ROOM_SIZES:
                list_path = f"RIRS_NOISES/simulated_rirs/{size}/rir_list"
                if list_path not in members:
                    continue
                sim_raw: list[tuple[str, str, str]] = []
                for line in zf.read(list_path).decode("utf-8").splitlines():
                    tokens = line.strip().split()
                    if not tokens:
                        continue
                    rir_id = _parse_flag_value(tokens, "--rir-id")
                    room_id = _parse_flag_value(tokens, "--room-id")
                    member = tokens[-1]
                    if member not in members:
                        raise ValueError(f"missing SLR28 simulated RIR member: {member}")
                    sim_raw.append((rir_id, room_id, member))
                room_map = _partition_groups(
                    [r[1] for r in sim_raw],
                    seed=self.seed,
                    domain=f"slr28-sim-{size}-v1",
                    val_fraction=self.val_fraction,
                    test_fraction=self.test_fraction,
                )
                for rir_id, room_id, member in sim_raw:
                    sp = room_map[room_id]
                    sim_room_sets[sp].add(room_id)
                    self.sim_rirs[sp][size].append(
                        Slr28RirEntry(
                            rir_id=rir_id,
                            room_id=room_id,
                            room_family=room_id,
                            member_path=member,
                            kind="simulated",
                            room_size=size,
                            split=sp,
                        )
                    )

            # Point-source noises partitioned by unique noise_id
            ps_raw: list[tuple[str, str]] = []
            for line in zf.read("RIRS_NOISES/pointsource_noises/noise_list").decode("utf-8").splitlines():
                tokens = line.strip().split()
                if not tokens:
                    continue
                noise_id = _parse_flag_value(tokens, "--noise-id")
                member = tokens[-1]
                if member not in members:
                    raise ValueError(f"missing SLR28 point-source noise member: {member}")
                ps_raw.append((noise_id, member))
            ps_map = _partition_groups(
                [p[0] for p in ps_raw],
                seed=self.seed,
                domain="slr28-pointsource-noise-v1",
                val_fraction=self.val_fraction,
                test_fraction=self.test_fraction,
            )
            for noise_id, member in ps_raw:
                sp = ps_map[noise_id]
                self.pointsource_noises[sp].append(
                    Slr28NoiseEntry(
                        noise_id=noise_id,
                        noise_group=noise_id,
                        member_path=member,
                        noise_type="point-source",
                        split=sp,
                    )
                )

        # Deterministic sort in every split bucket
        for sp in SPLITS:
            self.real_rirs[sp].sort(key=lambda e: e.rir_id)
            for sz in VALID_SIM_ROOM_SIZES:
                self.sim_rirs[sp][sz].sort(key=lambda e: e.rir_id)
            self.isotropic_noises[sp].sort(key=lambda e: e.noise_id)
            self.pointsource_noises[sp].sort(key=lambda e: e.noise_id)

        # Verify strict zero intersection across splits
        real_sets = {sp: {e.room_family for e in self.real_rirs[sp]} | {e.noise_group for e in self.isotropic_noises[sp]} for sp in SPLITS}
        ps_sets = {sp: {e.noise_id for e in self.pointsource_noises[sp]} for sp in SPLITS}
        leaks: dict[str, list[str]] = {}
        for domain_name, group_sets in (
            ("real_room_families", real_sets),
            ("simulated_rooms", sim_room_sets),
            ("pointsource_noises", ps_sets),
        ):
            for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
                inter = sorted(group_sets[left] & group_sets[right])
                leaks[f"{domain_name}:{left}_{right}"] = inter
                if inter:
                    raise ValueError(f"SLR28 split leakage in {domain_name} ({left}/{right}): {inter[:5]}")

        self.audit = {
            "schema_version": SLR28_SPLIT_AUDIT_SCHEMA,
            "seed": self.seed,
            "val_fraction": self.val_fraction,
            "test_fraction": self.test_fraction,
            "counts": {
                sp: {
                    "real_rirs": len(self.real_rirs[sp]),
                    "real_room_families": len(real_sets[sp]),
                    "simulated_rirs": {sz: len(self.sim_rirs[sp][sz]) for sz in VALID_SIM_ROOM_SIZES},
                    "simulated_rooms": len(sim_room_sets[sp]),
                    "isotropic_noises": len(self.isotropic_noises[sp]),
                    "pointsource_noises": len(self.pointsource_noises[sp]),
                }
                for sp in SPLITS
            },
            "intersections": leaks,
            "zero_leakage": all(len(v) == 0 for v in leaks.values()),
        }


def _read_zip_wav_float64(
    zf: zipfile.ZipFile,
    member_path: str,
    channel_seed: int,
) -> tuple[np.ndarray, int]:
    """Read a WAV from OpenSLR 28 zip as 16 kHz mono float64, selecting channel deterministically."""
    raw_bytes = zf.read(member_path)
    data, sr = sf.read(io.BytesIO(raw_bytes), dtype="float64", always_2d=False)
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 2:
        ch = int(channel_seed % arr.shape[1])
        mono = arr[:, ch]
    else:
        mono = arr
        ch = 0
    if int(sr) != 16000:
        mono = np.asarray(to_mono_16k(mono.astype(np.float32), int(sr)), dtype=np.float64)
    return mono, ch


def _prepare_rir_float64(
    rir: np.ndarray,
    *,
    max_rir_samples: int,
    pre_samples: int,
) -> np.ndarray:
    """Direct-path align, trim tail, and L2-normalize an RIR in float64."""
    arr = np.asarray(rir, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("empty_rir")
    peak_idx = int(np.argmax(np.abs(arr)))
    start = max(0, peak_idx - int(pre_samples))
    stop = min(arr.size, peak_idx + int(max_rir_samples))
    trimmed = arr[start:stop].copy()
    energy = float(np.sqrt(np.sum(np.square(trimmed, dtype=np.float64))))
    if not math.isfinite(energy) or energy <= 1e-12:
        raise ValueError("zero_energy_rir")
    return trimmed / energy


def convolve_rir_float64(
    clean: np.ndarray,
    rir: np.ndarray,
    *,
    max_rir_samples: int = 24000,
    pre_samples: int = 16,
) -> np.ndarray:
    """Deterministic float64 FFT convolution with direct-path alignment and RMS matching."""
    x = np.asarray(clean, dtype=np.float64)
    h = _prepare_rir_float64(rir, max_rir_samples=max_rir_samples, pre_samples=pre_samples)
    n_out = x.size + h.size - 1
    n_fft = 1 << (n_out - 1).bit_length()
    y_full = np.fft.irfft(np.fft.rfft(x, n=n_fft) * np.fft.rfft(h, n=n_fft), n=n_fft)
    y = y_full[: x.size]
    x_rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    y_rms = float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))
    if y_rms > 1e-12 and x_rms > 1e-12:
        y = y * (x_rms / y_rms)
    return _soft_limit_float64(y)


def mix_real_noise_float64(
    speech: np.ndarray,
    noise: np.ndarray,
    *,
    snr_db: float,
    offset_seed: int,
) -> np.ndarray:
    """Deterministic float64 additive mixing of real noise at exact target SNR (dB)."""
    x = np.asarray(speech, dtype=np.float64)
    n = np.asarray(noise, dtype=np.float64)
    if n.size == 0:
        raise ValueError("empty_noise")
    # Loop noise if shorter than speech, then crop at deterministic offset
    if n.size < x.size:
        reps = int(math.ceil(x.size / n.size)) + 1
        n = np.tile(n, reps)
    max_start = n.size - x.size
    start = int(offset_seed % (max_start + 1)) if max_start > 0 else 0
    seg = n[start : start + x.size].copy()
    seg = seg - float(np.mean(seg, dtype=np.float64))
    sig_pow = float(np.mean(np.square(x, dtype=np.float64)))
    noi_pow = float(np.mean(np.square(seg, dtype=np.float64)))
    if sig_pow <= 1e-15 or noi_pow <= 1e-15:
        return _soft_limit_float64(x)
    target_noi_pow = sig_pow / (10.0 ** (float(snr_db) / 10.0))
    scale = math.sqrt(target_noi_pow / noi_pow)
    return _soft_limit_float64(x + seg * scale)


def apply_dropout_float64(clean: np.ndarray, *, start_ratio: float, width_ratio: float) -> np.ndarray:
    audio = np.array(clean, dtype=np.float64, copy=True)
    start = int(audio.size * float(start_ratio))
    stop = int(audio.size * (float(start_ratio) + float(width_ratio)))
    stop = min(audio.size, max(start + 1, stop))
    audio[start:stop] = 0.0
    return audio


def _soft_limit_float64(audio: np.ndarray, ceiling: float = 0.99) -> np.ndarray:
    peak = float(np.max(np.abs(audio)))
    if peak > ceiling:
        audio = audio * (ceiling / peak)
    return np.clip(audio, -1.0, 1.0)


def _select_rir(
    catalog: Slr28AssetCatalog,
    *,
    split: str,
    seed: int,
    source_id: str,
    scenario: str,
    real_rir_prob: float,
    sim_room_sizes: list[str],
) -> Slr28RirEntry:
    use_real = _hash_unit(seed, source_id, f"{scenario}:rir-kind") < float(real_rir_prob)
    real_pool = catalog.real_rirs[split]
    sim_pool: list[Slr28RirEntry] = []
    for sz in sim_room_sizes:
        sim_pool.extend(catalog.sim_rirs[split][sz])
    if use_real and real_pool:
        pool = real_pool
    elif sim_pool:
        pool = sim_pool
    elif real_pool:
        pool = real_pool
    else:
        raise ValueError(f"no SLR28 RIR available for split={split} scenario={scenario}")
    idx = _deterministic_int(seed, source_id, f"{scenario}:rir-idx") % len(pool)
    return pool[idx]


def _select_noise(
    catalog: Slr28AssetCatalog,
    *,
    split: str,
    seed: int,
    source_id: str,
    scenario: str,
    isotropic_prob: float,
) -> Slr28NoiseEntry:
    use_iso = _hash_unit(seed, source_id, f"{scenario}:noise-kind") < float(isotropic_prob)
    iso_pool = catalog.isotropic_noises[split]
    ps_pool = catalog.pointsource_noises[split]
    if use_iso and iso_pool:
        pool = iso_pool
    elif ps_pool:
        pool = ps_pool
    elif iso_pool:
        pool = iso_pool
    else:
        raise ValueError(f"no SLR28 noise available for split={split} scenario={scenario}")
    idx = _deterministic_int(seed, source_id, f"{scenario}:noise-idx") % len(pool)
    return pool[idx]


def render_slr28_scenarios(
    clean: np.ndarray,
    *,
    seed: int,
    source_id: str,
    split: str,
    config: dict[str, Any],
    catalog: Slr28AssetCatalog,
    zf: zipfile.ZipFile,
    wav_cache: dict[tuple[str, int], np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    """Render all 5 scenarios in float64 using only assets from the source's own split."""
    if split not in SPLITS:
        raise ValueError(f"invalid split: {split}")
    x = np.asarray(clean, dtype=np.float64)
    aug = config["augmentation"]
    max_rir_samples = int(round(float(config["slr28"]["max_rir_seconds"]) * 16000))
    pre_samples = int(config["slr28"]["direct_peak_pre_samples"])

    def load_asset(member_path: str, ch_seed: int) -> tuple[np.ndarray, int]:
        # Cache raw bytes read per (member_path, ch_seed % 16) is tiny if only channel matters
        raw_bytes = zf.read(member_path)
        data, sr = sf.read(io.BytesIO(raw_bytes), dtype="float64", always_2d=False)
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim == 2:
            ch = int(ch_seed % arr.shape[1])
            mono = arr[:, ch]
        else:
            ch = 0
            mono = arr
        if int(sr) != 16000:
            mono = np.asarray(to_mono_16k(mono.astype(np.float32), int(sr)), dtype=np.float64)
        return mono, ch

    def get_asset(member_path: str, ch_seed: int) -> tuple[np.ndarray, int]:
        if wav_cache is None:
            return load_asset(member_path, ch_seed)
        # First probe channel count via cache key (member_path, -1) if unknown
        meta_key = (member_path, -1)
        if meta_key in wav_cache:
            n_ch = int(wav_cache[meta_key][0])
            ch = int(ch_seed % n_ch)
            ckey = (member_path, ch)
            if ckey in wav_cache:
                return wav_cache[ckey], ch
        raw_bytes = zf.read(member_path)
        data, sr = sf.read(io.BytesIO(raw_bytes), dtype="float64", always_2d=False)
        arr = np.asarray(data, dtype=np.float64)
        n_ch = arr.shape[1] if arr.ndim == 2 else 1
        ch = int(ch_seed % n_ch)
        mono = arr[:, ch] if arr.ndim == 2 else arr
        if int(sr) != 16000:
            mono = np.asarray(to_mono_16k(mono.astype(np.float32), int(sr)), dtype=np.float64)
        if len(wav_cache) < 4096:
            wav_cache[meta_key] = np.array([n_ch], dtype=np.int64)
            wav_cache[(member_path, ch)] = mono
        return mono, ch

    waves: dict[str, np.ndarray] = {"clean": _soft_limit_float64(x, ceiling=1.0)}
    meta: dict[str, dict[str, Any]] = {}

    # 1. noise: real SLR28 point-source / isotropic noise at [snr_min, snr_max]
    n_cfg = aug["noise"]
    n_entry = _select_noise(
        catalog,
        split=split,
        seed=seed,
        source_id=source_id,
        scenario="noise",
        isotropic_prob=float(n_cfg["isotropic_prob"]),
    )
    n_ch_seed = _deterministic_int(seed, source_id, "noise:ch")
    n_off_seed = _deterministic_int(seed, source_id, "noise:offset")
    n_snr = round(
        float(n_cfg["snr_db_min"])
        + _hash_unit(seed, source_id, "noise:snr") * (float(n_cfg["snr_db_max"]) - float(n_cfg["snr_db_min"])),
        3,
    )
    n_wave, n_ch = get_asset(n_entry.member_path, n_ch_seed)
    waves["noise"] = mix_real_noise_float64(x, n_wave, snr_db=n_snr, offset_seed=n_off_seed)
    meta["noise"] = {
        "type": "noise",
        "processor_version": SLR28_PROCESSOR_VERSION,
        "noise_id": n_entry.noise_id,
        "noise_group": n_entry.noise_group,
        "noise_type": n_entry.noise_type,
        "noise_channel": n_ch,
        "snr_db": n_snr,
        "offset_seed": n_off_seed,
    }

    # 2. reverb: pure real + medium/large simulated RIR convolution
    r_cfg = aug["reverb"]
    r_entry = _select_rir(
        catalog,
        split=split,
        seed=seed,
        source_id=source_id,
        scenario="reverb",
        real_rir_prob=float(r_cfg["real_rir_prob"]),
        sim_room_sizes=list(r_cfg["sim_room_sizes"]),
    )
    r_ch_seed = _deterministic_int(seed, source_id, "reverb:ch")
    r_wave, r_ch = get_asset(r_entry.member_path, r_ch_seed)
    waves["reverb"] = convolve_rir_float64(
        x,
        r_wave,
        max_rir_samples=max_rir_samples,
        pre_samples=pre_samples,
    )
    meta["reverb"] = {
        "type": "reverb",
        "processor_version": SLR28_PROCESSOR_VERSION,
        "rir_id": r_entry.rir_id,
        "rir_room_id": r_entry.room_id,
        "rir_room_family": r_entry.room_family,
        "rir_kind": r_entry.kind,
        "rir_room_size": r_entry.room_size,
        "rir_channel": r_ch,
    }

    # 3. far_field: distant medium/large RIR + real room ambient/point-source noise + gain attenuation
    f_cfg = aug["far_field"]
    f_rir = _select_rir(
        catalog,
        split=split,
        seed=seed,
        source_id=source_id,
        scenario="far_field",
        real_rir_prob=float(f_cfg["real_rir_prob"]),
        sim_room_sizes=list(f_cfg["sim_room_sizes"]),
    )
    f_rir_ch_seed = _deterministic_int(seed, source_id, "far_field:rir_ch")
    f_rir_wave, f_rir_ch = get_asset(f_rir.member_path, f_rir_ch_seed)
    reverb_ff = convolve_rir_float64(
        x,
        f_rir_wave,
        max_rir_samples=max_rir_samples,
        pre_samples=pre_samples,
    )
    f_noise = _select_noise(
        catalog,
        split=split,
        seed=seed,
        source_id=source_id,
        scenario="far_field",
        isotropic_prob=float(f_cfg["isotropic_prob"]),
    )
    f_n_ch_seed = _deterministic_int(seed, source_id, "far_field:noise_ch")
    f_n_off_seed = _deterministic_int(seed, source_id, "far_field:noise_offset")
    f_snr = round(
        float(f_cfg["snr_db_min"])
        + _hash_unit(seed, source_id, "far_field:snr") * (float(f_cfg["snr_db_max"]) - float(f_cfg["snr_db_min"])),
        3,
    )
    f_gain = round(
        float(f_cfg["gain_min"])
        + _hash_unit(seed, source_id, "far_field:gain") * (float(f_cfg["gain_max"]) - float(f_cfg["gain_min"])),
        4,
    )
    f_n_wave, f_n_ch = get_asset(f_noise.member_path, f_n_ch_seed)
    noisy_ff = mix_real_noise_float64(reverb_ff, f_n_wave, snr_db=f_snr, offset_seed=f_n_off_seed)
    waves["far_field"] = _soft_limit_float64(noisy_ff * f_gain)
    meta["far_field"] = {
        "type": "far_field",
        "processor_version": SLR28_PROCESSOR_VERSION,
        "rir_id": f_rir.rir_id,
        "rir_room_id": f_rir.room_id,
        "rir_room_family": f_rir.room_family,
        "rir_kind": f_rir.kind,
        "rir_room_size": f_rir.room_size,
        "rir_channel": f_rir_ch,
        "noise_id": f_noise.noise_id,
        "noise_group": f_noise.noise_group,
        "noise_type": f_noise.noise_type,
        "noise_channel": f_n_ch,
        "snr_db": f_snr,
        "gain": f_gain,
        "offset_seed": f_n_off_seed,
    }

    # 4. dropout: deterministic interior silence guard
    d_cfg = aug["dropout"]
    waves["dropout"] = apply_dropout_float64(
        x,
        start_ratio=float(d_cfg["start_ratio"]),
        width_ratio=float(d_cfg["width_ratio"]),
    )
    meta["dropout"] = {
        "type": "dropout",
        "processor_version": SLR28_PROCESSOR_VERSION,
        "start_ratio": float(d_cfg["start_ratio"]),
        "width_ratio": float(d_cfg["width_ratio"]),
    }

    return waves, meta
