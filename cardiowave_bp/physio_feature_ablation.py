from __future__ import annotations

import argparse
import json
import logging
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.ensemble import RandomForestRegressor

from .pipeline import (
    RunConfig,
    apply_filters,
    clean_records,
    configure_logging,
    current_timestamp,
    load_raw_dataframe,
    set_seed,
    stratified_holdout_split,
    subject_summary,
    to_serializable,
    zscore,
)
from .tabpfn_regression import META_COLUMNS, signal_features

logger = logging.getLogger(__name__)

PHYSIO_FEATURE_SETS = [
    "meta_only",
    "ecg_ppg_basic",
    "physio_only",
    "ecg_ppg_basic_physio",
    "ecg_ppg_basic_meta",
    "ecg_ppg_basic_physio_meta",
]

DEFAULT_MODELS = ["catboost", "lightgbm", "xgboost", "randomforest"]


@dataclass
class PhysioAblationConfig:
    data_path: str = "data/cardiowave_portable_dataset.pkl"
    split_path: str = "resources/manuscript_subject_level_split.csv"
    output_root: str = "outputs/01_physio_feature_ablation"
    batch_id: str = "physio_feature_v1"
    seed: int = 42
    holdout_ratio: float = 0.20
    n_folds: int = 5
    sampling_rate: int = 500
    apply_filter: bool = True
    ecg_clean_method: str = "neurokit"
    ppg_clean_method: str = "elgendi"
    feature_sets: str = ",".join(PHYSIO_FEATURE_SETS)
    models: str = ",".join(DEFAULT_MODELS)
    max_records: int = 0
    n_jobs: int = 4
    run_cv: bool = True
    # Landmark parameters. These defaults are deliberately conservative.
    ecg_min_distance_sec: float = 0.30
    ppg_min_distance_sec: float = 0.35
    min_pat_sec: float = 0.08
    max_pat_sec: float = 0.80
    ppg_foot_lookback_sec: float = 0.40
    ppg_decay_lookahead_sec: float = 0.80
    min_pulse_amp_z: float = 0.05

    @property
    def output_dir(self) -> Path:
        return Path(self.output_root) / self.batch_id

    def feature_set_list(self) -> list[str]:
        items = [x.strip() for x in str(self.feature_sets).split(",") if x.strip()]
        bad = [x for x in items if x not in PHYSIO_FEATURE_SETS]
        if bad:
            raise ValueError(f"Unsupported feature_sets={bad}. Supported={PHYSIO_FEATURE_SETS}")
        return items

    def model_list(self) -> list[str]:
        items = [x.strip().lower() for x in str(self.models).split(",") if x.strip()]
        supported = set(DEFAULT_MODELS)
        bad = [x for x in items if x not in supported]
        if bad:
            raise ValueError(f"Unsupported models={bad}. Supported={sorted(supported)}")
        return items


def parse_bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    value = str(v).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse bool from {v!r}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Physiological ECG-PPG fusion feature ablation for BP regression.")
    p.add_argument("--data-path", default=PhysioAblationConfig.data_path)
    p.add_argument("--split-path", default=PhysioAblationConfig.split_path)
    p.add_argument("--output-root", default=PhysioAblationConfig.output_root)
    p.add_argument("--batch-id", default=PhysioAblationConfig.batch_id)
    p.add_argument("--seed", type=int, default=PhysioAblationConfig.seed)
    p.add_argument("--holdout-ratio", type=float, default=PhysioAblationConfig.holdout_ratio)
    p.add_argument("--n-folds", type=int, default=PhysioAblationConfig.n_folds)
    p.add_argument("--sampling-rate", type=int, default=PhysioAblationConfig.sampling_rate)
    p.add_argument("--apply-filter", type=parse_bool, default=PhysioAblationConfig.apply_filter)
    p.add_argument("--ecg-clean-method", default=PhysioAblationConfig.ecg_clean_method)
    p.add_argument("--ppg-clean-method", default=PhysioAblationConfig.ppg_clean_method)
    p.add_argument("--feature-sets", default=PhysioAblationConfig.feature_sets)
    p.add_argument("--models", default=PhysioAblationConfig.models)
    p.add_argument("--max-records", type=int, default=PhysioAblationConfig.max_records)
    p.add_argument("--n-jobs", type=int, default=PhysioAblationConfig.n_jobs)
    p.add_argument("--run-cv", type=parse_bool, default=PhysioAblationConfig.run_cv)
    p.add_argument("--ecg-min-distance-sec", type=float, default=PhysioAblationConfig.ecg_min_distance_sec)
    p.add_argument("--ppg-min-distance-sec", type=float, default=PhysioAblationConfig.ppg_min_distance_sec)
    p.add_argument("--min-pat-sec", type=float, default=PhysioAblationConfig.min_pat_sec)
    p.add_argument("--max-pat-sec", type=float, default=PhysioAblationConfig.max_pat_sec)
    p.add_argument("--ppg-foot-lookback-sec", type=float, default=PhysioAblationConfig.ppg_foot_lookback_sec)
    p.add_argument("--ppg-decay-lookahead-sec", type=float, default=PhysioAblationConfig.ppg_decay_lookahead_sec)
    p.add_argument("--min-pulse-amp-z", type=float, default=PhysioAblationConfig.min_pulse_amp_z)
    return p


def finite_arr(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def stat_block(values: Any, prefix: str) -> dict[str, float]:
    arr = finite_arr(values)
    keys = ["count", "mean", "std", "median", "q25", "q75", "iqr", "min", "max"]
    out = {f"{prefix}_{k}": math.nan for k in keys}
    out[f"{prefix}_count"] = float(len(arr))
    if len(arr) == 0:
        return out
    q25, q75 = np.quantile(arr, [0.25, 0.75])
    out.update({
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_q25": float(q25),
        f"{prefix}_q75": float(q75),
        f"{prefix}_iqr": float(q75 - q25),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
    })
    return out


def detect_ecg_r_peaks(ecg: np.ndarray, fs: int, cfg: PhysioAblationConfig) -> np.ndarray:
    ecg = np.asarray(ecg, dtype=float)
    if len(ecg) < fs:
        return np.asarray([], dtype=int)
    try:
        import neurokit2 as nk
        _, info = nk.ecg_peaks(ecg, sampling_rate=fs, method="neurokit")
        peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
        peaks = peaks[(peaks >= 0) & (peaks < len(ecg))]
        if len(peaks) >= 2:
            return peaks
    except Exception:
        pass
    x = zscore(ecg)
    distance = max(1, int(cfg.ecg_min_distance_sec * fs))
    prominence = max(float(np.std(x)) * 0.25, 1e-6)
    try:
        peaks, _ = find_peaks(x, distance=distance, prominence=prominence)
    except Exception:
        peaks = np.asarray([], dtype=int)
    return peaks.astype(int)


def detect_ppg_landmarks(ppg: np.ndarray, fs: int, cfg: PhysioAblationConfig) -> dict[str, np.ndarray]:
    """Detect approximate PPG landmarks on an inverted cleaned PPG waveform.

    The project data has historically used -PPG as the positive pulse direction.
    Landmarks are intentionally conservative and are meant as robust approximate
    physiological descriptors, not beat-level clinical annotations.
    """
    pulse = zscore(-np.asarray(ppg, dtype=float))
    n = len(pulse)
    empty = {k: np.asarray([], dtype=float if k != "peaks" else int) for k in [
        "peaks", "feet", "slope_points", "next_feet", "amplitude", "rise_time", "decay_time",
        "width50", "rise_area", "decay_area", "total_area", "interval",
    ]}
    if n < fs:
        return empty
    distance = max(1, int(cfg.ppg_min_distance_sec * fs))
    prominence = max(float(np.std(pulse)) * 0.15, 1e-6)
    try:
        peaks, _ = find_peaks(pulse, distance=distance, prominence=prominence)
    except Exception:
        peaks = np.asarray([], dtype=int)
    peaks = peaks.astype(int)
    if len(peaks) == 0:
        return empty

    feet, slope_points, next_feet = [], [], []
    amps, rises, decays, widths, rise_areas, decay_areas, total_areas = [], [], [], [], [], [], []
    lookback = int(cfg.ppg_foot_lookback_sec * fs)
    lookahead = int(cfg.ppg_decay_lookahead_sec * fs)
    for i, pk in enumerate(peaks):
        start = max(0, pk - lookback)
        # Avoid using a segment that crosses the previous peak too much.
        if i > 0:
            start = max(start, int((peaks[i - 1] + pk) // 2))
        pre = pulse[start: pk + 1]
        if len(pre) < 3:
            continue
        foot = int(start + np.argmin(pre))
        amp = float(pulse[pk] - pulse[foot])
        if not np.isfinite(amp) or amp < cfg.min_pulse_amp_z:
            continue
        seg_rise = pulse[foot: pk + 1]
        if len(seg_rise) > 2:
            diff = np.diff(seg_rise)
            slope = int(foot + np.argmax(diff))
        else:
            slope = foot
        if i + 1 < len(peaks):
            end = min(n, int((pk + peaks[i + 1]) // 2) + 1)
        else:
            end = min(n, pk + lookahead)
        post = pulse[pk:end]
        if len(post) > 2:
            nf = int(pk + np.argmin(post))
        else:
            nf = min(n - 1, pk + 1)

        # Half-amplitude width around the systolic peak.
        half = float(pulse[foot] + 0.5 * amp)
        left = foot
        for j in range(pk, foot, -1):
            if pulse[j] <= half:
                left = j
                break
        right = nf
        for j in range(pk, nf):
            if pulse[j] <= half:
                right = j
                break
        width50 = max(0.0, (right - left) / float(fs))

        feet.append(foot)
        slope_points.append(slope)
        next_feet.append(nf)
        amps.append(amp)
        rises.append((pk - foot) / float(fs))
        decays.append(max(0.0, (nf - pk) / float(fs)))
        widths.append(width50)
        # Areas use baseline at foot level; clip below zero for robustness.
        rise_y = np.maximum(pulse[foot: pk + 1] - pulse[foot], 0.0)
        decay_y = np.maximum(pulse[pk: nf + 1] - pulse[nf], 0.0)
        rise_areas.append(float(np.trapz(rise_y, dx=1.0 / fs)) if len(rise_y) else math.nan)
        decay_areas.append(float(np.trapz(decay_y, dx=1.0 / fs)) if len(decay_y) else math.nan)
        total_areas.append(float(rise_areas[-1] + decay_areas[-1]) if np.isfinite(rise_areas[-1]) and np.isfinite(decay_areas[-1]) else math.nan)

    intervals = np.diff(peaks) / float(fs) if len(peaks) > 1 else np.asarray([], dtype=float)
    return {
        "peaks": peaks.astype(int),
        "feet": np.asarray(feet, dtype=int),
        "slope_points": np.asarray(slope_points, dtype=int),
        "next_feet": np.asarray(next_feet, dtype=int),
        "amplitude": np.asarray(amps, dtype=float),
        "rise_time": np.asarray(rises, dtype=float),
        "decay_time": np.asarray(decays, dtype=float),
        "width50": np.asarray(widths, dtype=float),
        "rise_area": np.asarray(rise_areas, dtype=float),
        "decay_area": np.asarray(decay_areas, dtype=float),
        "total_area": np.asarray(total_areas, dtype=float),
        "interval": intervals,
    }


def pair_ecg_ppg_timings(r_peaks: np.ndarray, landmarks: dict[str, np.ndarray], fs: int, cfg: PhysioAblationConfig) -> dict[str, np.ndarray]:
    peaks = np.asarray(landmarks.get("peaks", []), dtype=int)
    feet = np.asarray(landmarks.get("feet", []), dtype=int)
    slopes = np.asarray(landmarks.get("slope_points", []), dtype=int)
    if len(r_peaks) == 0 or len(peaks) == 0:
        return {"pat_peak": np.asarray([]), "pat_foot": np.asarray([]), "pat_slope": np.asarray([])}
    min_s = cfg.min_pat_sec
    max_s = cfg.max_pat_sec
    pat_peak, pat_foot, pat_slope = [], [], []
    for r in r_peaks:
        j = int(np.searchsorted(peaks, r + int(min_s * fs), side="left"))
        if j >= len(peaks):
            continue
        pk = int(peaks[j])
        dt_peak = (pk - int(r)) / float(fs)
        if not (min_s <= dt_peak <= max_s):
            continue
        pat_peak.append(dt_peak)
        # The feet/slope arrays may have fewer entries than peaks if pulses were rejected.
        # Use the closest available landmark before the selected peak when possible.
        if len(feet):
            jf = int(np.argmin(np.abs(feet - pk)))
            foot = int(feet[jf])
            dt = (foot - int(r)) / float(fs)
            if min_s <= dt <= max_s:
                pat_foot.append(dt)
        if len(slopes):
            js = int(np.argmin(np.abs(slopes - pk)))
            sp = int(slopes[js])
            dt = (sp - int(r)) / float(fs)
            if min_s <= dt <= max_s:
                pat_slope.append(dt)
    return {
        "pat_peak": np.asarray(pat_peak, dtype=float),
        "pat_foot": np.asarray(pat_foot, dtype=float),
        "pat_slope": np.asarray(pat_slope, dtype=float),
    }


def physiological_features(ecg: np.ndarray, ppg: np.ndarray, fs: int, cfg: PhysioAblationConfig) -> dict[str, float]:
    out: dict[str, float] = {}
    r_peaks = detect_ecg_r_peaks(ecg, fs, cfg)
    landmarks = detect_ppg_landmarks(ppg, fs, cfg)
    pairs = pair_ecg_ppg_timings(r_peaks, landmarks, fs, cfg)

    duration_min = len(ecg) / float(fs) / 60.0 if len(ecg) else math.nan
    out["phys_ecg_r_count"] = float(len(r_peaks))
    out["phys_ppg_peak_count"] = float(len(landmarks.get("peaks", [])))
    out["phys_pair_peak_count"] = float(len(pairs["pat_peak"]))
    out["phys_pair_foot_count"] = float(len(pairs["pat_foot"]))
    out["phys_pair_slope_count"] = float(len(pairs["pat_slope"]))
    out["phys_pair_peak_rate"] = float(len(pairs["pat_peak"]) / max(len(r_peaks), 1))
    out["phys_pair_foot_rate"] = float(len(pairs["pat_foot"]) / max(len(r_peaks), 1))
    out["phys_ecg_hr_bpm"] = float(len(r_peaks) / duration_min) if duration_min and duration_min > 0 else math.nan
    out["phys_ppg_rate_bpm"] = float(len(landmarks.get("peaks", [])) / duration_min) if duration_min and duration_min > 0 else math.nan

    if len(r_peaks) > 1:
        out.update(stat_block(np.diff(r_peaks) / float(fs), "phys_rr_interval_sec"))
    else:
        out.update(stat_block([], "phys_rr_interval_sec"))
    out.update(stat_block(landmarks.get("interval", []), "phys_ppg_interval_sec"))

    # PAT / PWTT-like descriptors. Strictly speaking, without a proximal PPG site this is PAT.
    # We keep pwtt-like aliases for paper readability while documenting it as ECG-to-PPG timing.
    for name, arr in pairs.items():
        suffix = name.replace("pat_", "")
        out.update(stat_block(arr, f"phys_pat_{suffix}_sec"))
        inv = 1.0 / np.asarray(arr, dtype=float) if len(arr) else np.asarray([])
        out.update(stat_block(inv, f"phys_pat_{suffix}_inv"))
        out.update(stat_block(arr, f"phys_pwtt_like_{suffix}_sec"))

    # PPG morphology.
    for k in ["amplitude", "rise_time", "decay_time", "width50", "rise_area", "decay_area", "total_area"]:
        out.update(stat_block(landmarks.get(k, []), f"phys_ppg_{k}"))
    amp = finite_arr(landmarks.get("amplitude", []))
    width = finite_arr(landmarks.get("width50", []))
    rise = finite_arr(landmarks.get("rise_time", []))
    decay = finite_arr(landmarks.get("decay_time", []))
    pat_foot = finite_arr(pairs.get("pat_foot", []))
    out["phys_ppg_rise_decay_ratio_mean"] = float(np.nanmean(rise / np.maximum(decay[: len(rise)], 1e-6))) if len(rise) and len(decay) >= len(rise) else math.nan
    out["phys_amp_width_ratio_mean"] = float(np.nanmean(amp / np.maximum(width[: len(amp)], 1e-6))) if len(amp) and len(width) >= len(amp) else math.nan
    out["phys_hr_x_pat_foot_mean"] = float(out.get("phys_ecg_hr_bpm", math.nan) * np.mean(pat_foot)) if len(pat_foot) else math.nan
    out["phys_hr_x_pat_peak_mean"] = float(out.get("phys_ecg_hr_bpm", math.nan) * np.mean(finite_arr(pairs.get("pat_peak", [])))) if len(finite_arr(pairs.get("pat_peak", []))) else math.nan
    return out


def feature_groups_for_frame(frame: pd.DataFrame) -> dict[str, list[str]]:
    meta = [c for c in META_COLUMNS if c in frame.columns]
    basic = [c for c in frame.columns if c.startswith("ecg_") or c.startswith("ppg_")]
    phys = [c for c in frame.columns if c.startswith("phys_")]
    return {"meta": meta, "basic": basic, "physio": phys}


def feature_columns_for_set(frame: pd.DataFrame, feature_set: str) -> list[str]:
    groups = feature_groups_for_frame(frame)
    if feature_set == "meta_only":
        cols = groups["meta"]
    elif feature_set == "ecg_ppg_basic":
        cols = groups["basic"]
    elif feature_set == "physio_only":
        cols = groups["physio"]
    elif feature_set == "ecg_ppg_basic_physio":
        cols = groups["basic"] + groups["physio"]
    elif feature_set == "ecg_ppg_basic_meta":
        cols = groups["basic"] + groups["meta"]
    elif feature_set == "ecg_ppg_basic_physio_meta":
        cols = groups["basic"] + groups["physio"] + groups["meta"]
    else:
        raise ValueError(f"Unknown feature_set={feature_set}")
    # Preserve order, remove duplicates and non-existent columns.
    seen, out = set(), []
    for c in cols:
        if c in frame.columns and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def build_feature_frame(records_df: pd.DataFrame, cfg: PhysioAblationConfig) -> pd.DataFrame:
    dl_cfg = RunConfig(
        data_path=cfg.data_path,
        seed=cfg.seed,
        holdout_ratio=cfg.holdout_ratio,
        n_folds=cfg.n_folds,
        sampling_rate=cfg.sampling_rate,
        apply_filter=cfg.apply_filter,
        ecg_clean_method=cfg.ecg_clean_method,
        ppg_clean_method=cfg.ppg_clean_method,
    ).finalize()
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(records_df.itertuples(index=False)):
        payload: dict[str, Any] = {
            "record_id": str(row.record_id),
            "subject_id": str(row.subject_id),
            "sbp": float(row.sbp),
            "dbp": float(row.dbp),
        }
        for col in META_COLUMNS:
            if hasattr(row, col):
                payload[col] = getattr(row, col)
        try:
            ecg, ppg = apply_filters(row.ecg, row.ppg, dl_cfg)  # type: ignore[arg-type]
            n = min(len(ecg), len(ppg))
            ecg = np.asarray(ecg[:n], dtype=float)
            ppg = np.asarray(ppg[:n], dtype=float)
            payload.update(signal_features(ecg, cfg.sampling_rate, "ecg"))
            payload.update(signal_features(-ppg, cfg.sampling_rate, "ppg"))
            payload.update(physiological_features(ecg, ppg, cfg.sampling_rate, cfg))
        except Exception as exc:
            logger.warning("Feature extraction failed for record %s: %s", getattr(row, "record_id", i), exc)
        rows.append(payload)
        if (i + 1) % 250 == 0:
            logger.info("Extracted features for %s/%s records", i + 1, len(records_df))
    return pd.DataFrame(rows)



# Strict aligned preprocessing for physiological feature ablation.
# The generic preprocessor inferred type from pandas dtype; in this dataset,
# continuous metadata such as height/weight/BMI may be loaded as object columns and
# incorrectly one-hot encoded. We force clinically continuous metadata and all signal
# / physiological descriptors to be numeric, while only true categorical metadata are
# one-hot encoded.
FORCE_NUMERIC_COLUMNS = {"age_clean", "height_clean", "weight_clean", "bmi", "drug_binary"}
FORCE_CATEGORICAL_COLUMNS = {"sex_clean", "dx_htn_clean", "pre_smoke_clean", "pre_coffee_clean"}
NUMERIC_PREFIXES = ("ecg_", "ppg_", "phys_")

@dataclass
class AlignedTabularPreprocessor:
    numeric_columns: list[str]
    categorical_columns: list[str]
    medians: dict[str, float]
    stds: dict[str, float]
    categories: dict[str, list[str]]

    @classmethod
    def fit(cls, frame: pd.DataFrame, feature_columns: list[str]) -> "AlignedTabularPreprocessor":
        numeric_columns: list[str] = []
        categorical_columns: list[str] = []
        for col in feature_columns:
            if col in FORCE_NUMERIC_COLUMNS or col.startswith(NUMERIC_PREFIXES):
                numeric_columns.append(col)
            elif col in FORCE_CATEGORICAL_COLUMNS:
                categorical_columns.append(col)
            elif pd.api.types.is_numeric_dtype(frame[col]):
                numeric_columns.append(col)
            else:
                categorical_columns.append(col)

        medians: dict[str, float] = {}
        stds: dict[str, float] = {}
        for col in numeric_columns:
            values = pd.to_numeric(frame[col], errors="coerce").astype(float)
            median = float(values.median()) if values.notna().any() else 0.0
            filled = values.fillna(median)
            std = float(filled.std())
            medians[col] = median if math.isfinite(median) else 0.0
            stds[col] = std if math.isfinite(std) and std > 1e-8 else 1.0

        categories = {
            col: sorted(frame[col].fillna("missing").astype(str).unique().tolist())
            for col in categorical_columns
        }
        return cls(numeric_columns, categorical_columns, medians, stds, categories)

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        arrays: list[np.ndarray] = []
        names: list[str] = []
        for col in self.numeric_columns:
            values = pd.to_numeric(frame[col], errors="coerce").astype(float)
            missing = values.isna().astype(float).to_numpy().reshape(-1, 1)
            filled = values.fillna(self.medians[col])
            scaled = ((filled - self.medians[col]) / self.stds[col]).to_numpy().reshape(-1, 1)
            arrays.extend([scaled, missing])
            names.extend([col, f"{col}_missing"])
        for col in self.categorical_columns:
            values = frame[col].fillna("missing").astype(str)
            for category in self.categories[col]:
                arrays.append((values == category).astype(float).to_numpy().reshape(-1, 1))
                names.append(f"{col}_{category}")
        if arrays:
            return np.concatenate(arrays, axis=1).astype(np.float32), names
        return np.zeros((len(frame), 0), dtype=np.float32), names

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_model(name: str, seed: int, n_jobs: int):
    if name == "randomforest":
        return RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=seed, n_jobs=n_jobs)
    if name == "catboost":
        try:
            from catboost import CatBoostRegressor
        except Exception as exc:
            raise RuntimeError("catboost is not installed. Run `pip install catboost` or remove catboost from --models.") from exc
        return CatBoostRegressor(
            iterations=800,
            depth=6,
            learning_rate=0.03,
            loss_function="RMSE",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=n_jobs,
        )
    if name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except Exception as exc:
            raise RuntimeError("lightgbm is not installed. Run `pip install lightgbm` or remove lightgbm from --models.") from exc
        return LGBMRegressor(
            n_estimators=800,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=seed,
            n_jobs=n_jobs,
            verbose=-1,
        )
    if name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except Exception as exc:
            raise RuntimeError("xgboost is not installed. Run `pip install xgboost` or remove xgboost from --models.") from exc
        return XGBRegressor(
            n_estimators=800,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=n_jobs,
            tree_method="hist",
        )
    raise ValueError(f"Unsupported model: {name}")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "residual_me": float(np.mean(residual)),
        "residual_sd": float(np.std(residual)),
    }


def fit_predict_one(train: pd.DataFrame, test: pd.DataFrame, feature_columns: list[str], model_name: str, seed: int, n_jobs: int) -> tuple[pd.DataFrame, list[str], AlignedTabularPreprocessor]:
    preproc = AlignedTabularPreprocessor.fit(train, feature_columns)
    x_train, feature_names = preproc.transform(train)
    x_test, _ = preproc.transform(test)
    out = test[["record_id", "subject_id", "sbp", "dbp"]].copy()
    for target in ["sbp", "dbp"]:
        model = make_model(model_name, seed, n_jobs)
        model.fit(x_train, train[target].to_numpy(dtype=float))
        out[f"{target}_pred"] = model.predict(x_test)
    return out, feature_names, preproc


def evaluate_predictions(pred: pd.DataFrame, split: str, fold: str) -> pd.DataFrame:
    rows = []
    for target in ["sbp", "dbp"]:
        m = regression_metrics(pred[target].to_numpy(dtype=float), pred[f"{target}_pred"].to_numpy(dtype=float))
        rows.append({"split": split, "fold": fold, "target": target.upper(), **m})
    return pd.DataFrame(rows)


def cv_splitter(trainval: pd.DataFrame, cfg: PhysioAblationConfig):
    subject_df = subject_summary(trainval)
    strata_map = dict(zip(subject_df["subject_id"], subject_df["strata"]))
    y_strata = trainval["subject_id"].map(strata_map)
    if y_strata.notna().all() and y_strata.nunique() >= cfg.n_folds:
        return StratifiedGroupKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed), y_strata
    return GroupKFold(n_splits=cfg.n_folds), None


def summarize_cv_metrics(cv_metrics: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    if cv_metrics.empty:
        return out
    for target in ["SBP", "DBP"]:
        sub = cv_metrics[cv_metrics["target"] == target]
        for metric in ["mae", "rmse", "r2"]:
            out[f"cv_{target.lower()}_{metric}_mean"] = float(sub[metric].mean()) if len(sub) else math.nan
            out[f"cv_{target.lower()}_{metric}_std"] = float(sub[metric].std(ddof=0)) if len(sub) else math.nan
    return out


def run_one_setting(trainval: pd.DataFrame, holdout: pd.DataFrame, feature_set: str, model_name: str, cfg: PhysioAblationConfig) -> dict[str, Any]:
    feature_columns = feature_columns_for_set(trainval, feature_set)
    if not feature_columns:
        raise RuntimeError(f"No feature columns for feature_set={feature_set}")
    cv_metrics_rows = []
    if cfg.run_cv:
        splitter, y = cv_splitter(trainval, cfg)
        groups = trainval["subject_id"].to_numpy()
        split_iter = splitter.split(trainval, y if y is not None else None, groups=groups)
        for fold_idx, (tr_idx, va_idx) in enumerate(split_iter, start=1):
            tr = trainval.iloc[tr_idx].reset_index(drop=True)
            va = trainval.iloc[va_idx].reset_index(drop=True)
            pred, _, _ = fit_predict_one(tr, va, feature_columns, model_name, cfg.seed + fold_idx, cfg.n_jobs)
            cv_metrics_rows.append(evaluate_predictions(pred, "cv", f"fold_{fold_idx}"))
    cv_metrics = pd.concat(cv_metrics_rows, ignore_index=True) if cv_metrics_rows else pd.DataFrame()
    hold_pred, transformed_feature_names, final_preproc = fit_predict_one(trainval, holdout, feature_columns, model_name, cfg.seed, cfg.n_jobs)
    hold_metrics = evaluate_predictions(hold_pred, "holdout", "full_trainval")
    row: dict[str, Any] = {
        "feature_set": feature_set,
        "model_name": model_name,
        "raw_feature_count": len(feature_columns),
        "transformed_feature_count": len(transformed_feature_names),
        "numeric_feature_count": len(final_preproc.numeric_columns),
        "categorical_feature_count": len(final_preproc.categorical_columns),
        "numeric_columns": final_preproc.numeric_columns,
        "categorical_columns": final_preproc.categorical_columns,
        "feature_columns": feature_columns,
        "transformed_feature_names": transformed_feature_names,
    }
    for _, r in hold_metrics.iterrows():
        target = str(r["target"]).lower()
        for metric in ["mae", "rmse", "r2", "residual_me", "residual_sd"]:
            row[f"{target}_{metric}"] = float(r[metric])
    row["mae_sum"] = row.get("sbp_mae", math.nan) + row.get("dbp_mae", math.nan)
    row.update(summarize_cv_metrics(cv_metrics))
    return row, hold_pred, cv_metrics, hold_metrics


def build_report(cfg: PhysioAblationConfig, summary: pd.DataFrame, flow: dict[str, Any], groups: dict[str, list[str]]) -> str:
    lines = [
        f"# Physiological Feature Ablation Report: {cfg.batch_id}",
        "",
        "## Purpose",
        "",
        "Evaluate whether ECG-PPG physiological interaction features (PAT/PWTT-like timing and PPG morphology) add predictive value beyond basic statistical signal features and metadata.",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(to_serializable(asdict(cfg)), indent=2, ensure_ascii=True),
        "```",
        "",
        "## Sample Flow",
        "",
        "```json",
        json.dumps(to_serializable(flow), indent=2, ensure_ascii=True),
        "```",
        "",
        "## Feature Groups",
        "",
        f"- Metadata features: {len(groups.get('meta', []))}",
        f"- Basic ECG/PPG statistical features: {len(groups.get('basic', []))}",
        f"- Physiological fusion features: {len(groups.get('physio', []))}",
        "",
        "## Summary",
        "",
        summary.drop(columns=[c for c in ["feature_columns", "transformed_feature_names", "numeric_columns", "categorical_columns"] if c in summary.columns]).to_markdown(index=False),
        "",
        "## Notes",
        "",
        "- PAT/PWTT-like features are approximate ECG-to-PPG timing descriptors derived from automatically detected ECG R peaks and PPG foot/slope/peak landmarks.",
        "- They should be interpreted as robust physiological descriptors rather than manually verified beat-level clinical annotations.",
        "- Aligned preprocessing forces age/height/weight/BMI/drug_binary and all ECG/PPG/physio descriptors to numeric features; only sex/dx_htn/pre_smoke/pre_coffee are one-hot encoded.",
    ]
    return "\n".join(lines)


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()
    cfg = PhysioAblationConfig(**vars(args))
    set_seed(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Data loading and split follows the current DL/TabPFN framework for consistency.
    dl_cfg = RunConfig(
        data_path=cfg.data_path,
        seed=cfg.seed,
        holdout_ratio=cfg.holdout_ratio,
        n_folds=cfg.n_folds,
        sampling_rate=cfg.sampling_rate,
        apply_filter=cfg.apply_filter,
        ecg_clean_method=cfg.ecg_clean_method,
        ppg_clean_method=cfg.ppg_clean_method,
    ).finalize()
    raw = load_raw_dataframe(dl_cfg)
    cleaned, clean_flow = clean_records(raw)
    if cfg.max_records and cfg.max_records > 0:
        cleaned = cleaned.head(cfg.max_records).copy()
    split = pd.read_csv(cfg.split_path, dtype={"subject_id": str, "split": str})
    required = {"subject_id", "split"}
    if not required.issubset(split.columns):
        raise ValueError(f"Fixed split must contain columns {sorted(required)}")
    if split["subject_id"].duplicated().any():
        raise ValueError("Fixed split contains duplicated subject IDs")
    valid_labels = {"model_development", "independent_holdout"}
    if set(split["split"].dropna()) != valid_labels:
        raise ValueError(f"Fixed split labels must be {sorted(valid_labels)}")
    cohort_subjects = set(cleaned["subject_id"].astype(str).unique())
    split_subjects = set(split["subject_id"].astype(str).unique())
    if cohort_subjects != split_subjects:
        raise ValueError(
            "Fixed split and cleaned cohort subjects differ: "
            f"missing={sorted(cohort_subjects - split_subjects)[:10]}, "
            f"extra={sorted(split_subjects - cohort_subjects)[:10]}"
        )
    split_map = dict(zip(split["subject_id"].astype(str), split["split"]))
    trainval_records = cleaned[cleaned["subject_id"].astype(str).map(split_map) == "model_development"].copy()
    holdout_records = cleaned[cleaned["subject_id"].astype(str).map(split_map) == "independent_holdout"].copy()
    split_info = {
        "split_source": str(Path(cfg.split_path).resolve()),
        "split_policy": "preserved manuscript holdout; adult-only cohort",
        "trainval_subjects": int(trainval_records["subject_id"].nunique()),
        "trainval_records": int(len(trainval_records)),
        "holdout_subjects": int(holdout_records["subject_id"].nunique()),
        "holdout_records": int(len(holdout_records)),
    }
    flow = {**clean_flow, **split_info, "used_records_after_max_records": int(len(cleaned))}
    logger.info("Building feature frame: trainval=%s holdout=%s", len(trainval_records), len(holdout_records))
    all_records = pd.concat([trainval_records, holdout_records], ignore_index=True)
    feature_frame = build_feature_frame(all_records, cfg)
    trainval = feature_frame[feature_frame["record_id"].isin(trainval_records["record_id"])].reset_index(drop=True)
    holdout = feature_frame[feature_frame["record_id"].isin(holdout_records["record_id"])].reset_index(drop=True)
    groups = feature_groups_for_frame(feature_frame)

    feature_frame.to_parquet(cfg.output_dir / f"physio_feature_frame_{cfg.batch_id}.parquet", index=False)
    with open(cfg.output_dir / f"feature_groups_{cfg.batch_id}.json", "w", encoding="utf-8") as f:
        json.dump(to_serializable(groups), f, indent=2, ensure_ascii=True)
    with open(cfg.output_dir / f"config_{cfg.batch_id}.json", "w", encoding="utf-8") as f:
        json.dump(to_serializable(asdict(cfg)), f, indent=2, ensure_ascii=True)
    with open(cfg.output_dir / f"sample_flow_{cfg.batch_id}.json", "w", encoding="utf-8") as f:
        json.dump(to_serializable(flow), f, indent=2, ensure_ascii=True)

    summary_rows = []
    prediction_frames = []
    cv_metric_frames = []
    holdout_metric_frames = []
    for feature_set in cfg.feature_set_list():
        for model_name in cfg.model_list():
            logger.info("Running physio ablation feature_set=%s model=%s", feature_set, model_name)
            try:
                row, pred, cvm, hm = run_one_setting(trainval, holdout, feature_set, model_name, cfg)
                row["aborted"] = False
                row["abort_reason"] = ""
                pred.insert(0, "model_name", model_name)
                pred.insert(0, "feature_set", feature_set)
                if not cvm.empty:
                    cvm.insert(0, "model_name", model_name)
                    cvm.insert(0, "feature_set", feature_set)
                hm.insert(0, "model_name", model_name)
                hm.insert(0, "feature_set", feature_set)
                prediction_frames.append(pred)
                cv_metric_frames.append(cvm)
                holdout_metric_frames.append(hm)
            except Exception as exc:
                logger.exception("Ablation failed feature_set=%s model=%s", feature_set, model_name)
                row = {"feature_set": feature_set, "model_name": model_name, "aborted": True, "abort_reason": repr(exc)}
            summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    if "mae_sum" in summary.columns:
        summary = summary.sort_values(["mae_sum", "sbp_mae"], na_position="last")
    summary_path = cfg.output_dir / f"physio_ablation_summary_{cfg.batch_id}.csv"
    summary.to_csv(summary_path, index=False)
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_csv(cfg.output_dir / f"physio_ablation_predictions_{cfg.batch_id}.csv", index=False)
    if cv_metric_frames:
        pd.concat(cv_metric_frames, ignore_index=True).to_csv(cfg.output_dir / f"physio_ablation_cv_metrics_{cfg.batch_id}.csv", index=False)
    if holdout_metric_frames:
        pd.concat(holdout_metric_frames, ignore_index=True).to_csv(cfg.output_dir / f"physio_ablation_holdout_metrics_{cfg.batch_id}.csv", index=False)
    report = build_report(cfg, summary, flow, groups)
    (cfg.output_dir / f"physio_ablation_report_{cfg.batch_id}.md").write_text(report, encoding="utf-8")

    display_cols = [
        "feature_set", "model_name", "raw_feature_count", "transformed_feature_count",
        "sbp_mae", "sbp_rmse", "sbp_r2", "dbp_mae", "dbp_rmse", "dbp_r2", "mae_sum",
        "cv_sbp_mae_mean", "cv_dbp_mae_mean", "aborted", "abort_reason",
    ]
    display_cols = [c for c in display_cols if c in summary.columns]
    print(f"Saved summary: {summary_path}")
    print(summary[display_cols].to_string(index=False))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
