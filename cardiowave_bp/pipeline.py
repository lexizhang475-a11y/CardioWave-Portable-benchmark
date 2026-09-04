from __future__ import annotations

import argparse
import json
import logging
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import neurokit2 as nk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

SBP_MIN = 60.0
SBP_MAX = 200.0
DBP_MIN = 40.0
DBP_MAX = 120.0
DEFAULT_DATA_PATH = "data/cardiowave_portable_dataset.pkl"


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def current_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def zscore(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    return (signal - float(np.mean(signal))) / (float(np.std(signal)) + 1e-6)


def to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def parse_bool(text: str) -> bool:
    value = str(text).lower().strip()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value from {text!r}")


@dataclass
class RunConfig:
    data_path: str = DEFAULT_DATA_PATH
    output_root: str = "outputs"
    timestamp: str = ""
    seed: int = 42
    n_folds: int = 5
    holdout_ratio: float = 0.20
    sampling_rate: int = 500
    apply_filter: bool = True
    ecg_clean_method: str = "neurokit"
    ppg_clean_method: str = "elgendi"

    def finalize(self) -> "RunConfig":
        if not self.timestamp:
            self.timestamp = current_timestamp()
        return self

    @property
    def output_dir(self) -> Path:
        return Path(self.output_root) / self.timestamp


def qcut_labels(values: pd.Series, n_bins: int = 5) -> pd.Series | None:
    if values.nunique() < 2:
        return None
    try:
        labels = pd.qcut(values, q=min(n_bins, values.nunique()), labels=False, duplicates="drop")
    except ValueError:
        return None
    if labels.nunique(dropna=True) < 2:
        return None
    counts = labels.value_counts()
    if counts.min() < 2:
        return None
    return labels


def load_raw_dataframe(config: RunConfig) -> pd.DataFrame:
    """Load the public CardioWave-Portable pkl and normalize identifiers.

    The public file is expected to contain `subject_id` and `record_id`. A legacy
    fallback is retained for older internal snapshots that used `id` instead of
    `subject_id`.
    """
    df = pd.read_pickle(config.data_path).copy().reset_index(drop=True)
    df["source_row"] = np.arange(len(df), dtype=int)

    if "subject_id" not in df.columns:
        if "id" in df.columns:
            df["subject_id"] = df["id"].astype(str)
        else:
            raise KeyError("The dataset must contain `subject_id` or legacy `id`.")
    df["subject_id"] = df["subject_id"].astype(str)

    if "record_id" not in df.columns:
        df["record_id"] = df["source_row"].apply(lambda x: f"R{x + 1:06d}")
    df["record_id"] = df["record_id"].astype(str)

    return df


def clean_binary(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where(numeric.isin([0, 1]))


def _optional_series(df: pd.DataFrame, column: str, default: Any = np.nan) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series(default, index=df.index)


def engineer_tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive harmonized benchmark metadata fields from public variables."""
    out = df.copy()

    age = pd.to_numeric(_optional_series(out, "age"), errors="coerce")
    out["age_clean"] = age.where(age.between(0, 120))

    sex = pd.to_numeric(_optional_series(out, "sex"), errors="coerce")
    out["sex_clean"] = sex.where(sex.isin([0, 1, 2])).fillna(-1).astype(int).astype(str)

    height = pd.to_numeric(_optional_series(out, "height"), errors="coerce")
    weight = pd.to_numeric(_optional_series(out, "weight"), errors="coerce")
    out["height_clean"] = height.where(height.between(120, 220))
    out["weight_clean"] = weight.where(weight.between(30, 200))
    out["bmi"] = out["weight_clean"] / ((out["height_clean"] / 100.0) ** 2)
    out.loc[~out["bmi"].between(10, 80), "bmi"] = np.nan

    out["dx_htn_clean"] = clean_binary(_optional_series(out, "dx_htn"))
    out["pre_smoke_clean"] = clean_binary(_optional_series(out, "pre_smoke"))
    out["pre_coffee_clean"] = clean_binary(_optional_series(out, "pre_coffee"))

    if "drug_binary" in out.columns:
        out["drug_binary"] = clean_binary(out["drug_binary"])
    elif "drug" in out.columns:
        drug = out["drug"].astype("string")
        drug_binary = pd.Series(np.nan, index=out.index, dtype="float64")
        non_missing = drug.notna()
        drug_binary.loc[non_missing & (drug.str.strip() == "0")] = 0.0
        drug_binary.loc[non_missing & (drug.str.strip() != "0")] = 1.0
        out["drug_binary"] = drug_binary
    else:
        out["drug_binary"] = np.nan

    return out


def clean_records(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    flow = {
        "raw_records": int(len(df)),
        "raw_subjects": int(df["subject_id"].nunique()),
    }
    cleaned = df[(pd.to_numeric(df["sbp"], errors="coerce") >= 80) & (pd.to_numeric(df["sbp"], errors="coerce") <= 190)].copy()
    cleaned = cleaned[(pd.to_numeric(cleaned["dbp"], errors="coerce") >= 40) & (pd.to_numeric(cleaned["dbp"], errors="coerce") <= 110)].copy()
    cleaned["sbp"] = pd.to_numeric(cleaned["sbp"], errors="coerce")
    cleaned["dbp"] = pd.to_numeric(cleaned["dbp"], errors="coerce")
    flow["bp_range_records"] = int(len(cleaned))
    flow["bp_range_subjects"] = int(cleaned["subject_id"].nunique())

    range_stats = cleaned.groupby("subject_id").agg(
        sbp_range=("sbp", lambda x: float(np.max(x) - np.min(x))),
        dbp_range=("dbp", lambda x: float(np.max(x) - np.min(x))),
    )
    stable_ids = range_stats[(range_stats["sbp_range"] < 25) & (range_stats["dbp_range"] < 20)].index
    cleaned = cleaned[cleaned["subject_id"].isin(stable_ids)].copy().reset_index(drop=True)
    cleaned["sbp_norm"] = (cleaned["sbp"] - SBP_MIN) / (SBP_MAX - SBP_MIN)
    cleaned["dbp_norm"] = (cleaned["dbp"] - DBP_MIN) / (DBP_MAX - DBP_MIN)
    cleaned = engineer_tabular_features(cleaned)
    flow["stable_records"] = int(len(cleaned))
    flow["stable_subjects"] = int(cleaned["subject_id"].nunique())
    return cleaned, flow


def subject_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.groupby("subject_id").agg(subject_mean_sbp=("sbp", "mean")).reset_index()
    summary["strata"] = qcut_labels(summary["subject_mean_sbp"])
    return summary


def stratified_holdout_split(df: pd.DataFrame, config: RunConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    subject_df = subject_summary(df)
    stratify = subject_df["strata"] if subject_df["strata"].notna().all() else None
    if stratify is not None:
        n_test = max(1, int(round(len(subject_df) * config.holdout_ratio)))
        n_classes = int(pd.Series(stratify).nunique())
        if n_test < n_classes:
            logger.warning(
                "Holdout stratification disabled because test_size=%s is smaller than n_classes=%s",
                n_test,
                n_classes,
            )
            stratify = None
    train_ids, holdout_ids = train_test_split(
        subject_df["subject_id"],
        test_size=config.holdout_ratio,
        random_state=config.seed,
        stratify=stratify,
    )
    train_df = df[df["subject_id"].isin(train_ids)].copy().reset_index(drop=True)
    holdout_df = df[df["subject_id"].isin(holdout_ids)].copy().reset_index(drop=True)
    info = {
        "holdout_records": int(len(holdout_df)),
        "holdout_subjects": int(holdout_df["subject_id"].nunique()),
        "trainval_records": int(len(train_df)),
        "trainval_subjects": int(train_df["subject_id"].nunique()),
    }
    return train_df, holdout_df, info


def apply_filters(ecg: np.ndarray, ppg: np.ndarray, config: RunConfig) -> tuple[np.ndarray, np.ndarray]:
    ecg = np.asarray(ecg, dtype=np.float64)
    ppg = np.asarray(ppg, dtype=np.float64)
    if not config.apply_filter:
        return ecg, ppg
    try:
        ecg = np.asarray(nk.ecg_clean(ecg, sampling_rate=config.sampling_rate, method=config.ecg_clean_method), dtype=np.float64)
    except Exception as exc:
        logger.warning("ECG cleaning failed for one record; using raw ECG. Reason: %s", exc)
    try:
        ppg = np.asarray(nk.ppg_clean(ppg, sampling_rate=config.sampling_rate, method=config.ppg_clean_method), dtype=np.float64)
    except Exception as exc:
        logger.warning("PPG cleaning failed for one record; using raw PPG. Reason: %s", exc)
    return ecg, ppg
