#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Redefined structured-feature ablation for the strict-v1 BP benchmark.

Purpose
-------
The original strict-v1 feature frame used four broad families:
    ECG basic / PPG basic / physio / metadata

However, the old "physio" family mixes several conceptually different
descriptors: ECG landmark/rate features, PPG landmark/morphology features,
ECG-PPG timing features, and derived interactions.

This script re-groups the existing strict-v1 feature frame into clearer
families for paper writing:

    1. ECG descriptors
       - ecg_*
       - phys_ecg_*
       - phys_rr_*

    2. PPG descriptors
       - ppg_*
       - phys_ppg_*
       - PPG morphology interactions:
         phys_ppg_rise_decay_ratio_mean, phys_amp_width_ratio_mean

    3. ECG-PPG timing descriptors
       - phys_pat_*
       - phys_pwtt_like_*
       - phys_pair_*
       - phys_hr_x_pat_*

    4. Metadata
       - age_clean, sex_clean, height_clean, weight_clean, bmi,
         dx_htn_clean, drug_binary, pre_smoke_clean, pre_coffee_clean

It does NOT redo waveform feature extraction. It reuses:
    outputs/01_physio_feature_ablation/physio_feature_strict_v1/
        physio_feature_frame_physio_feature_strict_v1.parquet

It uses the original strict-v1 prediction file to recover the exact holdout
record IDs, so the subject-level holdout split is aligned with the saved
strict-v1 result.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold


# -----------------------------
# Constants
# -----------------------------

DEFAULT_FEATURE_FRAME = (
    "outputs/01_physio_feature_ablation/physio_feature_strict_v1/"
    "physio_feature_frame_physio_feature_strict_v1.parquet"
)

DEFAULT_SPLIT_SOURCE = (
    "outputs/01_physio_feature_ablation/physio_feature_strict_v1/"
    "physio_ablation_predictions_physio_feature_strict_v1.csv"
)

DEFAULT_OUTPUT_ROOT = "outputs/02_p1_p2_structured_benchmark"
DEFAULT_BATCH_ID = "strict_v1_redefined_groups_full_v1"

ID_COLUMNS = {"record_id", "subject_id"}
TARGET_COLUMNS = {"sbp", "dbp", "sbp_norm", "dbp_norm"}

META_COLUMNS = [
    "age_clean",
    "sex_clean",
    "height_clean",
    "weight_clean",
    "bmi",
    "dx_htn_clean",
    "drug_binary",
    "pre_smoke_clean",
    "pre_coffee_clean",
]

CATEGORICAL_COLUMNS = {
    "sex_clean",
    "dx_htn_clean",
    "pre_smoke_clean",
    "pre_coffee_clean",
}

PPG_MORPH_INTERACTION_COLUMNS = {
    "phys_ppg_rise_decay_ratio_mean",
    "phys_amp_width_ratio_mean",
}

ECG_PPG_TIMING_PREFIXES = (
    "phys_pat_",
    "phys_pwtt_like_",
    "phys_pair_",
    "phys_hr_x_pat_",
)

FEATURE_SET_ORDER = [
    "meta_only",
    "ecg_descriptors",
    "ppg_descriptors",
    "ecg_ppg_descriptors",
    "ecg_ppg_descriptors_meta",
    "ecg_ppg_descriptors_timing",
    "full_structured",
]


# -----------------------------
# Config
# -----------------------------

@dataclass
class Config:
    feature_frame: str = DEFAULT_FEATURE_FRAME
    split_source: str = DEFAULT_SPLIT_SOURCE
    output_root: str = DEFAULT_OUTPUT_ROOT
    batch_id: str = DEFAULT_BATCH_ID
    feature_sets: str = "all"
    models: str = "catboost,xgboost,lightgbm,randomforest"
    seed: int = 42
    n_folds: int = 5
    n_jobs: int = 4
    run_cv: bool = True
    fail_on_unclassified: bool = False


def parse_bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse bool from {v!r}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Redefined strict-v1 structured feature-group ablation."
    )
    p.add_argument("--feature-frame", default=Config.feature_frame)
    p.add_argument("--split-source", default=Config.split_source)
    p.add_argument("--output-root", default=Config.output_root)
    p.add_argument("--batch-id", default=Config.batch_id)
    p.add_argument(
        "--feature-sets",
        default=Config.feature_sets,
        help=(
            "Comma-separated feature sets or 'all'. Supported: "
            + ",".join(FEATURE_SET_ORDER)
        ),
    )
    p.add_argument(
        "--models",
        default=Config.models,
        help="Comma-separated models: catboost,randomforest,lightgbm,xgboost",
    )
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--n-folds", type=int, default=Config.n_folds)
    p.add_argument("--n-jobs", type=int, default=Config.n_jobs)
    p.add_argument("--run-cv", type=parse_bool, default=Config.run_cv)
    p.add_argument(
        "--fail-on-unclassified",
        type=parse_bool,
        default=Config.fail_on_unclassified,
        help="If true, stop when any feature is assigned to other/other_physio.",
    )
    return p


# -----------------------------
# Feature grouping
# -----------------------------

def is_feature_column(col: str) -> bool:
    return col not in ID_COLUMNS and col not in TARGET_COLUMNS


def assign_family(col: str) -> str:
    """Assign every existing raw feature column to a new paper-friendly family."""
    if col in META_COLUMNS:
        return "metadata"

    # Original basic waveform descriptors
    if col.startswith("ecg_"):
        return "ecg_descriptors"
    if col.startswith("ppg_"):
        return "ppg_descriptors"

    # ECG-specific landmark/rate/interval descriptors from the old physio family
    if col.startswith("phys_ecg_") or col.startswith("phys_rr_"):
        return "ecg_descriptors"

    # PPG-specific landmark/rate/interval/morphology descriptors
    if col.startswith("phys_ppg_") or col in PPG_MORPH_INTERACTION_COLUMNS:
        return "ppg_descriptors"

    # Cross-modal ECG-PPG timing / pairing descriptors
    if col.startswith(ECG_PPG_TIMING_PREFIXES):
        return "ecg_ppg_timing"

    # Anything with phys_ that is not yet mapped should be explicitly reviewed.
    if col.startswith("phys_"):
        return "other_physio"

    return "other"


def build_feature_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in frame.columns:
        if not is_feature_column(col):
            continue
        family = assign_family(col)
        old_prefix = "metadata"
        if col.startswith("ecg_"):
            old_prefix = "ecg_basic"
        elif col.startswith("ppg_"):
            old_prefix = "ppg_basic"
        elif col.startswith("phys_"):
            old_prefix = "old_physio"
        elif col in META_COLUMNS:
            old_prefix = "metadata"
        else:
            old_prefix = "unknown"

        rows.append({
            "feature": col,
            "new_family": family,
            "old_prefix_group": old_prefix,
        })
    return pd.DataFrame(rows)


def columns_by_family(manifest: pd.DataFrame) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for fam, sub in manifest.groupby("new_family"):
        out[fam] = sub["feature"].tolist()
    for fam in ["metadata", "ecg_descriptors", "ppg_descriptors", "ecg_ppg_timing", "other_physio", "other"]:
        out.setdefault(fam, [])
    return out


def unique_keep_order(cols: list[str]) -> list[str]:
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def build_feature_sets(families: dict[str, list[str]]) -> dict[str, list[str]]:
    ecg = families["ecg_descriptors"]
    ppg = families["ppg_descriptors"]
    timing = families["ecg_ppg_timing"]
    meta = families["metadata"]
    other_physio = families.get("other_physio", [])
    other = families.get("other", [])

    # If anything remains unclassified, include it only in full_structured so
    # the complete 250-feature representation is not accidentally changed.
    return {
        "meta_only": unique_keep_order(meta),
        "ecg_descriptors": unique_keep_order(ecg),
        "ppg_descriptors": unique_keep_order(ppg),
        "ecg_ppg_descriptors": unique_keep_order(ecg + ppg),
        "ecg_ppg_descriptors_meta": unique_keep_order(ecg + ppg + meta),
        "ecg_ppg_descriptors_timing": unique_keep_order(ecg + ppg + timing),
        "full_structured": unique_keep_order(ecg + ppg + timing + meta + other_physio + other),
    }


def parse_list_arg(value: str, supported: list[str]) -> list[str]:
    items = [x.strip() for x in str(value).split(",") if x.strip()]
    if len(items) == 1 and items[0].lower() == "all":
        return supported
    bad = [x for x in items if x not in supported]
    if bad:
        raise ValueError(f"Unsupported values={bad}. Supported={supported} or all")
    return items


# -----------------------------
# Preprocessing
# -----------------------------

@dataclass
class TabularPreprocessor:
    numeric_columns: list[str]
    categorical_columns: list[str]
    medians: dict[str, float]
    stds: dict[str, float]
    categories: dict[str, list[str]]
    output_columns: list[str]

    @classmethod
    def fit(cls, frame: pd.DataFrame, feature_columns: list[str]) -> "TabularPreprocessor":
        categorical = [c for c in feature_columns if c in CATEGORICAL_COLUMNS]
        numeric = [c for c in feature_columns if c not in categorical]

        medians: dict[str, float] = {}
        stds: dict[str, float] = {}
        for c in numeric:
            s = pd.to_numeric(frame[c], errors="coerce")
            med = float(s.median()) if s.notna().any() else 0.0
            filled = s.fillna(med)
            std = float(filled.std(ddof=0))
            if not math.isfinite(std) or std <= 1e-12:
                std = 1.0
            medians[c] = med
            stds[c] = std

        categories: dict[str, list[str]] = {}
        for c in categorical:
            s = frame[c].astype("string").fillna("__MISSING__").astype(str)
            cats = sorted(s.unique().tolist())
            if "__MISSING__" not in cats:
                cats.append("__MISSING__")
            categories[c] = cats

        output_columns: list[str] = []
        for c in numeric:
            output_columns.append(c)
            output_columns.append(f"{c}__missing")
        for c in categorical:
            for cat in categories[c]:
                output_columns.append(f"{c}__{cat}")

        return cls(
            numeric_columns=numeric,
            categorical_columns=categorical,
            medians=medians,
            stds=stds,
            categories=categories,
            output_columns=output_columns,
        )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []

        for c in self.numeric_columns:
            s = pd.to_numeric(frame[c], errors="coerce")
            miss = s.isna().astype(float)
            x = (s.fillna(self.medians[c]) - self.medians[c]) / self.stds[c]
            parts.append(pd.DataFrame({
                c: x.astype(float).to_numpy(),
                f"{c}__missing": miss.to_numpy(),
            }, index=frame.index))

        for c in self.categorical_columns:
            s = frame[c].astype("string").fillna("__MISSING__").astype(str)
            cat_df = pd.DataFrame(index=frame.index)
            for cat in self.categories[c]:
                cat_df[f"{c}__{cat}"] = (s == cat).astype(float).to_numpy()
            parts.append(cat_df)

        if parts:
            out = pd.concat(parts, axis=1)
        else:
            out = pd.DataFrame(index=frame.index)

        # Enforce stable order.
        for c in self.output_columns:
            if c not in out.columns:
                out[c] = 0.0
        return out[self.output_columns].astype(float)


# -----------------------------
# Models and metrics
# -----------------------------

def build_model(model_name: str, seed: int, n_jobs: int):
    name = model_name.lower()

    if name == "catboost":
        try:
            from catboost import CatBoostRegressor
        except Exception as exc:
            raise ImportError("catboost is not installed or cannot be imported.") from exc
        return CatBoostRegressor(
            loss_function="RMSE",
            iterations=800,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=n_jobs,
        )

    if name == "randomforest":
        return RandomForestRegressor(
            n_estimators=800,
            random_state=seed,
            n_jobs=n_jobs,
            min_samples_leaf=1,
            max_features="sqrt",
        )

    if name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except Exception as exc:
            raise ImportError("lightgbm is not installed or cannot be imported.") from exc
        return LGBMRegressor(
            n_estimators=800,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            n_jobs=n_jobs,
            verbosity=-1,
        )

    if name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except Exception as exc:
            raise ImportError("xgboost is not installed or cannot be imported.") from exc
        return XGBRegressor(
            n_estimators=800,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=n_jobs,
            verbosity=0,
        )

    raise ValueError(f"Unsupported model_name={model_name}")


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict[str, float]:
    residual = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return {
        f"{prefix}_mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        f"{prefix}_r2": float(r2_score(y_true, y_pred)),
        f"{prefix}_residual_me": float(np.mean(residual)),
        f"{prefix}_residual_sd": float(np.std(residual, ddof=1)) if len(residual) > 1 else 0.0,
        f"{prefix}_within_5": float(np.mean(np.abs(residual) <= 5.0)),
        f"{prefix}_within_10": float(np.mean(np.abs(residual) <= 10.0)),
        f"{prefix}_within_15": float(np.mean(np.abs(residual) <= 15.0)),
    }


def fit_predict_two_targets(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
    seed: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, dict[str, float], int]:
    pre = TabularPreprocessor.fit(train, feature_columns)
    train_x = pre.transform(train)
    test_x = pre.transform(test)

    pred = test[["record_id", "subject_id", "sbp", "dbp"]].copy()
    metrics: dict[str, float] = {
        "raw_feature_count": int(len(feature_columns)),
        "transformed_feature_count": int(train_x.shape[1]),
        "numeric_feature_count": int(len(pre.numeric_columns)),
        "categorical_feature_count": int(len(pre.categorical_columns)),
    }

    for idx, target in enumerate(["sbp", "dbp"]):
        model = build_model(model_name, seed + 101 * (idx + 1), n_jobs)
        model.fit(train_x, train[target].to_numpy(dtype=float))
        y_pred = np.asarray(model.predict(test_x), dtype=float)
        pred[f"{target}_pred"] = y_pred
        metrics.update(metric_dict(test[target].to_numpy(dtype=float), y_pred, target))

    metrics["mae_sum"] = metrics["sbp_mae"] + metrics["dbp_mae"]
    return pred, metrics, int(train_x.shape[1])


def run_group_cv(
    trainval: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
    seed: int,
    n_jobs: int,
    n_folds: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    groups = trainval["subject_id"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return pd.DataFrame(), {}

    n_splits = min(int(n_folds), len(unique_groups))
    if n_splits < 2:
        return pd.DataFrame(), {}

    gkf = GroupKFold(n_splits=n_splits)
    rows = []

    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(trainval, groups=groups), start=1):
        tr = trainval.iloc[tr_idx].reset_index(drop=True)
        va = trainval.iloc[va_idx].reset_index(drop=True)
        pred, metrics, _ = fit_predict_two_targets(
            tr, va, feature_columns, model_name, seed + fold_idx * 1000, n_jobs
        )
        row = {
            "fold": fold_idx,
            "train_records": int(len(tr)),
            "valid_records": int(len(va)),
            "train_subjects": int(tr["subject_id"].nunique()),
            "valid_subjects": int(va["subject_id"].nunique()),
        }
        row.update(metrics)
        rows.append(row)

    cv = pd.DataFrame(rows)
    summary = {}
    for key in ["sbp_mae", "sbp_rmse", "sbp_r2", "dbp_mae", "dbp_rmse", "dbp_r2", "mae_sum"]:
        if key in cv.columns:
            summary[f"cv_{key}_mean"] = float(cv[key].mean())
            summary[f"cv_{key}_std"] = float(cv[key].std(ddof=1)) if len(cv) > 1 else 0.0
    return cv, summary


# -----------------------------
# I/O and main
# -----------------------------

def recover_split(frame: pd.DataFrame, split_source: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not split_source.exists():
        raise FileNotFoundError(f"Split source not found: {split_source}")

    pred = pd.read_csv(split_source)
    if "record_id" not in pred.columns:
        raise KeyError(f"split_source has no record_id column: {split_source}")

    frame = frame.copy()
    frame["record_id"] = frame["record_id"].astype(str)
    frame["subject_id"] = frame["subject_id"].astype(str)

    holdout_ids = set(pred["record_id"].astype(str).unique().tolist())
    holdout = frame[frame["record_id"].isin(holdout_ids)].copy().reset_index(drop=True)
    trainval = frame[~frame["record_id"].isin(holdout_ids)].copy().reset_index(drop=True)

    subject_overlap = sorted(
        set(trainval["subject_id"].astype(str)).intersection(set(holdout["subject_id"].astype(str)))
    )

    info = {
        "split_source": str(split_source),
        "split_source_rows": int(len(pred)),
        "inferred_holdout_record_ids": int(len(holdout_ids)),
        "trainval_records": int(len(trainval)),
        "holdout_records": int(len(holdout)),
        "trainval_subjects": int(trainval["subject_id"].nunique()),
        "holdout_subjects": int(holdout["subject_id"].nunique()),
        "subject_overlap_count": int(len(subject_overlap)),
    }

    if len(holdout) == 0:
        raise RuntimeError("No holdout rows recovered from split_source.")
    if subject_overlap:
        warnings.warn(f"Subject leakage detected: {len(subject_overlap)} overlapping subjects.")

    return trainval, holdout, info


def save_json(obj: Any, path: Path) -> None:
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        return str(o)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=default)


def make_report(
    cfg: Config,
    split_info: dict[str, Any],
    counts: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
) -> str:
    lines = [
        f"# Redefined strict-v1 structured feature groups: {cfg.batch_id}",
        "",
        "## Purpose",
        "",
        "This run redefines the old broad `physio` feature family into clearer paper-oriented groups:",
        "",
        "- ECG descriptors",
        "- PPG descriptors",
        "- ECG–PPG timing descriptors",
        "- Subject-level metadata",
        "",
        "The experiment reuses the existing strict-v1 feature frame and the original strict-v1 holdout split.",
        "",
        "## Split",
        "",
        "```json",
        json.dumps(split_info, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Feature group counts",
        "",
        counts.to_markdown(index=False),
        "",
        "## Holdout summary",
        "",
    ]

    if not summary.empty:
        show_cols = [
            "feature_set", "model_name",
            "raw_feature_count", "transformed_feature_count",
            "sbp_mae", "sbp_rmse", "sbp_r2",
            "dbp_mae", "dbp_rmse", "dbp_r2",
            "mae_sum",
        ]
        show_cols = [c for c in show_cols if c in summary.columns]
        lines.append(summary[show_cols].to_markdown(index=False))
    else:
        lines.append("_No summary rows._")

    lines += [
        "",
        "## Notes for manuscript wording",
        "",
        "- Avoid using the old term `Physio features` as a main manuscript feature family.",
        "- Use `ECG descriptors`, `PPG descriptors`, `ECG–PPG timing descriptors`, and `metadata` instead.",
        "- Interpret the ablation as progressively richer structured representations, not as strictly independent physiological-feature ablation.",
        "",
    ]
    report = "\n".join(lines)
    with open(output_dir / f"redefined_feature_groups_report_{cfg.batch_id}.md", "w", encoding="utf-8") as f:
        f.write(report)
    return report


def main() -> None:
    args = build_parser().parse_args()
    cfg = Config(**vars(args))

    output_dir = Path(cfg.output_root) / cfg.batch_id
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_frame_path = Path(cfg.feature_frame)
    split_source_path = Path(cfg.split_source)

    if not feature_frame_path.exists():
        raise FileNotFoundError(f"Feature frame not found: {feature_frame_path}")

    print(f"[LOAD] feature_frame={feature_frame_path}")
    frame = pd.read_parquet(feature_frame_path)
    frame["record_id"] = frame["record_id"].astype(str)
    frame["subject_id"] = frame["subject_id"].astype(str)

    manifest = build_feature_manifest(frame)
    families = columns_by_family(manifest)
    feature_sets = build_feature_sets(families)

    counts = (
        manifest.groupby("new_family")
        .size()
        .rename("feature_count")
        .reset_index()
        .sort_values("new_family")
    )

    # Add feature set counts.
    fs_counts = pd.DataFrame([
        {"new_family": f"FEATURE_SET::{k}", "feature_count": len(v)}
        for k, v in feature_sets.items()
    ])
    counts_all = pd.concat([counts, fs_counts], ignore_index=True)

    unclassified = manifest[manifest["new_family"].isin(["other", "other_physio"])].copy()
    manifest.to_csv(output_dir / f"redefined_feature_manifest_{cfg.batch_id}.csv", index=False)
    counts_all.to_csv(output_dir / f"redefined_feature_group_counts_{cfg.batch_id}.csv", index=False)
    if not unclassified.empty:
        unclassified.to_csv(output_dir / f"redefined_unclassified_features_{cfg.batch_id}.csv", index=False)
        msg = (
            f"[WARN] Found {len(unclassified)} unclassified features. "
            f"See redefined_unclassified_features_{cfg.batch_id}.csv"
        )
        print(msg)
        if cfg.fail_on_unclassified:
            raise RuntimeError(msg)

    trainval, holdout, split_info = recover_split(frame, split_source_path)

    sample_flow = {
        "feature_frame": str(feature_frame_path),
        "records_total": int(len(frame)),
        "subjects_total": int(frame["subject_id"].nunique()),
        **split_info,
        "feature_group_counts": counts_all.to_dict(orient="records"),
    }
    save_json(asdict(cfg), output_dir / f"config_{cfg.batch_id}.json")
    save_json(sample_flow, output_dir / f"sample_flow_{cfg.batch_id}.json")

    selected_feature_sets = parse_list_arg(cfg.feature_sets, FEATURE_SET_ORDER)
    selected_models = [x.strip().lower() for x in cfg.models.split(",") if x.strip()]
    supported_models = ["catboost", "randomforest", "lightgbm", "xgboost"]
    bad_models = [m for m in selected_models if m not in supported_models]
    if bad_models:
        raise ValueError(f"Unsupported models={bad_models}. Supported={supported_models}")

    summary_rows = []
    prediction_rows = []
    cv_rows = []

    for feature_set in selected_feature_sets:
        feature_columns = feature_sets[feature_set]
        if not feature_columns:
            print(f"[SKIP] feature_set={feature_set} has zero features.")
            continue

        missing_cols = [c for c in feature_columns if c not in frame.columns]
        if missing_cols:
            raise RuntimeError(f"Feature set {feature_set} contains missing columns: {missing_cols[:10]}")

        for model_name in selected_models:
            start = time.time()
            print(f"[RUN] feature_set={feature_set} model={model_name} raw_features={len(feature_columns)}")

            try:
                pred, metrics, transformed_count = fit_predict_two_targets(
                    trainval, holdout, feature_columns, model_name, cfg.seed, cfg.n_jobs
                )
                runtime_sec = time.time() - start

                row = {
                    "feature_set": feature_set,
                    "model_name": model_name,
                    "raw_feature_count": int(len(feature_columns)),
                    "transformed_feature_count": int(transformed_count),
                    "train_records": int(len(trainval)),
                    "holdout_records": int(len(holdout)),
                    "train_subjects": int(trainval["subject_id"].nunique()),
                    "holdout_subjects": int(holdout["subject_id"].nunique()),
                    "runtime_sec": float(runtime_sec),
                    "aborted": False,
                    "abort_reason": "",
                }
                row.update(metrics)

                if cfg.run_cv:
                    cv, cv_summary = run_group_cv(
                        trainval,
                        feature_columns,
                        model_name,
                        cfg.seed,
                        cfg.n_jobs,
                        cfg.n_folds,
                    )
                    if not cv.empty:
                        cv.insert(0, "model_name", model_name)
                        cv.insert(0, "feature_set", feature_set)
                        cv_rows.append(cv)
                    row.update(cv_summary)

                pred.insert(0, "model_name", model_name)
                pred.insert(0, "feature_set", feature_set)
                prediction_rows.append(pred)

            except Exception as exc:
                runtime_sec = time.time() - start
                print(f"[ERROR] feature_set={feature_set} model={model_name}: {exc!r}")
                row = {
                    "feature_set": feature_set,
                    "model_name": model_name,
                    "raw_feature_count": int(len(feature_columns)),
                    "transformed_feature_count": np.nan,
                    "train_records": int(len(trainval)),
                    "holdout_records": int(len(holdout)),
                    "train_subjects": int(trainval["subject_id"].nunique()),
                    "holdout_subjects": int(holdout["subject_id"].nunique()),
                    "runtime_sec": float(runtime_sec),
                    "aborted": True,
                    "abort_reason": repr(exc),
                }

            summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    if "mae_sum" in summary.columns:
        summary = summary.sort_values(["aborted", "mae_sum", "sbp_mae"], na_position="last")

    summary_path = output_dir / f"redefined_regression_summary_{cfg.batch_id}.csv"
    pred_path = output_dir / f"redefined_regression_predictions_{cfg.batch_id}.csv"
    cv_path = output_dir / f"redefined_regression_cv_{cfg.batch_id}.csv"

    summary.to_csv(summary_path, index=False)

    if prediction_rows:
        pd.concat(prediction_rows, ignore_index=True).to_csv(pred_path, index=False)
    else:
        pd.DataFrame().to_csv(pred_path, index=False)

    if cv_rows:
        pd.concat(cv_rows, ignore_index=True).to_csv(cv_path, index=False)
    else:
        pd.DataFrame().to_csv(cv_path, index=False)

    make_report(cfg, split_info, counts_all, summary, output_dir)

    print("\nSaved outputs:")
    for p in [
        output_dir / f"config_{cfg.batch_id}.json",
        output_dir / f"sample_flow_{cfg.batch_id}.json",
        output_dir / f"redefined_feature_manifest_{cfg.batch_id}.csv",
        output_dir / f"redefined_feature_group_counts_{cfg.batch_id}.csv",
        summary_path,
        pred_path,
        cv_path,
        output_dir / f"redefined_feature_groups_report_{cfg.batch_id}.md",
    ]:
        print("-", p)

    if not summary.empty and "mae_sum" in summary.columns:
        print("\nTop results:")
        show_cols = [
            "feature_set", "model_name",
            "raw_feature_count", "transformed_feature_count",
            "sbp_mae", "sbp_r2", "dbp_mae", "dbp_r2", "mae_sum",
        ]
        show_cols = [c for c in show_cols if c in summary.columns]
        print(summary[show_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()