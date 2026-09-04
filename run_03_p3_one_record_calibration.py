#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A3 structured-feature 1-of-4 subject-specific calibration protocol.

Protocol definition
-------------------
A3. Structured 1-of-4 subject-specific calibration

For each subject:
- Sort records by record_id.
- Use the first available record as the calibration record.
- Use the remaining records from the same subject as target records.

Input representation:
- target structured signal features
- calibration structured signal features
- target - calibration feature differences
- calibration BP label

Target-specific label input:
- SBP model uses calibration_sbp.
- DBP model uses calibration_dbp.

Default feature set:
- ecg_ppg_descriptors_timing
  = ECG descriptors + PPG descriptors + ECG-PPG timing descriptors
  = signal-only structured representation, no metadata.

Split:
- The original participant-disjoint fixed model-development / holdout split
  is recovered from the existing full_structured CatBoost holdout prediction file.
- Development subjects and holdout subjects do not overlap.
- Each holdout subject uses only its own calibration record.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from run_02_p1_p2_structured_benchmark import (
    DEFAULT_FEATURE_FRAME,
    TabularPreprocessor,
    build_feature_manifest,
    columns_by_family,
    build_feature_sets,
    build_model,
)


DEFAULT_PREDICTIONS = (
    "outputs/02_p1_p2_structured_benchmark/strict_v1_redefined_groups_full_v1/"
    "redefined_regression_predictions_strict_v1_redefined_groups_full_v1.csv"
)

DEFAULT_OUTPUT_ROOT = "outputs/03_p3_calibration"
DEFAULT_BATCH_ID = "structured_a3_1of4_calibration_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-frame", default=DEFAULT_FEATURE_FRAME)
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--batch-id", default=DEFAULT_BATCH_ID)

    parser.add_argument("--feature-set", default="ecg_ppg_descriptors_timing")
    parser.add_argument("--global-feature-set", default="full_structured")
    parser.add_argument("--global-model-name", default="catboost")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--min-train-records", type=int, default=100)
    return parser.parse_args()


def to_str_id(s: pd.Series) -> pd.Series:
    return s.astype(str)


def save_json(obj: Any, path: Path) -> None:
    def default(o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=default)


def record_sort_key(record_id: Any) -> tuple[int, str]:
    text = str(record_id)
    nums = re.findall(r"\d+", text)
    if nums:
        return int(nums[-1]), text
    return 10**12, text


def metric_summary(df: pd.DataFrame, sbp_pred_col: str, dbp_pred_col: str) -> dict[str, float]:
    y_sbp = df["sbp"].to_numpy(dtype=float)
    y_dbp = df["dbp"].to_numpy(dtype=float)
    p_sbp = df[sbp_pred_col].to_numpy(dtype=float)
    p_dbp = df[dbp_pred_col].to_numpy(dtype=float)

    e_sbp = p_sbp - y_sbp
    e_dbp = p_dbp - y_dbp

    def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if len(y_true) < 2:
            return np.nan
        return float(r2_score(y_true, y_pred))

    out = {
        "records": int(len(df)),
        "subjects": int(df["subject_id"].astype(str).nunique()),

        "sbp_me": float(np.mean(e_sbp)),
        "sbp_sd": float(np.std(e_sbp, ddof=1)) if len(e_sbp) > 1 else 0.0,
        "sbp_mae": float(mean_absolute_error(y_sbp, p_sbp)),
        "sbp_rmse": float(np.sqrt(mean_squared_error(y_sbp, p_sbp))),
        "sbp_r2": safe_r2(y_sbp, p_sbp),
        "sbp_within_5": float(np.mean(np.abs(e_sbp) <= 5.0)),
        "sbp_within_10": float(np.mean(np.abs(e_sbp) <= 10.0)),
        "sbp_within_15": float(np.mean(np.abs(e_sbp) <= 15.0)),

        "dbp_me": float(np.mean(e_dbp)),
        "dbp_sd": float(np.std(e_dbp, ddof=1)) if len(e_dbp) > 1 else 0.0,
        "dbp_mae": float(mean_absolute_error(y_dbp, p_dbp)),
        "dbp_rmse": float(np.sqrt(mean_squared_error(y_dbp, p_dbp))),
        "dbp_r2": safe_r2(y_dbp, p_dbp),
        "dbp_within_5": float(np.mean(np.abs(e_dbp) <= 5.0)),
        "dbp_within_10": float(np.mean(np.abs(e_dbp) <= 10.0)),
        "dbp_within_15": float(np.mean(np.abs(e_dbp) <= 15.0)),
    }
    out["mae_sum"] = out["sbp_mae"] + out["dbp_mae"]
    return out


def recover_subject_level_split(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    global_feature_set: str,
    global_model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = frame.copy()
    frame["record_id"] = to_str_id(frame["record_id"])
    frame["subject_id"] = to_str_id(frame["subject_id"])

    pred = predictions[
        (predictions["feature_set"] == global_feature_set) &
        (predictions["model_name"] == global_model_name)
    ].copy()

    if pred.empty:
        raise RuntimeError(
            f"No prediction rows found for feature_set={global_feature_set}, "
            f"model_name={global_model_name}"
        )

    pred["record_id"] = to_str_id(pred["record_id"])
    pred["subject_id"] = to_str_id(pred["subject_id"])

    holdout_record_ids = set(pred["record_id"].unique().tolist())

    holdout = frame[frame["record_id"].isin(holdout_record_ids)].copy().reset_index(drop=True)
    development = frame[~frame["record_id"].isin(holdout_record_ids)].copy().reset_index(drop=True)

    overlap = set(development["subject_id"]).intersection(set(holdout["subject_id"]))
    if overlap:
        raise RuntimeError(f"Subject-level leakage detected: {len(overlap)} overlapping subjects.")

    return development, holdout, pred


def choose_calibration_and_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each subject:
    - first record by record_id = calibration
    - remaining records = targets
    """
    calibration_rows = []
    target_rows = []

    for _, g in df.groupby("subject_id", sort=False):
        g2 = g.copy()
        g2["_sort_key"] = g2["record_id"].map(record_sort_key)
        g2 = g2.sort_values("_sort_key").drop(columns=["_sort_key"])

        calibration_rows.append(g2.iloc[[0]].copy())

        if len(g2) > 1:
            target_rows.append(g2.iloc[1:].copy())

    calibration = pd.concat(calibration_rows, ignore_index=True) if calibration_rows else pd.DataFrame()
    targets = pd.concat(target_rows, ignore_index=True) if target_rows else pd.DataFrame()

    return calibration, targets


def build_calibrated_frame(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    calibration, targets = choose_calibration_and_targets(df)

    if calibration.empty or targets.empty:
        return pd.DataFrame(), calibration, targets

    calib_cols = ["subject_id", "record_id", "sbp", "dbp"] + feature_columns
    calib = calibration[calib_cols].copy()

    rename = {
        "record_id": "calibration_record_id",
        "sbp": "calibration_sbp",
        "dbp": "calibration_dbp",
    }
    for f in feature_columns:
        rename[f] = f"calib__{f}"

    calib = calib.rename(columns=rename)

    target_cols = ["subject_id", "record_id", "sbp", "dbp"] + feature_columns
    target = targets[target_cols].copy()

    paired = target.merge(calib, on="subject_id", how="left", validate="many_to_one")

    if paired["calibration_record_id"].isna().any():
        raise RuntimeError("Some target records have no calibration record after merge.")

    # Rename target features and add differences.
    rename_target = {f: f"target__{f}" for f in feature_columns}
    paired = paired.rename(columns=rename_target)

    for f in feature_columns:
        paired[f"diff__{f}"] = paired[f"target__{f}"] - paired[f"calib__{f}"]

    # Useful error-analysis columns.
    paired["target_order_after_calibration"] = paired.groupby("subject_id").cumcount() + 1

    return paired, calibration, targets


def feature_columns_for_target(paired: pd.DataFrame, target_name: str) -> list[str]:
    base = [
        c for c in paired.columns
        if c.startswith("target__") or c.startswith("calib__") or c.startswith("diff__")
    ]

    if target_name == "sbp":
        return base + ["calibration_sbp"]
    if target_name == "dbp":
        return base + ["calibration_dbp"]

    raise ValueError(target_name)


def train_and_predict(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    seed: int,
    n_jobs: int,
) -> pd.DataFrame:
    out = holdout[
        [
            "subject_id",
            "record_id",
            "calibration_record_id",
            "sbp",
            "dbp",
            "calibration_sbp",
            "calibration_dbp",
            "target_order_after_calibration",
        ]
    ].copy()

    for i, target in enumerate(["sbp", "dbp"]):
        features = feature_columns_for_target(train, target)

        pre = TabularPreprocessor.fit(train, features)
        train_x = pre.transform(train)
        holdout_x = pre.transform(holdout)

        model = build_model("catboost", seed=seed + 1000 + i, n_jobs=n_jobs)
        model.fit(train_x, train[target].to_numpy(dtype=float))

        pred = np.asarray(model.predict(holdout_x), dtype=float)
        out[f"{target}_pred"] = pred
        out[f"{target}_error"] = pred - out[target].to_numpy(dtype=float)

    return out


def round_numeric(df: pd.DataFrame, digits: int = 2) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]) or pd.api.types.is_integer_dtype(out[c]):
            if c in {"records", "subjects", "holdout_records", "holdout_subjects", "train_records", "train_subjects"}:
                continue
            out[c] = out[c].round(digits)
    return out


def make_report(
    output_dir: Path,
    batch_id: str,
    summary: pd.DataFrame,
    sample_flow: dict[str, Any],
) -> None:
    lines = [
        f"# A3 structured 1-of-4 subject-specific calibration: {batch_id}",
        "",
        "## Calibration protocol",
        "",
        "- For each subject, records were sorted by `record_id`.",
        "- The first available record was used as the calibration record.",
        "- The remaining records from the same subject were used as target records.",
        "- The SBP model used `calibration_sbp`; the DBP model used `calibration_dbp`.",
        "- The original subject-level split was preserved.",
        "- Each independent-holdout subject used only their own calibration record.",
        "",
        "## Feature construction",
        "",
        "- Target structured signal features.",
        "- Calibration structured signal features.",
        "- Target-calibration feature differences.",
        "- Calibration BP label.",
        "",
        "## Summary",
        "",
        summary.to_string(index=False),
        "",
        "## Sample flow",
        "",
        json.dumps(sample_flow, indent=2, ensure_ascii=False),
        "",
    ]

    with open(output_dir / f"structured_a3_report_{batch_id}.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_root) / args.batch_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[LOAD] feature_frame={args.feature_frame}")
    frame = pd.read_parquet(args.feature_frame)
    frame["record_id"] = to_str_id(frame["record_id"])
    frame["subject_id"] = to_str_id(frame["subject_id"])

    print(f"[LOAD] predictions={args.predictions}")
    pred = pd.read_csv(args.predictions)

    development, holdout, split_pred = recover_subject_level_split(
        frame=frame,
        predictions=pred,
        global_feature_set=args.global_feature_set,
        global_model_name=args.global_model_name,
    )

    manifest = build_feature_manifest(frame)
    families = columns_by_family(manifest)
    feature_sets = build_feature_sets(families)

    if args.feature_set not in feature_sets:
        raise ValueError(f"Unknown feature_set={args.feature_set}. Available={list(feature_sets)}")

    raw_features = feature_sets[args.feature_set]

    print(f"[FEATURES] feature_set={args.feature_set}, raw_feature_count={len(raw_features)}")
    print("[BUILD] calibrated development frame")
    dev_calibrated, dev_calibration, dev_targets = build_calibrated_frame(development, raw_features)

    print("[BUILD] calibrated holdout frame")
    holdout_calibrated, holdout_calibration, holdout_targets = build_calibrated_frame(holdout, raw_features)

    if len(dev_calibrated) < args.min_train_records:
        raise RuntimeError(f"Too few calibrated development target records: {len(dev_calibrated)}")

    print("[FIT] CatBoost A3 SBP/DBP models")
    start = time.time()
    predictions = train_and_predict(
        train=dev_calibrated,
        holdout=holdout_calibrated,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )
    runtime_sec = time.time() - start

    metrics = metric_summary(predictions, "sbp_pred", "dbp_pred")

    summary = pd.DataFrame([
        {
            "representation_family": "structured-feature ML",
            "protocol_level": "A3",
            "protocol_name": "ECG+PPG structured 1-of-4 subject-specific calibration",
            "input_information": (
                "target ECG/PPG/timing descriptors + calibration ECG/PPG/timing descriptors "
                "+ target-calibration descriptor differences + calibration BP label"
            ),
            "calibration_setting": "1-of-4 subject-specific calibration; first record by record_id used as calibration",
            "model_name": "CatBoost",
            "feature_set": args.feature_set,
            "raw_feature_count": len(raw_features),
            "train_subjects": int(development["subject_id"].nunique()),
            "train_records": int(len(development)),
            "train_calibration_subjects": int(dev_calibration["subject_id"].nunique()),
            "train_calibration_records": int(len(dev_calibration)),
            "train_target_records": int(len(dev_calibrated)),
            "holdout_subjects": int(holdout["subject_id"].nunique()),
            "holdout_records_original": int(len(holdout)),
            "holdout_calibration_subjects": int(holdout_calibration["subject_id"].nunique()),
            "holdout_calibration_records": int(len(holdout_calibration)),
            "holdout_records": int(len(holdout_calibrated)),
            "runtime_sec": runtime_sec,
            **metrics,
        }
    ])

    sample_flow = {
        "feature_frame": args.feature_frame,
        "split_source_predictions": args.predictions,
        "feature_set": args.feature_set,
        "raw_feature_count": int(len(raw_features)),
        "development_subjects_original": int(development["subject_id"].nunique()),
        "development_records_original": int(len(development)),
        "holdout_subjects_original": int(holdout["subject_id"].nunique()),
        "holdout_records_original": int(len(holdout)),
        "development_calibration_subjects": int(dev_calibration["subject_id"].nunique()),
        "development_calibration_records": int(len(dev_calibration)),
        "development_target_subjects": int(dev_calibrated["subject_id"].nunique()),
        "development_target_records": int(len(dev_calibrated)),
        "holdout_calibration_subjects": int(holdout_calibration["subject_id"].nunique()),
        "holdout_calibration_records": int(len(holdout_calibration)),
        "holdout_target_subjects": int(holdout_calibrated["subject_id"].nunique()),
        "holdout_target_records": int(len(holdout_calibrated)),
        "calibration_record_selection": "first record after sorting each subject's records by record_id",
        "target_record_selection": "remaining records from the same subject",
        "calibration_bp_as_input": {
            "sbp_model": "calibration_sbp",
            "dbp_model": "calibration_dbp",
        },
        "subject_level_split_preserved": True,
        "development_holdout_subject_overlap": 0,
        "leakage_note": (
            "No subject appears in both development and holdout. "
            "Holdout calibration uses only each holdout subject's own calibration record."
        ),
    }

    config_path = output_dir / f"config_{args.batch_id}.json"
    sample_flow_path = output_dir / f"structured_a3_sample_flow_{args.batch_id}.json"
    summary_path = output_dir / f"structured_a3_summary_{args.batch_id}.csv"
    summary_round_path = output_dir / f"structured_a3_summary_rounded_2dec_{args.batch_id}.csv"
    prediction_path = output_dir / f"structured_a3_predictions_{args.batch_id}.csv"
    prediction_round_path = output_dir / f"structured_a3_predictions_rounded_2dec_{args.batch_id}.csv"

    save_json(vars(args), config_path)
    save_json(sample_flow, sample_flow_path)

    summary.to_csv(summary_path, index=False)
    round_numeric(summary, 2).to_csv(summary_round_path, index=False)

    predictions.to_csv(prediction_path, index=False)
    round_numeric(predictions, 2).to_csv(prediction_round_path, index=False)

    make_report(output_dir, args.batch_id, summary, sample_flow)

    print("\nSaved outputs in:", output_dir)
    print("-", summary_path)
    print("-", summary_round_path)
    print("-", prediction_path)
    print("-", prediction_round_path)
    print("-", sample_flow_path)
    print("-", output_dir / f"structured_a3_report_{args.batch_id}.md")

    print("\nA3 summary:")
    show_cols = [
        "protocol_name",
        "holdout_subjects",
        "holdout_records_original",
        "holdout_records",
        "sbp_mae",
        "sbp_rmse",
        "sbp_r2",
        "dbp_mae",
        "dbp_rmse",
        "dbp_r2",
        "mae_sum",
    ]
    print(summary[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
