#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Final Scientific Data revision package for the BP dataset paper.

This script collects existing results and computes final audit/statistical outputs for:
1. Public dataset integrity checks.
2. P1/P2 all-model structured-feature benchmark.
3. P1/P2/P3 main protocol comparison.
4. Same-subset sensitivity check for P1/P2/P3 CatBoost, including P3-naive carry-forward baseline.
5. P3 CatBoost Bland-Altman analysis.
6. P3 CatBoost subgroup error analysis.
7. Final markdown report.

Run from the CardioWave-Portable project root:

    python run_05_make_final_revision_outputs.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Paths
# -----------------------------

OUT_DIR = Path("outputs/05_final_revision_outputs")

PUBLIC_PKL = Path("data/cardiowave_portable_dataset.pkl")

STRUCTURED_SUMMARY = Path(
    "outputs/02_p1_p2_structured_benchmark/"
    "strict_v1_redefined_groups_full_v1/"
    "redefined_regression_summary_strict_v1_redefined_groups_full_v1.csv"
)

STRUCTURED_PREDICTIONS = Path(
    "outputs/02_p1_p2_structured_benchmark/"
    "strict_v1_redefined_groups_full_v1/"
    "redefined_regression_predictions_strict_v1_redefined_groups_full_v1.csv"
)

P3_SUMMARY = Path(
    "outputs/03_p3_calibration/"
    "structured_a3_1of4_calibration_v1/"
    "structured_a3_summary_structured_a3_1of4_calibration_v1.csv"
)

P3_PREDICTIONS = Path(
    "outputs/03_p3_calibration/"
    "structured_a3_1of4_calibration_v1/"
    "structured_a3_predictions_structured_a3_1of4_calibration_v1.csv"
)

P3_SAMPLE_FLOW = Path(
    "outputs/03_p3_calibration/"
    "structured_a3_1of4_calibration_v1/"
    "structured_a3_sample_flow_structured_a3_1of4_calibration_v1.json"
)

FEATURE_FRAME = Path(
    "outputs/01_physio_feature_ablation/"
    "physio_feature_strict_v1/"
    "physio_feature_frame_physio_feature_strict_v1.parquet"
)

ADULT_SOURCE_SUBJECTS_DOCUMENTED = 1055
ADULT_SOURCE_RECORDS_DOCUMENTED = 4220


# -----------------------------
# Utilities
# -----------------------------

def ensure_inputs() -> None:
    required = [
        PUBLIC_PKL,
        STRUCTURED_SUMMARY,
        STRUCTURED_PREDICTIONS,
        P3_SUMMARY,
        P3_PREDICTIONS,
        P3_SAMPLE_FLOW,
        FEATURE_FRAME,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(missing))


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: Path) -> None:
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


def round_numeric(df: pd.DataFrame, digits: int = 2) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(digits)
    return out


def r2_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return np.nan
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom <= 1e-12:
        return np.nan
    num = np.sum((y_true - y_pred) ** 2)
    return float(1.0 - num / denom)


def metric_from_arrays(
    sbp_true: np.ndarray,
    sbp_pred: np.ndarray,
    dbp_true: np.ndarray,
    dbp_pred: np.ndarray,
) -> dict[str, float]:
    sbp_true = np.asarray(sbp_true, dtype=float)
    sbp_pred = np.asarray(sbp_pred, dtype=float)
    dbp_true = np.asarray(dbp_true, dtype=float)
    dbp_pred = np.asarray(dbp_pred, dtype=float)

    sbp_err = sbp_pred - sbp_true
    dbp_err = dbp_pred - dbp_true

    sbp_mae = float(np.mean(np.abs(sbp_err)))
    dbp_mae = float(np.mean(np.abs(dbp_err)))

    return {
        "sbp_mae": sbp_mae,
        "sbp_rmse": float(np.sqrt(np.mean(sbp_err ** 2))),
        "sbp_r2": r2_safe(sbp_true, sbp_pred),
        "sbp_me": float(np.mean(sbp_err)),
        "sbp_sd": float(np.std(sbp_err, ddof=1)) if len(sbp_err) > 1 else 0.0,
        "dbp_mae": dbp_mae,
        "dbp_rmse": float(np.sqrt(np.mean(dbp_err ** 2))),
        "dbp_r2": r2_safe(dbp_true, dbp_pred),
        "dbp_me": float(np.mean(dbp_err)),
        "dbp_sd": float(np.std(dbp_err, ddof=1)) if len(dbp_err) > 1 else 0.0,
        "mae_sum": sbp_mae + dbp_mae,
    }


def metric_from_prediction_df(
    df: pd.DataFrame,
    true_sbp_col: str,
    pred_sbp_col: str,
    true_dbp_col: str,
    pred_dbp_col: str,
    subject_col: str = "subject_id",
) -> dict[str, Any]:
    m = metric_from_arrays(
        df[true_sbp_col].to_numpy(dtype=float),
        df[pred_sbp_col].to_numpy(dtype=float),
        df[true_dbp_col].to_numpy(dtype=float),
        df[pred_dbp_col].to_numpy(dtype=float),
    )
    return {
        "evaluation_subjects": int(df[subject_col].astype(str).nunique()),
        "evaluation_records": int(len(df)),
        **m,
    }


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    show = df.copy()
    if max_rows is not None:
        show = show.head(max_rows)
    try:
        return show.to_markdown(index=False)
    except Exception:
        return "```\n" + show.to_string(index=False) + "\n```"


def write_latex_table(df: pd.DataFrame, path: Path, float_format: str = "%.2f") -> None:
    try:
        txt = df.to_latex(index=False, escape=False, float_format=float_format)
    except Exception:
        txt = df.to_string(index=False)
    path.write_text(txt, encoding="utf-8")


# -----------------------------
# Task 1 and 2: public dataset audit
# -----------------------------

def len_or_nan(x: Any) -> int | float:
    try:
        return int(len(x))
    except Exception:
        return np.nan


def value_counts_df(df: pd.DataFrame, col: str) -> pd.DataFrame:
    vc = df[col].astype("string").fillna("<NA>").value_counts(dropna=False)
    rows = []
    for value, count in vc.items():
        rows.append({
            "column": col,
            "value": str(value),
            "count": int(count),
            "percent": float(count / len(df) * 100.0),
        })
    return pd.DataFrame(rows)


def create_public_dataset_integrity_outputs() -> dict[str, Any]:
    public = pd.read_pickle(PUBLIC_PKL)

    public["subject_id"] = public["subject_id"].astype(str)
    public["record_id"] = public["record_id"].astype(str)

    ecg_lengths = public["ecg"].map(len_or_nan)
    ppg_lengths = public["ppg"].map(len_or_nan)

    key_fields = [
        "subject_id", "record_id", "ecg", "ppg", "sbp", "dbp",
        "card", "start_idx", "end_idx", "ecg_label", "ppg_label",
    ]

    metadata_fields = [
        "sex", "age", "height", "weight", "dx_htn", "fh_htn", "bp_measured",
        "drug_binary", "smoke_status", "alcohol_status", "coffee_tea",
        "pre_smoke", "pre_coffee", "pre_exercise", "unwell",
        "dx_htn_years", "fh_htn_persons", "sbp_week", "dbp_week",
        "smoke_quit_year", "smoke_cigs_day",
    ]
    metadata_fields = [c for c in metadata_fields if c in public.columns]

    integrity_rows = [
        {"item": "public_release_file", "value": str(PUBLIC_PKL)},
        {"item": "records", "value": int(len(public))},
        {"item": "subjects", "value": int(public["subject_id"].nunique())},
        {"item": "columns", "value": int(public.shape[1])},
        {"item": "ecg_min_length", "value": int(ecg_lengths.min())},
        {"item": "ecg_median_length", "value": float(ecg_lengths.median())},
        {"item": "ecg_max_length", "value": int(ecg_lengths.max())},
        {"item": "ecg_unique_lengths", "value": ",".join(map(str, sorted(ecg_lengths.dropna().unique().astype(int))))},
        {"item": "ppg_min_length", "value": int(ppg_lengths.min())},
        {"item": "ppg_median_length", "value": float(ppg_lengths.median())},
        {"item": "ppg_max_length", "value": int(ppg_lengths.max())},
        {"item": "ppg_unique_lengths", "value": ",".join(map(str, sorted(ppg_lengths.dropna().unique().astype(int))))},
        {"item": "start_idx_unique", "value": ",".join(map(str, sorted(public["start_idx"].dropna().unique())))},
        {"item": "end_idx_unique", "value": ",".join(map(str, sorted(public["end_idx"].dropna().unique())))},
        {"item": "card_unique", "value": ",".join(map(str, sorted(public["card"].dropna().unique())))},
    ]
    integrity_summary = pd.DataFrame(integrity_rows)
    integrity_summary.to_csv(OUT_DIR / "public_dataset_integrity_summary.csv", index=False)

    key_missing_rows = []
    for col in key_fields:
        key_missing_rows.append({
            "field": col,
            "records": int(len(public)),
            "nonmissing": int(public[col].notna().sum()),
            "missing": int(public[col].isna().sum()),
            "missing_rate": float(public[col].isna().mean()),
        })
    key_missing = pd.DataFrame(key_missing_rows)
    key_missing.to_csv(OUT_DIR / "public_dataset_key_field_missingness.csv", index=False)

    metadata_missing_rows = []
    for col in metadata_fields:
        metadata_missing_rows.append({
            "field": col,
            "records": int(len(public)),
            "nonmissing": int(public[col].notna().sum()),
            "missing": int(public[col].isna().sum()),
            "missing_rate": float(public[col].isna().mean()),
            "unique_nonmissing": int(public[col].dropna().nunique()),
        })
    metadata_missing = pd.DataFrame(metadata_missing_rows)
    metadata_missing.to_csv(OUT_DIR / "public_dataset_metadata_missingness.csv", index=False)

    label_counts = pd.concat(
        [
            value_counts_df(public, "ecg_label"),
            value_counts_df(public, "ppg_label"),
        ],
        ignore_index=True,
    )
    if "drug_binary" in public.columns:
        label_counts = pd.concat(
            [label_counts, value_counts_df(public, "drug_binary")],
            ignore_index=True,
        )
    label_counts.to_csv(OUT_DIR / "public_dataset_label_counts.csv", index=False)

    report = [
        "# Public dataset integrity report",
        "",
        f"- Public release file: `{PUBLIC_PKL}`",
        f"- Records: {len(public)}",
        f"- Subjects: {public['subject_id'].nunique()}",
        f"- Columns: {public.shape[1]}",
        f"- ECG length distribution: {ecg_lengths.value_counts().sort_index().to_dict()}",
        f"- PPG length distribution: {ppg_lengths.value_counts().sort_index().to_dict()}",
        f"- Card types: {public['card'].value_counts(dropna=False).to_dict()}",
        "",
        "## Key-field missingness",
        "",
        md_table(key_missing),
        "",
        "## Signal-quality labels and drug_binary counts",
        "",
        md_table(label_counts),
        "",
        "## Metadata missingness",
        "",
        md_table(metadata_missing),
        "",
    ]
    (OUT_DIR / "public_dataset_integrity_report.md").write_text("\n".join(report), encoding="utf-8")

    return {
        "records": int(len(public)),
        "subjects": int(public["subject_id"].nunique()),
        "columns": int(public.shape[1]),
        "ecg_lengths": ecg_lengths.value_counts().sort_index().to_dict(),
        "ppg_lengths": ppg_lengths.value_counts().sort_index().to_dict(),
    }


def create_dataset_flow_audit(
    p1_catboost_row: pd.Series,
    p3_flow: dict[str, Any],
    public_integrity_info: dict[str, Any],
) -> pd.DataFrame:
    final_benchmark_subjects = int(p1_catboost_row["train_subjects"] + p1_catboost_row["holdout_subjects"])
    final_benchmark_records = int(p1_catboost_row["train_records"] + p1_catboost_row["holdout_records"])

    rows = [
        {
            "stage": "adult source collection",
            "subjects": ADULT_SOURCE_SUBJECTS_DOCUMENTED,
            "records": ADULT_SOURCE_RECORDS_DOCUMENTED,
            "source": "adult-only documented project trace",
            "notes": "Adult records available before final public dataset construction.",
        },
        {
            "stage": "public released dataset",
            "subjects": int(public_integrity_info["subjects"]),
            "records": int(public_integrity_info["records"]),
            "source": str(PUBLIC_PKL),
            "notes": "De-identified adult-only modeling input derived from the PhysioNet release.",
        },
        {
            "stage": "final benchmark cohort",
            "subjects": final_benchmark_subjects,
            "records": final_benchmark_records,
            "source": str(STRUCTURED_SUMMARY),
            "notes": "After benchmark cohort filtering.",
        },
        {
            "stage": "model-development set",
            "subjects": int(p1_catboost_row["train_subjects"]),
            "records": int(p1_catboost_row["train_records"]),
            "source": str(STRUCTURED_SUMMARY),
            "notes": "Subject-level development partition.",
        },
        {
            "stage": "participant-disjoint fixed holdout set",
            "subjects": int(p1_catboost_row["holdout_subjects"]),
            "records": int(p1_catboost_row["holdout_records"]),
            "source": str(STRUCTURED_SUMMARY),
            "notes": "Subject-level holdout partition.",
        },
        {
            "stage": "P3 holdout target-record evaluation subset",
            "subjects": int(p3_flow["holdout_target_subjects"]),
            "records": int(p3_flow["holdout_target_records"]),
            "source": str(P3_SAMPLE_FLOW),
            "notes": "After excluding one calibration record per holdout subject.",
        },
    ]

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "dataset_flow_audit.csv", index=False)
    return out


# -----------------------------
# Task 3: P1/P2 all-model benchmark
# -----------------------------

def extract_p1_p2_all_model_benchmark(summary: pd.DataFrame) -> pd.DataFrame:
    protocol_map = {
        "ecg_ppg_descriptors_timing": {
            "protocol": "P1 signal-only no-calibration protocol",
            "input_setting": "ECG descriptors + PPG descriptors + ECG–PPG timing descriptors",
            "calibration_setting": "No metadata; no same-subject calibration record; no calibration BP",
        },
        "full_structured": {
            "protocol": "P2 metadata-assisted individual-prior protocol",
            "input_setting": "ECG descriptors + PPG descriptors + ECG–PPG timing descriptors + harmonized metadata",
            "calibration_setting": "Metadata-assisted individual prior; no same-subject calibration record; no calibration BP",
        },
    }

    allowed_models = ["catboost", "xgboost", "lightgbm", "randomforest"]

    sub = summary[
        summary["feature_set"].isin(protocol_map.keys())
        & summary["model_name"].astype(str).str.lower().isin(allowed_models)
    ].copy()

    rows = []
    for _, r in sub.iterrows():
        info = protocol_map[r["feature_set"]]
        rows.append({
            "protocol": info["protocol"],
            "feature_set": r["feature_set"],
            "input_setting": info["input_setting"],
            "calibration_setting": info["calibration_setting"],
            "model": r["model_name"],
            "evaluation_subjects": int(r["holdout_subjects"]),
            "evaluation_records": int(r["holdout_records"]),
            "sbp_mae": float(r["sbp_mae"]),
            "sbp_rmse": float(r["sbp_rmse"]),
            "sbp_r2": float(r["sbp_r2"]),
            "sbp_me": float(r["sbp_residual_me"]),
            "sbp_sd": float(r["sbp_residual_sd"]),
            "dbp_mae": float(r["dbp_mae"]),
            "dbp_rmse": float(r["dbp_rmse"]),
            "dbp_r2": float(r["dbp_r2"]),
            "dbp_me": float(r["dbp_residual_me"]),
            "dbp_sd": float(r["dbp_residual_sd"]),
            "mae_sum": float(r["mae_sum"]),
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(["protocol", "mae_sum"]).reset_index(drop=True)

    out.to_csv(OUT_DIR / "model_family_benchmark_p1_p2_full_precision.csv", index=False)
    round_numeric(out).to_csv(OUT_DIR / "model_family_benchmark_p1_p2_rounded.csv", index=False)

    return out


# -----------------------------
# Task 4: protocol comparison
# -----------------------------

def get_summary_row(summary: pd.DataFrame, feature_set: str, model_name: str) -> pd.Series:
    mask = (
        (summary["feature_set"] == feature_set)
        & (summary["model_name"].astype(str).str.lower() == model_name.lower())
    )
    if not mask.any():
        raise RuntimeError(f"Missing row: feature_set={feature_set}, model={model_name}")
    return summary[mask].iloc[0]


def create_main_protocol_comparison(summary: pd.DataFrame, p3_summary: pd.DataFrame) -> pd.DataFrame:
    p1 = get_summary_row(summary, "ecg_ppg_descriptors_timing", "catboost")
    p2 = get_summary_row(summary, "full_structured", "catboost")
    p3 = p3_summary.iloc[0]

    rows = [
        {
            "protocol": "P1 signal-only no-calibration protocol",
            "input_setting": "ECG descriptors + PPG descriptors + ECG–PPG timing descriptors",
            "calibration_setting": "No metadata; no same-subject calibration record; no calibration BP",
            "model": "CatBoost",
            "evaluation_subjects": int(p1["holdout_subjects"]),
            "evaluation_records": int(p1["holdout_records"]),
            "sbp_mae": float(p1["sbp_mae"]),
            "sbp_rmse": float(p1["sbp_rmse"]),
            "sbp_r2": float(p1["sbp_r2"]),
            "dbp_mae": float(p1["dbp_mae"]),
            "dbp_rmse": float(p1["dbp_rmse"]),
            "dbp_r2": float(p1["dbp_r2"]),
            "mae_sum": float(p1["mae_sum"]),
            "notes": "Evaluated on complete participant-disjoint fixed holdout set.",
        },
        {
            "protocol": "P2 metadata-assisted individual-prior protocol",
            "input_setting": "ECG descriptors + PPG descriptors + ECG–PPG timing descriptors + harmonized metadata",
            "calibration_setting": "Metadata-assisted individual prior; no same-subject calibration record; no calibration BP",
            "model": "CatBoost",
            "evaluation_subjects": int(p2["holdout_subjects"]),
            "evaluation_records": int(p2["holdout_records"]),
            "sbp_mae": float(p2["sbp_mae"]),
            "sbp_rmse": float(p2["sbp_rmse"]),
            "sbp_r2": float(p2["sbp_r2"]),
            "dbp_mae": float(p2["dbp_mae"]),
            "dbp_rmse": float(p2["dbp_rmse"]),
            "dbp_r2": float(p2["dbp_r2"]),
            "mae_sum": float(p2["mae_sum"]),
            "notes": "Evaluated on complete participant-disjoint fixed holdout set.",
        },
        {
            "protocol": "P3 one-record subject-specific calibration protocol",
            "input_setting": (
                "target ECG/PPG/timing descriptors + calibration ECG/PPG/timing descriptors "
                "+ target-minus-calibration descriptor differences + calibration BP label"
            ),
            "calibration_setting": "First eligible record per subject used as calibration; remaining records used as target records",
            "model": "CatBoost",
            "evaluation_subjects": int(p3["subjects"]),
            "evaluation_records": int(p3["records"]),
            "sbp_mae": float(p3["sbp_mae"]),
            "sbp_rmse": float(p3["sbp_rmse"]),
            "sbp_r2": float(p3["sbp_r2"]),
            "dbp_mae": float(p3["dbp_mae"]),
            "dbp_rmse": float(p3["dbp_rmse"]),
            "dbp_r2": float(p3["dbp_r2"]),
            "mae_sum": float(p3["mae_sum"]),
            "notes": "Evaluated on target records after excluding one calibration record per holdout subject.",
        },
    ]

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "protocol_comparison_main_full_precision.csv", index=False)
    round_numeric(out).to_csv(OUT_DIR / "protocol_comparison_main_rounded.csv", index=False)

    latex_cols = [
        "protocol", "model", "evaluation_subjects", "evaluation_records",
        "sbp_mae", "sbp_rmse", "sbp_r2", "dbp_mae", "dbp_rmse", "dbp_r2", "mae_sum",
    ]
    write_latex_table(round_numeric(out[latex_cols]), OUT_DIR / "protocol_comparison_main_latex_table.txt")

    return out


# -----------------------------
# Task 5: same-subset sensitivity
# -----------------------------

def normalize_structured_pred(pred: pd.DataFrame) -> pd.DataFrame:
    out = pred.copy()
    out["record_id"] = out["record_id"].astype(str)
    out["subject_id"] = out["subject_id"].astype(str)
    return out


def normalize_p3_pred(pred: pd.DataFrame) -> pd.DataFrame:
    out = pred.copy()
    out["record_id"] = out["record_id"].astype(str)
    out["subject_id"] = out["subject_id"].astype(str)
    return out


def create_same_subset_sensitivity(
    structured_predictions: pd.DataFrame,
    p3_predictions: pd.DataFrame,
) -> pd.DataFrame:
    pred = normalize_structured_pred(structured_predictions)
    p3 = normalize_p3_pred(p3_predictions)

    target_record_ids = set(p3["record_id"].astype(str).tolist())

    rows = []

    settings = [
        ("P1 signal-only no-calibration protocol", "ecg_ppg_descriptors_timing"),
        ("P2 metadata-assisted individual-prior protocol", "full_structured"),
    ]

    for protocol, feature_set in settings:
        sub = pred[
            (pred["feature_set"] == feature_set)
            & (pred["model_name"].astype(str).str.lower() == "catboost")
            & (pred["record_id"].isin(target_record_ids))
        ].copy()

        if len(sub) == 0:
            raise RuntimeError(f"No P1/P2 predictions found for same-subset check: {protocol}")

        m = metric_from_prediction_df(
            sub,
            true_sbp_col="sbp",
            pred_sbp_col="sbp_pred",
            true_dbp_col="dbp",
            pred_dbp_col="dbp_pred",
        )

        rows.append({
            "protocol": protocol,
            "model": "CatBoost",
            "evaluation_subset": "P3 holdout target-record subset",
            **m,
            "notes": f"Filtered from complete holdout predictions using {len(target_record_ids)} P3 target record_id values.",
        })

    required_naive_cols = {"sbp", "dbp", "calibration_sbp", "calibration_dbp"}
    missing_naive_cols = required_naive_cols.difference(set(p3.columns))
    if missing_naive_cols:
        raise RuntimeError(
            "Cannot compute P3-naive carry-forward baseline because the following "
            f"columns are missing from P3 predictions: {sorted(missing_naive_cols)}"
        )

    p3_naive_m = metric_from_prediction_df(
        p3,
        true_sbp_col="sbp",
        pred_sbp_col="calibration_sbp",
        true_dbp_col="dbp",
        pred_dbp_col="calibration_dbp",
    )
    rows.append({
        "protocol": "P3-naive same-session calibration carry-forward baseline",
        "model": "Carry-forward",
        "evaluation_subset": "P3 holdout target-record subset",
        **p3_naive_m,
        "notes": "Naive baseline using the same-subject calibration SBP/DBP value as the prediction for each target record.",
    })

    p3_m = metric_from_prediction_df(
        p3,
        true_sbp_col="sbp",
        pred_sbp_col="sbp_pred",
        true_dbp_col="dbp",
        pred_dbp_col="dbp_pred",
    )
    rows.append({
        "protocol": "P3 one-record subject-specific calibration protocol",
        "model": "CatBoost",
        "evaluation_subset": "P3 holdout target-record subset",
        **p3_m,
        "notes": "Original P3 calibrated target-record evaluation.",
    })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "same_subset_sensitivity_catboost_full_precision.csv", index=False)
    round_numeric(out).to_csv(OUT_DIR / "same_subset_sensitivity_catboost_rounded.csv", index=False)

    return out


# -----------------------------
# Task 6: Bland-Altman
# -----------------------------

def create_p3_predictions_for_export(
    p3_predictions: pd.DataFrame,
    feature_frame: pd.DataFrame,
) -> pd.DataFrame:
    p3 = normalize_p3_pred(p3_predictions)
    frame = feature_frame.copy()
    frame["record_id"] = frame["record_id"].astype(str)

    meta_candidates = [
        "record_id",
        "age_clean", "sex_clean", "height_clean", "weight_clean", "bmi",
        "dx_htn_clean", "drug_binary", "pre_smoke_clean", "pre_coffee_clean",
    ]
    meta_cols = [c for c in meta_candidates if c in frame.columns]

    export = p3.rename(
        columns={
            "sbp": "reference_sbp",
            "dbp": "reference_dbp",
            "sbp_pred": "predicted_sbp",
            "dbp_pred": "predicted_dbp",
            "sbp_error": "error_sbp",
            "dbp_error": "error_dbp",
        }
    )

    if meta_cols:
        export = export.merge(frame[meta_cols], on="record_id", how="left")

    desired = [
        "subject_id", "record_id", "calibration_record_id",
        "target_order_after_calibration",
        "calibration_sbp", "calibration_dbp",
        "reference_sbp", "predicted_sbp", "error_sbp",
        "reference_dbp", "predicted_dbp", "error_dbp",
        "age_clean", "sex_clean", "height_clean", "weight_clean", "bmi",
        "dx_htn_clean", "drug_binary", "pre_smoke_clean", "pre_coffee_clean",
    ]
    desired = [c for c in desired if c in export.columns]
    export = export[desired]

    export.to_csv(OUT_DIR / "p3_catboost_holdout_predictions.csv", index=False)
    return export


def create_bland_altman(p3_export: pd.DataFrame) -> pd.DataFrame:
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, target, label in [
        (axes[0], "sbp", "SBP"),
        (axes[1], "dbp", "DBP"),
    ]:
        ref = p3_export[f"reference_{target}"].to_numpy(dtype=float)
        pred = p3_export[f"predicted_{target}"].to_numpy(dtype=float)
        error = pred - ref
        mean_bp = (pred + ref) / 2.0

        mean_error = float(np.mean(error))
        sd_error = float(np.std(error, ddof=1))
        lower = mean_error - 1.96 * sd_error
        upper = mean_error + 1.96 * sd_error

        rows.append({
            "target": label,
            "records": int(len(p3_export)),
            "subjects": int(p3_export["subject_id"].astype(str).nunique()),
            "mean_error": mean_error,
            "sd_error": sd_error,
            "lower_loa": lower,
            "upper_loa": upper,
        })

        ax.scatter(mean_bp, error, s=12, alpha=0.65)
        ax.axhline(mean_error, linestyle="--", linewidth=1.2)
        ax.axhline(lower, linestyle=":", linewidth=1.2)
        ax.axhline(upper, linestyle=":", linewidth=1.2)
        ax.set_xlabel(f"Mean of reference and predicted {label} (mmHg)")
        ax.set_ylabel(f"Predicted - reference {label} (mmHg)")
        ax.set_title(f"{label} Bland–Altman")
        ax.text(
            0.02,
            0.98,
            f"ME={mean_error:.2f}\nLoA=[{lower:.2f}, {upper:.2f}]",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(OUT_DIR / "bland_altman_p3_catboost.png", dpi=300)
    plt.close(fig)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "bland_altman_p3_catboost_summary_full_precision.csv", index=False)
    round_numeric(out).to_csv(OUT_DIR / "bland_altman_p3_catboost_summary_rounded.csv", index=False)

    return out


def create_additional_validation_statistics(
    p3_export: pd.DataFrame,
    n_boot: int = 10_000,
    seed: int = 20260904,
) -> dict[str, Any]:
    """Quantify proportional bias and paired P3-versus-naive MAE differences.

    Bootstrap resampling is performed at the participant level so that all target
    records from a sampled participant remain together.
    """
    df = p3_export.copy()
    df["subject_id"] = df["subject_id"].astype(str)

    def slope(x: np.ndarray, y: np.ndarray) -> float:
        return float(np.polyfit(x, y, 1)[0])

    for target in ["sbp", "dbp"]:
        df[f"mean_{target}"] = (
            df[f"reference_{target}"] + df[f"predicted_{target}"]
        ) / 2.0
        df[f"naive_error_{target}"] = (
            df[f"calibration_{target}"] - df[f"reference_{target}"]
        )
        df[f"paired_abs_improvement_{target}"] = (
            df[f"naive_error_{target}"].abs() - df[f"error_{target}"].abs()
        )

    subjects = df["subject_id"].drop_duplicates().to_numpy()
    by_subject = {sid: group for sid, group in df.groupby("subject_id", sort=False)}
    rng = np.random.default_rng(seed)
    boot = {
        "sbp_slope": [], "dbp_slope": [],
        "sbp_mae_improvement": [], "dbp_mae_improvement": [],
        "mae_sum_improvement": [],
    }

    for _ in range(n_boot):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        sample = pd.concat([by_subject[sid] for sid in sampled], ignore_index=True)
        sbp_gain = float(sample["paired_abs_improvement_sbp"].mean())
        dbp_gain = float(sample["paired_abs_improvement_dbp"].mean())
        boot["sbp_slope"].append(slope(sample["mean_sbp"], sample["error_sbp"]))
        boot["dbp_slope"].append(slope(sample["mean_dbp"], sample["error_dbp"]))
        boot["sbp_mae_improvement"].append(sbp_gain)
        boot["dbp_mae_improvement"].append(dbp_gain)
        boot["mae_sum_improvement"].append(sbp_gain + dbp_gain)

    def ci(values: list[float]) -> list[float]:
        return [float(v) for v in np.quantile(values, [0.025, 0.975])]

    result: dict[str, Any] = {
        "records": int(len(df)),
        "subjects": int(len(subjects)),
        "bootstrap": {
            "resampling_unit": "participant",
            "replicates": int(n_boot),
            "seed": int(seed),
            "interval": "percentile 95% confidence interval",
        },
    }

    for target in ["sbp", "dbp"]:
        q = pd.qcut(df[f"mean_{target}"], 4, labels=False, duplicates="drop")
        result[target] = {
            "error_mean_correlation": float(
                df[f"mean_{target}"].corr(df[f"error_{target}"])
            ),
            "error_mean_slope": slope(df[f"mean_{target}"], df[f"error_{target}"]),
            "error_mean_slope_ci": ci(boot[f"{target}_slope"]),
            "lowest_mean_quartile_error": float(
                df.loc[q == 0, f"error_{target}"].mean()
            ),
            "highest_mean_quartile_error": float(
                df.loc[q == 3, f"error_{target}"].mean()
            ),
            "naive_mae": float(df[f"naive_error_{target}"].abs().mean()),
            "p3_mae": float(df[f"error_{target}"].abs().mean()),
            "paired_mae_improvement": float(
                df[f"paired_abs_improvement_{target}"].mean()
            ),
            "paired_mae_improvement_ci": ci(boot[f"{target}_mae_improvement"]),
        }

    result["mae_sum_improvement"] = {
        "estimate": float(
            df["paired_abs_improvement_sbp"].mean()
            + df["paired_abs_improvement_dbp"].mean()
        ),
        "ci": ci(boot["mae_sum_improvement"]),
    }
    write_json(result, OUT_DIR / "additional_validation_statistics.json")
    return result


# -----------------------------
# Task 7: subgroup
# -----------------------------

def yes_no_from_binary(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    out = pd.Series(pd.NA, index=s.index, dtype="object")
    out.loc[x == 0] = "No"
    out.loc[x == 1] = "Yes"
    return out


def create_subgroup_analysis(p3_export: pd.DataFrame) -> pd.DataFrame:
    df = p3_export.copy()

    if "age_clean" not in df.columns:
        raise RuntimeError("age_clean not found in P3 export; cannot compute age groups.")
    if "bmi" not in df.columns:
        raise RuntimeError("bmi not found in P3 export; cannot compute BMI groups.")
    if "dx_htn_clean" not in df.columns:
        raise RuntimeError("dx_htn_clean not found in P3 export; cannot compute hypertension subgroup.")
    if "drug_binary" not in df.columns:
        raise RuntimeError("drug_binary not found in P3 export; cannot compute medication subgroup.")

    age = pd.to_numeric(df["age_clean"], errors="coerce")
    bmi = pd.to_numeric(df["bmi"], errors="coerce")

    df["diagnosed_hypertension"] = yes_no_from_binary(df["dx_htn_clean"])
    df["bp_related_medication_use"] = yes_no_from_binary(df["drug_binary"])

    df["age_group"] = pd.cut(
        age,
        bins=[-np.inf, 45, 60, np.inf],
        labels=["<45", "45–<60", "≥60"],
        right=False,
    ).astype("object")

    df["bmi_group"] = pd.cut(
        bmi,
        bins=[-np.inf, 24.0, 28.0, np.inf],
        labels=["Normal", "Overweight", "Obese"],
        right=False,
    ).astype("object")

    subgroup_defs = [
        ("diagnosed hypertension", "diagnosed_hypertension", ["No", "Yes"]),
        ("BP-related medication use", "bp_related_medication_use", ["No", "Yes"]),
        ("age group", "age_group", ["<45", "45–<60", "≥60"]),
        ("BMI group", "bmi_group", ["Normal", "Overweight", "Obese"]),
    ]

    rows = []

    for subgroup_variable, col, order in subgroup_defs:
        for subgroup in order:
            part = df[df[col] == subgroup].copy()
            if len(part) == 0:
                continue

            m = metric_from_prediction_df(
                part,
                true_sbp_col="reference_sbp",
                pred_sbp_col="predicted_sbp",
                true_dbp_col="reference_dbp",
                pred_dbp_col="predicted_dbp",
            )

            rows.append({
                "subgroup_variable": subgroup_variable,
                "subgroup": subgroup,
                "records": m["evaluation_records"],
                "subjects": m["evaluation_subjects"],
                "sbp_mae": m["sbp_mae"],
                "sbp_rmse": m["sbp_rmse"],
                "sbp_r2": m["sbp_r2"],
                "dbp_mae": m["dbp_mae"],
                "dbp_rmse": m["dbp_rmse"],
                "dbp_r2": m["dbp_r2"],
                "mae_sum": m["mae_sum"],
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "subgroup_error_p3_catboost_full_precision.csv", index=False)
    round_numeric(out).to_csv(OUT_DIR / "subgroup_error_p3_catboost_rounded.csv", index=False)

    latex_cols = [
        "subgroup_variable", "subgroup", "records", "subjects",
        "sbp_mae", "dbp_mae", "mae_sum",
    ]
    write_latex_table(
        round_numeric(out[latex_cols]),
        OUT_DIR / "subgroup_error_p3_catboost_latex_table.txt",
    )

    return out


# -----------------------------
# Task 8: final report
# -----------------------------

def best_model_summary(p1p2: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for protocol in p1p2["protocol"].unique():
        sub = p1p2[p1p2["protocol"] == protocol].sort_values("mae_sum")
        rows.append(sub.iloc[0])
    return pd.DataFrame(rows)


def create_final_report(
    dataset_flow: pd.DataFrame,
    public_integrity_info: dict[str, Any],
    p1p2: pd.DataFrame,
    protocol_main: pd.DataFrame,
    same_subset: pd.DataFrame,
    ba: pd.DataFrame,
    subgroup: pd.DataFrame,
) -> None:
    best = best_model_summary(p1p2)

    generated_files = sorted([p.name for p in OUT_DIR.iterdir() if p.is_file()])

    lines = [
        "# Final Scientific Data experiment report",
        "",
        "## 1. Input files",
        "",
        f"- Public pkl: `{PUBLIC_PKL}`",
        f"- Structured summary: `{STRUCTURED_SUMMARY}`",
        f"- Structured predictions: `{STRUCTURED_PREDICTIONS}`",
        f"- P3 summary: `{P3_SUMMARY}`",
        f"- P3 predictions: `{P3_PREDICTIONS}`",
        f"- P3 sample flow: `{P3_SAMPLE_FLOW}`",
        f"- Feature frame for subgroup metadata: `{FEATURE_FRAME}`",
        "",
        "## 2. Output directory",
        "",
        f"`{OUT_DIR}`",
        "",
        "## 3. Subject-level split and leakage control",
        "",
        "The existing participant-disjoint fixed model-development / holdout split was preserved. "
        "P1 and P2 were evaluated on the complete fixed holdout set. "
        "P3 was evaluated on target records after excluding one calibration record per holdout subject. "
        "No record-level random split was introduced.",
        "",
        "## 4. Dataset flow audit",
        "",
        md_table(dataset_flow),
        "",
        "## 5. Public dataset integrity summary",
        "",
        f"- Records: {public_integrity_info['records']}",
        f"- Subjects: {public_integrity_info['subjects']}",
        f"- Columns: {public_integrity_info['columns']}",
        f"- ECG length distribution: {public_integrity_info['ecg_lengths']}",
        f"- PPG length distribution: {public_integrity_info['ppg_lengths']}",
        "",
        "Detailed public dataset integrity outputs:",
        "",
        "- `public_dataset_integrity_summary.csv`",
        "- `public_dataset_key_field_missingness.csv`",
        "- `public_dataset_metadata_missingness.csv`",
        "- `public_dataset_label_counts.csv`",
        "- `public_dataset_integrity_report.md`",
        "",
        "## 6. P1/P2 all-model benchmark",
        "",
        md_table(round_numeric(p1p2)),
        "",
        "Best model by protocol according to MAE sum:",
        "",
        md_table(round_numeric(best[["protocol", "model", "evaluation_subjects", "evaluation_records", "sbp_mae", "dbp_mae", "mae_sum"]])),
        "",
        "## 7. Main protocol comparison",
        "",
        md_table(round_numeric(protocol_main[[
            "protocol", "model", "evaluation_subjects", "evaluation_records",
            "sbp_mae", "sbp_rmse", "sbp_r2", "dbp_mae", "dbp_rmse", "dbp_r2", "mae_sum",
        ]])),
        "",
        "## 8. Same-subset sensitivity check",
        "",
        "This sensitivity check evaluates P1, P2, the P3-naive carry-forward baseline, and P3 on the same P3 holdout target-record subset. "
        "P1/P2 predictions were filtered to the P3 target record IDs; no retraining was performed. "
        "The P3-naive baseline directly uses the same-subject calibration SBP/DBP value as the prediction for each target record.",
        "",
        md_table(round_numeric(same_subset[[
            "protocol", "model", "evaluation_subjects", "evaluation_records",
            "sbp_mae", "sbp_rmse", "sbp_r2", "dbp_mae", "dbp_rmse", "dbp_r2", "mae_sum",
        ]])),
        "",
        "## 9. P3 Bland–Altman summary",
        "",
        md_table(round_numeric(ba)),
        "",
        "Figure:",
        "",
        "- `bland_altman_p3_catboost.png`",
        "",
        "## 10. P3 subgroup error analysis",
        "",
        md_table(round_numeric(subgroup)),
        "",
        "## 11. Generated files",
        "",
    ]

    for name in generated_files:
        lines.append(f"- `{name}`")

    lines.extend([
        "",
        "## 12. Reporting conventions",
        "",
        "- Use “final benchmark cohort”.",
        "- Use “model-development set” and “participant-disjoint fixed holdout set”.",
        "- Use “P1 signal-only no-calibration protocol”.",
        "- Use “P2 metadata-assisted individual-prior protocol”.",
        "- Use “P3-naive same-session calibration carry-forward baseline”.",
        "- Use “P3 one-record subject-specific calibration protocol”.",
        "- Use “P3 calibrated protocol” or “P3 calibrated benchmark” rather than “best model” in formal manuscript text when appropriate.",
        "- Do not report AAMI.",
        "- Do not report healthy-person subgroup analysis.",
        "- Do not report explainability analysis.",
        "- Do not do raw-window dual-window DL in this revision.",
        "",
    ])

    (OUT_DIR / "final_scidata_experiment_report.md").write_text("\n".join(lines), encoding="utf-8")


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_inputs()

    print("[LOAD] structured summary")
    summary = pd.read_csv(STRUCTURED_SUMMARY)

    print("[LOAD] structured predictions")
    structured_predictions = pd.read_csv(STRUCTURED_PREDICTIONS)

    print("[LOAD] P3 summary / predictions / flow")
    p3_summary = pd.read_csv(P3_SUMMARY)
    p3_predictions = pd.read_csv(P3_PREDICTIONS)
    p3_flow = read_json(P3_SAMPLE_FLOW)

    print("[LOAD] feature frame")
    feature_frame = pd.read_parquet(FEATURE_FRAME)

    print("[TASK 2] Public dataset integrity")
    public_integrity_info = create_public_dataset_integrity_outputs()

    print("[TASK 3] P1/P2 all-model benchmark")
    p1p2 = extract_p1_p2_all_model_benchmark(summary)

    print("[TASK 4] Main protocol comparison")
    protocol_main = create_main_protocol_comparison(summary, p3_summary)

    print("[TASK 1] Dataset flow audit")
    p1_catboost = get_summary_row(summary, "ecg_ppg_descriptors_timing", "catboost")
    dataset_flow = create_dataset_flow_audit(p1_catboost, p3_flow, public_integrity_info)

    print("[TASK 5] Same-subset sensitivity")
    same_subset = create_same_subset_sensitivity(structured_predictions, p3_predictions)

    print("[TASK 6] P3 prediction export + Bland-Altman")
    p3_export = create_p3_predictions_for_export(p3_predictions, feature_frame)
    ba = create_bland_altman(p3_export)

    print("[TASK 6B] Proportional-bias and participant-cluster bootstrap statistics")
    additional_validation = create_additional_validation_statistics(p3_export)

    print("[TASK 7] P3 subgroup analysis")
    subgroup = create_subgroup_analysis(p3_export)

    print("[TASK 8] Final report")
    create_final_report(
        dataset_flow=dataset_flow,
        public_integrity_info=public_integrity_info,
        p1p2=p1p2,
        protocol_main=protocol_main,
        same_subset=same_subset,
        ba=ba,
        subgroup=subgroup,
    )

    config = {
        "public_pkl": str(PUBLIC_PKL),
        "structured_summary": str(STRUCTURED_SUMMARY),
        "structured_predictions": str(STRUCTURED_PREDICTIONS),
        "p3_summary": str(P3_SUMMARY),
        "p3_predictions": str(P3_PREDICTIONS),
        "p3_sample_flow": str(P3_SAMPLE_FLOW),
        "feature_frame": str(FEATURE_FRAME),
        "output_dir": str(OUT_DIR),
    }
    write_json(config, OUT_DIR / "final_scidata_experiment_config.json")

    print("\n[DONE] Final outputs saved in:", OUT_DIR)

    print("\n[protocol_comparison_main_rounded.csv]")
    print(pd.read_csv(OUT_DIR / "protocol_comparison_main_rounded.csv").to_string(index=False))

    print("\n[model_family_benchmark_p1_p2_rounded.csv]")
    print(pd.read_csv(OUT_DIR / "model_family_benchmark_p1_p2_rounded.csv").to_string(index=False))

    print("\n[same_subset_sensitivity_catboost_rounded.csv]")
    print(pd.read_csv(OUT_DIR / "same_subset_sensitivity_catboost_rounded.csv").to_string(index=False))

    print("\n[bland_altman_p3_catboost_summary_rounded.csv]")
    print(pd.read_csv(OUT_DIR / "bland_altman_p3_catboost_summary_rounded.csv").to_string(index=False))

    print("\n[subgroup_error_p3_catboost_rounded.csv]")
    print(pd.read_csv(OUT_DIR / "subgroup_error_p3_catboost_rounded.csv").to_string(index=False))

    print("\n[additional_validation_statistics.json]")
    print(json.dumps(additional_validation, indent=2))

    print("\n[Final report]")
    print(OUT_DIR / "final_scidata_experiment_report.md")

    print("\n[Bland-Altman figure]")
    print(OUT_DIR / "bland_altman_p3_catboost.png")


if __name__ == "__main__":
    main()
