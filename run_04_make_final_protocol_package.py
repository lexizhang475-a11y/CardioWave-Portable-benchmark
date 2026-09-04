#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate final Scientific Data protocol revision package.

This script organizes existing and newly generated results into:
1. main_protocol_comparison.csv
2. main_protocol_comparison_rounded_2dec.csv
3. structured_protocol_predictions.csv
4. raw_window_protocol_predictions.csv
5. final_best_model_predictions.csv
6. final_best_model_predictions_rounded_2dec.csv
7. bland_altman_summary.csv
8. bland_altman_summary_rounded_2dec.csv
9. bland_altman_final_best_model.png
10. subgroup_final_best_model.csv
11. subgroup_final_best_model_rounded_2dec.csv
12. protocol_revision_short_report.md

Final converged protocol:
- P1: structured ECG+PPG signal-only, no calibration
- P2: structured ECG+PPG+metadata, metadata-assisted weak individual prior
- P3: structured dual-window 1-of-4 subject-specific calibration

Raw-window DL results are included only as baseline reference rows if available
from the current manuscript/report; no raw-window dual-window DL protocol is added.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OUT_DIR = Path("outputs/04_final_protocol_package/final_protocol_package_v1")

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

A3_SUMMARY = Path(
    "outputs/03_p3_calibration/"
    "structured_a3_1of4_calibration_v1/"
    "structured_a3_summary_structured_a3_1of4_calibration_v1.csv"
)

A3_PREDICTIONS = Path(
    "outputs/03_p3_calibration/"
    "structured_a3_1of4_calibration_v1/"
    "structured_a3_predictions_structured_a3_1of4_calibration_v1.csv"
)

A3_SAMPLE_FLOW = Path(
    "outputs/03_p3_calibration/"
    "structured_a3_1of4_calibration_v1/"
    "structured_a3_sample_flow_structured_a3_1of4_calibration_v1.json"
)

FEATURE_FRAME = Path(
    "outputs/01_physio_feature_ablation/"
    "physio_feature_strict_v1/"
    "physio_feature_frame_physio_feature_strict_v1.parquet"
)


def ensure_inputs() -> None:
    required = [
        STRUCTURED_SUMMARY,
        STRUCTURED_PREDICTIONS,
        A3_SUMMARY,
        A3_PREDICTIONS,
        A3_SAMPLE_FLOW,
        FEATURE_FRAME,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def r2_score_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return np.nan
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom <= 1e-12:
        return np.nan
    num = np.sum((y_true - y_pred) ** 2)
    return float(1.0 - num / denom)


def metric_summary(df: pd.DataFrame) -> dict[str, float]:
    sbp_true = df["true_sbp"].to_numpy(dtype=float)
    sbp_pred = df["pred_sbp"].to_numpy(dtype=float)
    dbp_true = df["true_dbp"].to_numpy(dtype=float)
    dbp_pred = df["pred_dbp"].to_numpy(dtype=float)

    sbp_err = sbp_pred - sbp_true
    dbp_err = dbp_pred - dbp_true

    sbp_mae = float(np.mean(np.abs(sbp_err)))
    dbp_mae = float(np.mean(np.abs(dbp_err)))

    return {
        "records": int(len(df)),
        "subjects": int(df["subject_id"].astype(str).nunique()),
        "sbp_me": float(np.mean(sbp_err)),
        "sbp_sd": float(np.std(sbp_err, ddof=1)) if len(sbp_err) > 1 else 0.0,
        "sbp_mae": sbp_mae,
        "sbp_rmse": float(np.sqrt(np.mean(sbp_err ** 2))),
        "sbp_r2": r2_score_safe(sbp_true, sbp_pred),
        "dbp_me": float(np.mean(dbp_err)),
        "dbp_sd": float(np.std(dbp_err, ddof=1)) if len(dbp_err) > 1 else 0.0,
        "dbp_mae": dbp_mae,
        "dbp_rmse": float(np.sqrt(np.mean(dbp_err ** 2))),
        "dbp_r2": r2_score_safe(dbp_true, dbp_pred),
        "mae_sum": sbp_mae + dbp_mae,
    }


def round_df(df: pd.DataFrame, digits: int = 2) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(digits)
    return out


def protocol_row_from_structured(
    row: pd.Series,
    protocol_level: str,
    protocol_name: str,
    input_information: str,
    calibration_setting: str,
) -> dict[str, Any]:
    return {
        "representation_family": "structured-feature ML",
        "protocol_level": protocol_level,
        "protocol_name": protocol_name,
        "input_information": input_information,
        "calibration_setting": calibration_setting,
        "model_name": row["model_name"],
        "holdout_subjects": int(row["holdout_subjects"]),
        "holdout_records": int(row["holdout_records"]),
        "sbp_mae": float(row["sbp_mae"]),
        "sbp_rmse": float(row["sbp_rmse"]),
        "sbp_r2": float(row["sbp_r2"]),
        "dbp_mae": float(row["dbp_mae"]),
        "dbp_rmse": float(row["dbp_rmse"]),
        "dbp_r2": float(row["dbp_r2"]),
        "mae_sum": float(row["mae_sum"]),
        "source": "redefined structured-feature full matrix",
        "notes": "",
    }


def build_protocol_comparison() -> pd.DataFrame:
    summary = pd.read_csv(STRUCTURED_SUMMARY)
    a3 = pd.read_csv(A3_SUMMARY).iloc[0]

    a1_row = summary[
        (summary["feature_set"] == "ecg_ppg_descriptors_timing")
        & (summary["model_name"] == "catboost")
    ].iloc[0]

    a2_row = summary[
        (summary["feature_set"] == "full_structured")
        & (summary["model_name"] == "catboost")
    ].iloc[0]

    rows: list[dict[str, Any]] = []

    rows.append(
        protocol_row_from_structured(
            a1_row,
            protocol_level="P1",
            protocol_name="ECG+PPG structured signal-only, no calibration",
            input_information="ECG descriptors + PPG descriptors + ECG-PPG timing descriptors",
            calibration_setting="No subject-specific calibration; no metadata",
        )
    )

    rows.append(
        protocol_row_from_structured(
            a2_row,
            protocol_level="P2",
            protocol_name="ECG+PPG structured metadata-assisted weak individual prior",
            input_information="ECG descriptors + PPG descriptors + ECG-PPG timing descriptors + metadata",
            calibration_setting="Metadata-assisted weak individual prior; no same-subject calibration record",
        )
    )

    rows.append(
        {
            "representation_family": "structured-feature ML",
            "protocol_level": "P3",
            "protocol_name": "ECG+PPG structured dual-window 1-of-4 subject-specific calibration",
            "input_information": (
                "target ECG/PPG/timing descriptors + calibration ECG/PPG/timing descriptors "
                "+ target-calibration descriptor differences + calibration BP label"
            ),
            "calibration_setting": "First record of each subject used as calibration; remaining records used as targets",
            "model_name": str(a3["model_name"]),
            "holdout_subjects": int(a3["subjects"]),
            "holdout_records": int(a3["records"]),
            "sbp_mae": float(a3["sbp_mae"]),
            "sbp_rmse": float(a3["sbp_rmse"]),
            "sbp_r2": float(a3["sbp_r2"]),
            "dbp_mae": float(a3["dbp_mae"]),
            "dbp_rmse": float(a3["dbp_rmse"]),
            "dbp_r2": float(a3["dbp_r2"]),
            "mae_sum": float(a3["mae_sum"]),
            "source": "new structured dual-window calibration experiment",
            "notes": "Evaluated on target records after excluding one calibration record per holdout subject",
        }
    )

    # Raw-window DL rows are retained as reference baselines from the current manuscript/report.
    # B1 DBP R2 was not available in the current manuscript table; keep it as NaN rather than guessing.
    rows.append(
        {
            "representation_family": "raw-window DL baseline",
            "protocol_level": "B1-reference",
            "protocol_name": "ECG+PPG raw-window waveform-only, no calibration",
            "input_information": "target 24-s ECG waveform + target 24-s PPG waveform",
            "calibration_setting": "No subject-specific calibration; no metadata",
            "model_name": "TCN",
            "holdout_subjects": 168,
            "holdout_records": 650,
            "sbp_mae": 13.3478,
            "sbp_rmse": 17.0009,
            "sbp_r2": 0.2738,
            "dbp_mae": 9.1609,
            "dbp_rmse": 11.7259,
            "dbp_r2": np.nan,
            "mae_sum": 13.3478 + 9.1609,
            "source": "existing raw-window DL baseline table",
            "notes": "Reference baseline only; DBP R2 not available in the current manuscript table",
        }
    )

    rows.append(
        {
            "representation_family": "raw-window DL baseline",
            "protocol_level": "B2-reference",
            "protocol_name": "ECG+PPG raw-window metadata-assisted weak individual prior",
            "input_information": "target 24-s ECG waveform + target 24-s PPG waveform + metadata",
            "calibration_setting": "Metadata-assisted weak individual prior; no same-subject calibration record",
            "model_name": "TCN",
            "holdout_subjects": 168,
            "holdout_records": 650,
            "sbp_mae": 12.1825,
            "sbp_rmse": 15.3940,
            "sbp_r2": 0.4046,
            "dbp_mae": 8.9767,
            "dbp_rmse": 11.3660,
            "dbp_r2": 0.2253,
            "mae_sum": 21.1592,
            "source": "existing raw-window DL baseline table",
            "notes": "Reference baseline only",
        }
    )

    out = pd.DataFrame(rows)
    out = out.sort_values(["representation_family", "protocol_level"]).reset_index(drop=True)
    return out


def build_structured_protocol_predictions() -> pd.DataFrame:
    pred = pd.read_csv(STRUCTURED_PREDICTIONS)
    rows = []

    mapping = [
        ("P1", "ECG+PPG structured signal-only, no calibration", "ecg_ppg_descriptors_timing"),
        ("P2", "ECG+PPG structured metadata-assisted weak individual prior", "full_structured"),
    ]

    for level, name, feature_set in mapping:
        sub = pred[
            (pred["feature_set"] == feature_set)
            & (pred["model_name"] == "catboost")
        ].copy()
        sub = sub.rename(
            columns={
                "sbp": "true_sbp",
                "dbp": "true_dbp",
                "sbp_pred": "pred_sbp",
                "dbp_pred": "pred_dbp",
            }
        )
        sub["protocol_level"] = level
        sub["protocol_name"] = name
        sub["calibration_record_id"] = ""
        sub["error_sbp"] = sub["pred_sbp"] - sub["true_sbp"]
        sub["error_dbp"] = sub["pred_dbp"] - sub["true_dbp"]
        rows.append(
            sub[
                [
                    "protocol_level",
                    "protocol_name",
                    "subject_id",
                    "record_id",
                    "calibration_record_id",
                    "true_sbp",
                    "pred_sbp",
                    "error_sbp",
                    "true_dbp",
                    "pred_dbp",
                    "error_dbp",
                ]
            ]
        )

    a3 = pd.read_csv(A3_PREDICTIONS).copy()
    a3 = a3.rename(
        columns={
            "sbp": "true_sbp",
            "dbp": "true_dbp",
            "sbp_pred": "pred_sbp",
            "dbp_pred": "pred_dbp",
            "sbp_error": "error_sbp",
            "dbp_error": "error_dbp",
        }
    )
    a3["protocol_level"] = "P3"
    a3["protocol_name"] = "ECG+PPG structured dual-window 1-of-4 subject-specific calibration"
    rows.append(
        a3[
            [
                "protocol_level",
                "protocol_name",
                "subject_id",
                "record_id",
                "calibration_record_id",
                "true_sbp",
                "pred_sbp",
                "error_sbp",
                "true_dbp",
                "pred_dbp",
                "error_dbp",
            ]
        ]
    )

    return pd.concat(rows, ignore_index=True)


def build_final_best_predictions() -> pd.DataFrame:
    a3 = pd.read_csv(A3_PREDICTIONS).copy()
    out = a3.rename(
        columns={
            "sbp": "true_sbp",
            "dbp": "true_dbp",
            "sbp_pred": "pred_sbp",
            "dbp_pred": "pred_dbp",
            "sbp_error": "error_sbp",
            "dbp_error": "error_dbp",
        }
    )

    frame = pd.read_parquet(FEATURE_FRAME).copy()
    frame["record_id"] = frame["record_id"].astype(str)
    out["record_id"] = out["record_id"].astype(str)

    meta_cols = [
        "record_id",
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
    meta_cols = [c for c in meta_cols if c in frame.columns]

    out = out.merge(frame[meta_cols], on="record_id", how="left")

    out["protocol_level"] = "P3"
    out["protocol_name"] = "ECG+PPG structured dual-window 1-of-4 subject-specific calibration"
    out["final_best_model"] = "CatBoost"
    out["final_best_model_reason"] = "lowest SBP+DBP MAE sum among organized protocols"

    cols = [
        "protocol_level",
        "protocol_name",
        "final_best_model",
        "subject_id",
        "record_id",
        "calibration_record_id",
        "target_order_after_calibration",
        "calibration_sbp",
        "calibration_dbp",
        "true_sbp",
        "pred_sbp",
        "error_sbp",
        "true_dbp",
        "pred_dbp",
        "error_dbp",
    ] + [c for c in meta_cols if c != "record_id"] + ["final_best_model_reason"]

    cols = [c for c in cols if c in out.columns]
    return out[cols]


def make_bland_altman(final_pred: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows = []

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, target, label in [
        (axes[0], "sbp", "SBP"),
        (axes[1], "dbp", "DBP"),
    ]:
        true = final_pred[f"true_{target}"].to_numpy(dtype=float)
        pred = final_pred[f"pred_{target}"].to_numpy(dtype=float)
        error = pred - true
        mean_bp = (pred + true) / 2.0

        me = float(np.mean(error))
        sd = float(np.std(error, ddof=1))
        lower = me - 1.96 * sd
        upper = me + 1.96 * sd

        rows.append(
            {
                "target": label,
                "mean_error": me,
                "sd_error": sd,
                "loa_lower": lower,
                "loa_upper": upper,
                "n_records": int(len(final_pred)),
                "n_subjects": int(final_pred["subject_id"].astype(str).nunique()),
            }
        )

        ax.scatter(mean_bp, error, s=12, alpha=0.65)
        ax.axhline(me, linestyle="--", linewidth=1.2)
        ax.axhline(lower, linestyle=":", linewidth=1.2)
        ax.axhline(upper, linestyle=":", linewidth=1.2)
        ax.set_xlabel(f"Mean of reference and predicted {label} (mmHg)")
        ax.set_ylabel(f"Prediction error for {label} (mmHg)")
        ax.set_title(f"({ 'A' if target == 'sbp' else 'B' }) {label} Bland-Altman")
        ax.text(
            0.02,
            0.98,
            f"ME={me:.2f}\nLoA=[{lower:.2f}, {upper:.2f}]",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_dir / "bland_altman_final_best_model.png", dpi=300)
    plt.close(fig)

    return pd.DataFrame(rows)


def subgroup_metrics(final_pred: pd.DataFrame) -> pd.DataFrame:
    df = final_pred.copy()

    df["age_group"] = pd.cut(
        pd.to_numeric(df["age_clean"], errors="coerce"),
        bins=[-np.inf, 45, 60, np.inf],
        labels=["<45", "45-<60", ">=60"],
        right=False,
    ).astype("string")

    df["bmi_group"] = pd.cut(
        pd.to_numeric(df["bmi"], errors="coerce"),
        bins=[-np.inf, 24.0, 28.0, np.inf],
        labels=["Normal BMI<24.0", "Overweight 24.0<=BMI<28.0", "Obese BMI>=28.0"],
        right=False,
    ).astype("string")

    dx = pd.to_numeric(df["dx_htn_clean"], errors="coerce")
    df["diagnosed_hypertension"] = np.where(dx >= 0.5, "Yes", "No")
    df.loc[dx.isna(), "diagnosed_hypertension"] = pd.NA

    drug = pd.to_numeric(df["drug_binary"], errors="coerce")
    df["bp_related_medication_use"] = np.where(drug >= 0.5, "Yes", "No")
    df.loc[drug.isna(), "bp_related_medication_use"] = pd.NA

    subgroup_defs = [
        ("diagnosed hypertension", "diagnosed_hypertension", ["No", "Yes"]),
        ("BP-related medication use", "bp_related_medication_use", ["No", "Yes"]),
        ("age group", "age_group", ["<45", "45-<60", ">=60"]),
        (
            "BMI group",
            "bmi_group",
            ["Normal BMI<24.0", "Overweight 24.0<=BMI<28.0", "Obese BMI>=28.0"],
        ),
    ]

    rows = []

    for var_label, col, order in subgroup_defs:
        for group in order:
            part = df[df[col] == group].copy()
            if len(part) == 0:
                continue
            m = metric_summary(part)
            rows.append(
                {
                    "subgroup_variable": var_label,
                    "subgroup": group,
                    "records": m["records"],
                    "subjects": m["subjects"],
                    "sbp_mae": m["sbp_mae"],
                    "sbp_rmse": m["sbp_rmse"],
                    "sbp_r2": m["sbp_r2"],
                    "dbp_mae": m["dbp_mae"],
                    "dbp_rmse": m["dbp_rmse"],
                    "dbp_r2": m["dbp_r2"],
                    "mae_sum": m["mae_sum"],
                }
            )

    return pd.DataFrame(rows)


def write_short_report(
    output_dir: Path,
    protocol_table: pd.DataFrame,
    ba: pd.DataFrame,
    subgroup: pd.DataFrame,
) -> None:
    a3 = protocol_table[protocol_table["protocol_level"] == "P3"].iloc[0]
    a1 = protocol_table[protocol_table["protocol_level"] == "P1"].iloc[0]
    a2 = protocol_table[protocol_table["protocol_level"] == "P2"].iloc[0]

    lines = [
        "# Protocol revision short report",
        "",
        "## What was changed",
        "",
        "The revised Scientific Data experiment is organized around a structured-feature machine-learning protocol hierarchy:",
        "",
        "1. P1: ECG+PPG structured signal-only, no calibration.",
        "2. P2: ECG+PPG structured features plus metadata, interpreted as a metadata-assisted weak individual-prior protocol.",
        "3. P3: ECG+PPG structured dual-window 1-of-4 subject-specific calibration.",
        "",
        "Raw-window DL results are retained only as reference baselines. No raw-window dual-window DL protocol was added, because the requested calibration experiment is implemented as a machine-learning dual-window feature protocol.",
        "",
        "## Reused and newly added results",
        "",
        "- Reused: P1 from the existing CatBoost `ecg_ppg_descriptors_timing` result.",
        "- Reused: P2 from the existing CatBoost `full_structured` result.",
        "- Reused: raw-window DL ECG+PPG and ECG+PPG+metadata baseline rows from the existing manuscript/report.",
        "- Newly added: P3 structured dual-window 1-of-4 subject-specific calibration using CatBoost.",
        "- Newly generated: Bland-Altman summary and figure based on the P3 final best model.",
        "- Newly generated: subgroup analysis based on the P3 final best model.",
        "",
        "## Calibration protocol",
        "",
        "For P3, each subject's records were sorted deterministically by `record_id`. The first record was used as the calibration record, and the remaining records from the same subject were used as target records. The model input consisted of target ECG/PPG/timing descriptors, calibration ECG/PPG/timing descriptors, target-calibration descriptor differences, and the calibration BP label. The SBP model used `calibration_sbp`, and the DBP model used `calibration_dbp`.",
        "",
        "The original participant-disjoint fixed model-development / holdout split was preserved. No subject appeared in both sets. Each holdout target record used only the calibration record from the same holdout subject.",
        "",
        "## Main results",
        "",
        f"- P1 signal-only no calibration: SBP MAE={a1['sbp_mae']:.2f}, DBP MAE={a1['dbp_mae']:.2f}, MAE sum={a1['mae_sum']:.2f}.",
        f"- P2 metadata-assisted weak prior: SBP MAE={a2['sbp_mae']:.2f}, DBP MAE={a2['dbp_mae']:.2f}, MAE sum={a2['mae_sum']:.2f}.",
        f"- P3 dual-window calibration: SBP MAE={a3['sbp_mae']:.2f}, DBP MAE={a3['dbp_mae']:.2f}, MAE sum={a3['mae_sum']:.2f}.",
        "",
        "P3 achieved the lowest MAE sum and was therefore selected as the final best model for Bland-Altman and subgroup analyses. Note that P3 was evaluated on target records after excluding one calibration record per holdout subject.",
        "",
        "## Bland-Altman outputs",
        "",
        ba.to_markdown(index=False),
        "",
        "## Subgroup outputs",
        "",
        subgroup.to_markdown(index=False),
        "",
        "## Important writing notes",
        "",
        "- Do not call P3 a no-calibration model.",
        "- Do not report AAMI results.",
        "- Do not include healthy-subgroup analysis.",
        "- Do not include feature-importance or explainability analysis in the main revision.",
        "- Report final manuscript table values to two decimal places.",
        "",
    ]

    with open(output_dir / "protocol_revision_short_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    ensure_inputs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    protocol_table = build_protocol_comparison()
    protocol_table.to_csv(OUT_DIR / "main_protocol_comparison.csv", index=False)
    round_df(protocol_table, 2).to_csv(OUT_DIR / "main_protocol_comparison_rounded_2dec.csv", index=False)

    structured_pred = build_structured_protocol_predictions()
    structured_pred.to_csv(OUT_DIR / "structured_protocol_predictions.csv", index=False)
    round_df(structured_pred, 2).to_csv(OUT_DIR / "structured_protocol_predictions_rounded_2dec.csv", index=False)

    # Raw-window prediction rows are not available from the current project outputs.
    raw_placeholder = pd.DataFrame(
        columns=[
            "protocol_level",
            "protocol_name",
            "subject_id",
            "record_id",
            "true_sbp",
            "pred_sbp",
            "error_sbp",
            "true_dbp",
            "pred_dbp",
            "error_dbp",
            "note",
        ]
    )
    raw_placeholder.to_csv(OUT_DIR / "raw_window_protocol_predictions.csv", index=False)

    final_pred = build_final_best_predictions()
    final_pred.to_csv(OUT_DIR / "final_best_model_predictions.csv", index=False)
    round_df(final_pred, 2).to_csv(OUT_DIR / "final_best_model_predictions_rounded_2dec.csv", index=False)

    ba = make_bland_altman(final_pred, OUT_DIR)
    ba.to_csv(OUT_DIR / "bland_altman_summary.csv", index=False)
    round_df(ba, 2).to_csv(OUT_DIR / "bland_altman_summary_rounded_2dec.csv", index=False)

    subgroup = subgroup_metrics(final_pred)
    subgroup.to_csv(OUT_DIR / "subgroup_final_best_model.csv", index=False)
    round_df(subgroup, 2).to_csv(OUT_DIR / "subgroup_final_best_model_rounded_2dec.csv", index=False)

    config = {
        "output_dir": str(OUT_DIR),
        "structured_summary": str(STRUCTURED_SUMMARY),
        "structured_predictions": str(STRUCTURED_PREDICTIONS),
        "a3_summary": str(A3_SUMMARY),
        "a3_predictions": str(A3_PREDICTIONS),
        "a3_sample_flow": str(A3_SAMPLE_FLOW),
        "feature_frame": str(FEATURE_FRAME),
        "final_best_model": "P3 structured dual-window 1-of-4 subject-specific calibration",
    }
    save_json(config, OUT_DIR / "final_protocol_package_config.json")

    write_short_report(OUT_DIR, protocol_table, ba, subgroup)

    print("Saved final protocol package in:", OUT_DIR)
    for name in [
        "main_protocol_comparison.csv",
        "main_protocol_comparison_rounded_2dec.csv",
        "structured_protocol_predictions.csv",
        "raw_window_protocol_predictions.csv",
        "final_best_model_predictions.csv",
        "final_best_model_predictions_rounded_2dec.csv",
        "bland_altman_summary.csv",
        "bland_altman_summary_rounded_2dec.csv",
        "bland_altman_final_best_model.png",
        "subgroup_final_best_model.csv",
        "subgroup_final_best_model_rounded_2dec.csv",
        "protocol_revision_short_report.md",
    ]:
        print("-", OUT_DIR / name)

    print("\nMain protocol comparison:")
    cols = [
        "representation_family",
        "protocol_level",
        "protocol_name",
        "model_name",
        "holdout_subjects",
        "holdout_records",
        "sbp_mae",
        "sbp_rmse",
        "sbp_r2",
        "dbp_mae",
        "dbp_rmse",
        "dbp_r2",
        "mae_sum",
    ]
    print(protocol_table[cols].to_string(index=False))

    print("\nBland-Altman summary:")
    print(ba.to_string(index=False))

    print("\nSubgroup summary:")
    print(subgroup.to_string(index=False))


if __name__ == "__main__":
    main()
