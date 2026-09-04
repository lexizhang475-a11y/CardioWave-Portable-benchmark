#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prepare a clean benchmark cohort from the public CardioWave-Portable dataset.

Input:
    data/cardiowave_portable_dataset.pkl

Outputs:
    outputs/00_benchmark_cohort/
    ├── clean_record_table.csv
    ├── final_benchmark_cohort.csv
    ├── subject_level_split.csv
    ├── p3_calibration_target_pairs.csv
    ├── benchmark_cohort_summary.json
    └── benchmark_readme.txt

Purpose:
    This script demonstrates the standard benchmark data preparation steps used for
    the public CardioWave-Portable dataset:

    1. Load the de-identified public record-level pkl.
    2. Derive cleaned metadata variables.
    3. Apply BP range filtering.
    4. Apply within-subject BP stability filtering.
    5. Apply the fixed manuscript participant-disjoint model-development / holdout split,
       with the original holdout membership preserved for the adult-only cohort.
    6. Create calibration-target pairs for the one-record subject-specific calibration protocol.

Notes:
    - This script intentionally does not include proprietary/internal paths.
    - This script does not perform ECG/PPG landmark extraction.
    - This script does not claim to reproduce every model metric from the manuscript by itself.
      It prepares the cleaned benchmark cohort and split tables from the public release.
    - Subject-level splitting is required to avoid leakage across repeated records from the
      same subject.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


INPUT_PKL = Path("data/cardiowave_portable_dataset.pkl")
OUT_DIR = Path("outputs/00_benchmark_cohort")

RANDOM_SEED = 42
HOLDOUT_RATIO = 0.20

# BP filtering rules used for the benchmark cohort.
SBP_MIN = 80.0
SBP_MAX = 190.0
DBP_MIN = 40.0
DBP_MAX = 110.0

# Within-subject stability rule.
MAX_WITHIN_SUBJECT_SBP_RANGE = 25.0
MAX_WITHIN_SUBJECT_DBP_RANGE = 20.0


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def clean_binary(series: pd.Series) -> pd.Series:
    """
    Convert a questionnaire-coded binary field into 0/1/NA.

    Accepts numeric values 0 and 1. Other values are set to missing.
    """
    x = pd.to_numeric(series, errors="coerce")
    out = pd.Series(pd.NA, index=series.index, dtype="Int64")
    out.loc[x == 0] = 0
    out.loc[x == 1] = 1
    return out


def derive_clean_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive harmonized benchmark metadata variables from the public pkl fields.

    Public raw/lightly processed fields:
        sex, age, height, weight, dx_htn, drug_binary, pre_smoke, pre_coffee

    Derived benchmark variables:
        sex_clean, age_clean, height_clean, weight_clean, bmi,
        dx_htn_clean, drug_binary_clean, pre_smoke_clean, pre_coffee_clean
    """
    out = df.copy()

    out["age_clean"] = to_numeric(out["age"]) if "age" in out.columns else np.nan
    out.loc[~out["age_clean"].between(0, 120), "age_clean"] = np.nan

    # sex coding in public data dictionary:
    # 0=female, 1=male, 2=not available
    out["sex_clean"] = to_numeric(out["sex"]) if "sex" in out.columns else np.nan
    out.loc[~out["sex_clean"].isin([0, 1, 2]), "sex_clean"] = np.nan

    out["height_clean"] = to_numeric(out["height"]) if "height" in out.columns else np.nan
    out.loc[~out["height_clean"].between(130, 220), "height_clean"] = np.nan

    out["weight_clean"] = to_numeric(out["weight"]) if "weight" in out.columns else np.nan
    out.loc[~out["weight_clean"].between(30, 200), "weight_clean"] = np.nan

    height_m = out["height_clean"] / 100.0
    out["bmi"] = out["weight_clean"] / (height_m ** 2)
    out.loc[~out["bmi"].between(10, 80), "bmi"] = np.nan

    out["dx_htn_clean"] = clean_binary(out["dx_htn"]) if "dx_htn" in out.columns else pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["drug_binary_clean"] = clean_binary(out["drug_binary"]) if "drug_binary" in out.columns else pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["pre_smoke_clean"] = clean_binary(out["pre_smoke"]) if "pre_smoke" in out.columns else pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["pre_coffee_clean"] = clean_binary(out["pre_coffee"]) if "pre_coffee" in out.columns else pd.Series(pd.NA, index=out.index, dtype="Int64")

    return out


def apply_bp_range_filter(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sbp"] = to_numeric(out["sbp"])
    out["dbp"] = to_numeric(out["dbp"])

    mask = (
        out["sbp"].between(SBP_MIN, SBP_MAX)
        & out["dbp"].between(DBP_MIN, DBP_MAX)
    )
    return out.loc[mask].copy()


def apply_within_subject_stability_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep subjects whose repeated records are within the predefined BP stability range:
        SBP range < 25 mmHg
        DBP range < 20 mmHg
    """
    stats = (
        df.groupby("subject_id", as_index=True)
        .agg(
            records=("record_id", "count"),
            sbp_min=("sbp", "min"),
            sbp_max=("sbp", "max"),
            dbp_min=("dbp", "min"),
            dbp_max=("dbp", "max"),
        )
    )
    stats["sbp_range"] = stats["sbp_max"] - stats["sbp_min"]
    stats["dbp_range"] = stats["dbp_max"] - stats["dbp_min"]

    keep_subjects = stats[
        (stats["sbp_range"] < MAX_WITHIN_SUBJECT_SBP_RANGE)
        & (stats["dbp_range"] < MAX_WITHIN_SUBJECT_DBP_RANGE)
    ].index

    return df[df["subject_id"].isin(keep_subjects)].copy()


def make_subject_level_split(df: pd.DataFrame) -> pd.DataFrame:
    """
    Load the fixed manuscript subject-level split.

    The manuscript benchmark uses a predefined subject-level split recovered from
    the final validated benchmark results. This avoids regenerating a different
    split and ensures that the public benchmark cohort matches the reported
    model-development and participant-disjoint fixed holdout sets.
    """
    split_path = Path("resources/manuscript_subject_level_split.csv")

    if not split_path.exists():
        raise FileNotFoundError(
            "Missing resources/manuscript_subject_level_split.csv. "
            "This file is required to reproduce the manuscript benchmark split."
        )

    split = pd.read_csv(split_path)

    required = {"subject_id", "split"}
    missing = required - set(split.columns)
    if missing:
        raise ValueError(f"Manuscript split file is missing columns: {sorted(missing)}")

    split = split[["subject_id", "split"]].copy()
    split["subject_id"] = split["subject_id"].astype(str)

    valid_labels = {"model_development", "independent_holdout"}
    bad_labels = sorted(set(split["split"].dropna()) - valid_labels)
    if bad_labels:
        raise ValueError(f"Unexpected split labels: {bad_labels}")

    cohort_subjects = set(df["subject_id"].astype(str).unique())
    split_subjects = set(split["subject_id"].astype(str).unique())

    missing_subjects = sorted(cohort_subjects - split_subjects)
    extra_subjects = sorted(split_subjects - cohort_subjects)

    if missing_subjects:
        raise ValueError(
            f"Manuscript split is missing {len(missing_subjects)} cohort subjects. "
            f"Examples: {missing_subjects[:10]}"
        )

    if extra_subjects:
        raise ValueError(
            f"Manuscript split contains {len(extra_subjects)} subjects not in cohort. "
            f"Examples: {extra_subjects[:10]}"
        )

    if split["subject_id"].duplicated().any():
        dup = split.loc[split["subject_id"].duplicated(), "subject_id"].head(10).tolist()
        raise ValueError(f"Duplicated subject_id values in manuscript split: {dup}")

    return split.sort_values("subject_id").reset_index(drop=True)


def assign_split(df: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(split[["subject_id", "split"]], on="subject_id", how="left")
    if out["split"].isna().any():
        raise RuntimeError("Some records were not assigned to a split.")
    return out


def sort_record_key(record_id: Any) -> tuple[int, str]:
    """
    Deterministic within-subject record ordering based on public record_id.
    Public record_id values have the form R000001, R000002, ...
    """
    text = str(record_id)
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        return int(digits), text
    return 10**12, text


def make_p3_calibration_target_pairs(df_with_split: pd.DataFrame) -> pd.DataFrame:
    """
    Build calibration-target pairs for the P3 one-record subject-specific calibration protocol.

    For each subject within each split:
        - Sort records deterministically by record_id.
        - Use the first record as calibration record.
        - Use all remaining records as target records.
        - Subjects with only one eligible record do not contribute target pairs.
    """
    rows = []

    for split_name, split_df in df_with_split.groupby("split"):
        for subject_id, subject_df in split_df.groupby("subject_id"):
            subject_df = subject_df.copy()
            subject_df["_sort_key"] = subject_df["record_id"].map(sort_record_key)
            subject_df = subject_df.sort_values("_sort_key").drop(columns=["_sort_key"])

            if len(subject_df) < 2:
                continue

            calib = subject_df.iloc[0]

            for target_order, (_, target) in enumerate(subject_df.iloc[1:].iterrows(), start=1):
                rows.append({
                    "split": split_name,
                    "subject_id": subject_id,
                    "calibration_record_id": calib["record_id"],
                    "target_record_id": target["record_id"],
                    "target_order_after_calibration": target_order,
                    "calibration_sbp": calib["sbp"],
                    "calibration_dbp": calib["dbp"],
                    "target_sbp": target["sbp"],
                    "target_dbp": target["dbp"],
                    "calibration_card": calib.get("card", pd.NA),
                    "target_card": target.get("card", pd.NA),
                })

    return pd.DataFrame(rows)


def summarize(df_public: pd.DataFrame, df_clean: pd.DataFrame, df_final: pd.DataFrame, split: pd.DataFrame, pairs: pd.DataFrame) -> dict[str, Any]:
    split_counts = (
        df_final.merge(split[["subject_id", "split"]], on="subject_id", how="left")
        .groupby("split")
        .agg(subjects=("subject_id", "nunique"), records=("record_id", "count"))
        .reset_index()
    )

    pair_counts = (
        pairs.groupby("split")
        .agg(
            subjects=("subject_id", "nunique"),
            target_records=("target_record_id", "count"),
            calibration_records=("calibration_record_id", "nunique"),
        )
        .reset_index()
        if len(pairs)
        else pd.DataFrame(columns=["split", "subjects", "target_records", "calibration_records"])
    )

    return {
        "input_file": str(INPUT_PKL),
        "public_dataset": {
            "subjects": int(df_public["subject_id"].nunique()),
            "records": int(len(df_public)),
            "columns": int(df_public.shape[1]),
        },
        "after_bp_range_filter": {
            "subjects": int(df_clean["subject_id"].nunique()),
            "records": int(len(df_clean)),
        },
        "final_benchmark_cohort": {
            "subjects": int(df_final["subject_id"].nunique()),
            "records": int(len(df_final)),
        },
        "split_counts": split_counts.to_dict(orient="records"),
        "p3_pair_counts": pair_counts.to_dict(orient="records"),
        "filtering_rules": {
            "sbp_range_mmHg": [SBP_MIN, SBP_MAX],
            "dbp_range_mmHg": [DBP_MIN, DBP_MAX],
            "within_subject_sbp_range_rule": f"< {MAX_WITHIN_SUBJECT_SBP_RANGE} mmHg",
            "within_subject_dbp_range_rule": f"< {MAX_WITHIN_SUBJECT_DBP_RANGE} mmHg",
        },
        "split_rule": {
            "type": "fixed manuscript subject-level split",
            "holdout_ratio": HOLDOUT_RATIO,
            "seed": RANDOM_SEED,
            "policy": "preserve original participant-disjoint fixed holdout; remove only subjects absent from the adult-only benchmark cohort",
        },
        "p3_rule": {
            "calibration_record": "first eligible record after deterministic within-subject record_id ordering",
            "target_records": "remaining eligible records from the same subject",
            "calibration_record_evaluated_as_target": False,
        },
    }


def write_benchmark_readme(summary: dict[str, Any]) -> None:
    text = f"""Clean benchmark preparation outputs
===================================

This folder was generated by prepare_clean_benchmark_dataset.py.

Input public dataset:
    {summary["input_file"]}

Public dataset:
    subjects = {summary["public_dataset"]["subjects"]}
    records  = {summary["public_dataset"]["records"]}

Final benchmark cohort after BP range and within-subject stability filtering:
    subjects = {summary["final_benchmark_cohort"]["subjects"]}
    records  = {summary["final_benchmark_cohort"]["records"]}

Generated files:
    clean_record_table.csv
        Public records with cleaned benchmark metadata variables and BP range filtering status.

    final_benchmark_cohort.csv
        Final benchmark cohort after BP range filtering and within-subject BP stability filtering.

    subject_level_split.csv
        Deterministic participant-disjoint fixed model-development / holdout split.

    p3_calibration_target_pairs.csv
        Calibration-target pair table for the P3 one-record subject-specific calibration protocol.

    benchmark_cohort_summary.json
        Machine-readable summary of cohort counts, filtering rules, split rules, and P3 pairing rules.

Important notes:
    - Use subject-level splitting to avoid leakage.
    - Do not place records from the same subject in both model-development and fixed holdout sets.
    - For P3 calibration experiments, the calibration record must not be evaluated as a target record.
    - The released questionnaire metadata are raw or lightly processed; cleaned variables are derived here.
"""
    (OUT_DIR / "benchmark_readme.txt").write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_PKL.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_PKL}. Place this script in the same folder as "
            "data/cardiowave_portable_dataset.pkl."
        )

    print("[LOAD]", INPUT_PKL)
    public = pd.read_pickle(INPUT_PKL)

    public["subject_id"] = public["subject_id"].astype(str)
    public["record_id"] = public["record_id"].astype(str)

    print("[STEP 1] Derive cleaned metadata")
    clean = derive_clean_metadata(public)

    print("[STEP 2] Apply BP range filter")
    bp_filtered = apply_bp_range_filter(clean)

    print("[STEP 3] Apply within-subject BP stability filter")
    final_cohort = apply_within_subject_stability_filter(bp_filtered)

    print("[STEP 4] Apply fixed manuscript subject-level split")
    split = make_subject_level_split(final_cohort)
    final_with_split = assign_split(final_cohort, split)

    print("[STEP 5] Create P3 calibration-target pairs")
    pairs = make_p3_calibration_target_pairs(final_with_split)

    print("[SAVE] CSV outputs")
    clean.to_csv(OUT_DIR / "clean_record_table.csv", index=False)
    final_cohort.to_csv(OUT_DIR / "final_benchmark_cohort.csv", index=False)
    split.to_csv(OUT_DIR / "subject_level_split.csv", index=False)
    pairs.to_csv(OUT_DIR / "p3_calibration_target_pairs.csv", index=False)

    summary = summarize(public, bp_filtered, final_cohort, split, pairs)

    with open(OUT_DIR / "benchmark_cohort_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    write_benchmark_readme(summary)

    print("\nDone.")
    print("Output directory:", OUT_DIR)
    print("Public dataset:", summary["public_dataset"])
    print("After BP range filter:", summary["after_bp_range_filter"])
    print("Final benchmark cohort:", summary["final_benchmark_cohort"])
    print("Split counts:", summary["split_counts"])
    print("P3 pair counts:", summary["p3_pair_counts"])


if __name__ == "__main__":
    main()
