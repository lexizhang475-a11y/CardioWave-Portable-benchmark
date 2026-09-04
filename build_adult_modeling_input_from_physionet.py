#!/usr/bin/env python
"""Build the adult-only modeling pickle from the final PhysioNet release package.

WFDB signals are intentionally loaded as stored digital int32 samples.  The
physical values exposed by the WFDB header are display-normalized units and are
not the values used by the manuscript modeling pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


RECORD_MAP = {
    "card_type": "card",
    "source_start_index_inclusive": "start_idx",
    "source_end_index_exclusive": "end_idx",
    "sbp_mmHg": "sbp",
    "dbp_mmHg": "dbp",
    "ecg_quality_label": "ecg_label",
    "ppg_quality_label": "ppg_label",
}

SUBJECT_MAP = {
    "sex_code": "sex",
    "age_years": "age",
    "height_cm": "height",
    "weight_kg": "weight",
    "diagnosed_hypertension": "dx_htn",
    "family_history_hypertension": "fh_htn",
    "recent_week_bp_available": "bp_measured",
    "bp_medication": "drug_binary",
    "smoking_status_code": "smoke_status",
    "alcohol_use_code": "alcohol_status",
    "coffee_tea_code": "coffee_tea",
    "pre_measurement_smoking": "pre_smoke",
    "pre_measurement_coffee_tea": "pre_coffee",
    "pre_measurement_exercise": "pre_exercise",
    "reported_unwell": "unwell",
    "hypertension_duration_years": "dx_htn_years",
    "family_members_with_hypertension": "fh_htn_persons",
    "recent_week_sbp_mmHg": "sbp_week",
    "recent_week_dbp_mmHg": "dbp_week",
    "smoking_cessation_duration_years": "smoke_quit_year",
    "cigarettes_per_day": "smoke_cigs_day",
}

OUTPUT_COLUMNS = [
    "subject_id", "record_id", "card", "start_idx", "end_idx", "ecg", "ppg",
    "sbp", "dbp", "ecg_label", "ppg_label", "sex", "age", "height", "weight",
    "dx_htn", "fh_htn", "bp_measured", "drug_binary", "smoke_status",
    "alcohol_status", "coffee_tea", "pre_smoke", "pre_coffee", "pre_exercise",
    "unwell", "dx_htn_years", "fh_htn_persons", "sbp_week", "dbp_week",
    "smoke_quit_year", "smoke_cigs_day",
]

INTEGER_COLUMNS = [
    "height", "weight", "dx_htn", "fh_htn", "bp_measured", "drug_binary",
    "smoke_status", "alcohol_status", "coffee_tea", "pre_smoke", "pre_coffee",
    "pre_exercise", "unwell", "fh_htn_persons", "sbp_week", "dbp_week",
]
FLOAT_COLUMNS = [
    "start_idx", "end_idx", "sbp", "dbp", "ecg_label", "ppg_label", "sex",
    "age", "dx_htn_years", "smoke_quit_year", "smoke_cigs_day",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_header(path: Path, expected_record_id: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    fields = lines[0].split()
    if len(fields) < 4 or fields[:4] != [expected_record_id, "2", "500", "12000"]:
        raise ValueError(f"Unexpected WFDB header in {path}: {lines[0]!r}")
    signal_lines = lines[1:3]
    if len(signal_lines) != 2 or not all(line.split()[1] == "32" for line in signal_lines):
        raise ValueError(f"Expected two 32-bit signals in {path}")


def load_digital_signals(base_path: Path, record_id: str) -> tuple[np.ndarray, np.ndarray]:
    header_path = base_path.with_suffix(".hea")
    data_path = base_path.with_suffix(".dat")
    if not header_path.is_file() or not data_path.is_file():
        raise FileNotFoundError(f"Missing WFDB pair for {base_path}")
    parse_header(header_path, record_id)
    raw = np.fromfile(data_path, dtype="<i4")
    if raw.size != 12000 * 2:
        raise ValueError(f"Unexpected signal size for {record_id}: {raw.size}")
    samples = raw.reshape(12000, 2)
    return samples[:, 0].copy(), samples[:, 1].copy()


def write_fixed_split(source: Path, destination: Path, cohort_subjects: set[str]) -> None:
    split = pd.read_csv(source, dtype={"subject_id": str, "split": str})
    split = split[split["subject_id"].isin(cohort_subjects)].copy()
    if set(split["subject_id"]) != cohort_subjects:
        missing = sorted(cohort_subjects - set(split["subject_id"]))
        raise ValueError(f"Fixed split is missing adult benchmark subjects: {missing[:10]}")
    if split["subject_id"].duplicated().any():
        raise ValueError("Fixed split contains duplicated subjects")
    destination.parent.mkdir(parents=True, exist_ok=True)
    split.sort_values("subject_id").to_csv(destination, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physionet-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/cardiowave_portable_dataset.pkl"))
    parser.add_argument(
        "--original-split",
        type=Path,
        default=Path("resources/manuscript_subject_level_split_original.csv"),
    )
    parser.add_argument(
        "--fixed-split-output",
        type=Path,
        default=Path("resources/manuscript_subject_level_split.csv"),
    )
    args = parser.parse_args()

    root = args.physionet_root.resolve()
    records_path = root / "metadata" / "records.csv"
    subjects_path = root / "subject-info.csv"
    records = pd.read_csv(records_path, dtype=str)
    subjects = pd.read_csv(subjects_path, dtype=str)
    if records["record_id"].duplicated().any() or subjects["subject_id"].duplicated().any():
        raise ValueError("Duplicate record_id or subject_id in PhysioNet metadata")

    subjects["age_years"] = subjects["age_years"].replace({"90+": "90"})
    adult_ages = pd.to_numeric(subjects["age_years"], errors="coerce")
    if adult_ages.isna().any() or (adult_ages < 18).any():
        raise ValueError("PhysioNet modeling input is not completely adult or has missing age")

    frame = records.merge(subjects, on="subject_id", how="left", validate="many_to_one")
    if frame["age_years"].isna().any():
        raise ValueError("Some records lack participant metadata")
    frame = frame.rename(columns={**RECORD_MAP, **SUBJECT_MAP})
    for col in INTEGER_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("Int64")
    for col in FLOAT_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("Float64")

    ecg_values: list[np.ndarray] = []
    ppg_values: list[np.ndarray] = []
    for idx, row in frame.iterrows():
        base = root / str(row["wfdb_record_path"])
        ecg, ppg = load_digital_signals(base, str(row["record_id"]))
        ecg_values.append(ecg)
        ppg_values.append(ppg)
        if (idx + 1) % 500 == 0:
            print(f"Validated and loaded {idx + 1}/{len(frame)} records", flush=True)
    frame["ecg"] = ecg_values
    frame["ppg"] = ppg_values
    frame = frame[OUTPUT_COLUMNS].copy()

    if len(frame) != 3849 or frame["subject_id"].nunique() != 977:
        raise ValueError(f"Unexpected adult release size: {len(frame)} records, {frame['subject_id'].nunique()} subjects")
    if pd.to_numeric(frame["age"], errors="coerce").min() < 18:
        raise ValueError("Under-18 participant found in output")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(args.output)

    # The benchmark cohort is smaller than the release after BP/stability filters.
    bp_ok = frame["sbp"].astype(float).between(80, 190) & frame["dbp"].astype(float).between(40, 110)
    bp_frame = frame.loc[bp_ok]
    ranges = bp_frame.groupby("subject_id").agg(sbp_min=("sbp", "min"), sbp_max=("sbp", "max"), dbp_min=("dbp", "min"), dbp_max=("dbp", "max"))
    keep = ranges.index[((ranges["sbp_max"] - ranges["sbp_min"]) < 25) & ((ranges["dbp_max"] - ranges["dbp_min"]) < 20)]
    write_fixed_split(args.original_split, args.fixed_split_output, set(map(str, keep)))

    manifest = {
        "source_release": str(root),
        "source_records_sha256": sha256(records_path),
        "source_subject_info_sha256": sha256(subjects_path),
        "waveform_read_mode": "WFDB digital int32 samples (physical=False equivalent)",
        "sampling_frequency_hz": 500,
        "samples_per_signal": 12000,
        "signal_names": ["ECG", "PPG"],
        "adult_definition": "age_years >= 18",
        "record_count": int(len(frame)),
        "subject_count": int(frame["subject_id"].nunique()),
        "minimum_age_years": float(pd.to_numeric(frame["age"], errors="coerce").min()),
        "maximum_age_years_for_modeling": float(pd.to_numeric(frame["age"], errors="coerce").max()),
        "bp_range_record_count": int(len(bp_frame)),
        "bp_range_subject_count": int(bp_frame["subject_id"].nunique()),
        "benchmark_subject_count": int(len(keep)),
        "modeling_pickle": str(args.output.resolve()),
        "modeling_pickle_sha256": sha256(args.output),
        "fixed_split": str(args.fixed_split_output.resolve()),
        "fixed_split_sha256": sha256(args.fixed_split_output),
        "record_column_mapping": RECORD_MAP,
        "subject_column_mapping": SUBJECT_MAP,
    }
    manifest_path = args.output.parent / "adult_modeling_input_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
