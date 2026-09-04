from pathlib import Path
import json

import pandas as pd


ROOT = Path("outputs/00_benchmark_cohort")

REQUIRED_FILES = {
    "clean_record_table.csv",
    "final_benchmark_cohort.csv",
    "subject_level_split.csv",
    "p3_calibration_target_pairs.csv",
    "benchmark_cohort_summary.json",
    "benchmark_readme.txt",
}

print("=== Benchmark output directory check ===")
print("Root:", ROOT.resolve())

if not ROOT.exists():
    raise SystemExit("ERROR: outputs/00_benchmark_cohort does not exist. Run prepare_clean_benchmark_dataset.py first.")

files = sorted([p for p in ROOT.rglob("*") if p.is_file()])
missing_files = sorted(name for name in REQUIRED_FILES if not (ROOT / name).is_file())
if missing_files:
    raise SystemExit(f"ERROR: missing required benchmark outputs: {missing_files}")
print("\nGenerated files:")
for p in files:
    print("-", p)

print("\n=== Table summaries ===")

for p in files:
    suffix = p.suffix.lower()
    try:
        if suffix == ".csv":
            df = pd.read_csv(p)
        elif suffix in [".parquet", ".pq"]:
            df = pd.read_parquet(p)
        elif suffix == ".json":
            with p.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            print(f"\n{p}")
            print(obj)
            continue
        else:
            continue
    except Exception as e:
        print(f"\n{p}")
        print("Could not read:", e)
        continue

    print(f"\n{p}")
    print("shape:", df.shape)

    if "subject_id" in df.columns:
        print("subjects:", df["subject_id"].nunique())

    possible_split_cols = [
        c for c in df.columns
        if c.lower() in ["split", "set", "subset", "group", "fold", "data_split"]
        or "split" in c.lower()
        or "subset" in c.lower()
    ]

    for c in possible_split_cols:
        print(f"value counts for {c}:")
        print(df[c].value_counts(dropna=False))

    if {"subject_id", "split"}.issubset(df.columns):
        overlap_check = df.groupby("subject_id")["split"].nunique()
        n_overlap = int((overlap_check > 1).sum())
        print("subjects appearing in multiple splits:", n_overlap)

final_cohort = pd.read_csv(ROOT / "final_benchmark_cohort.csv")
split = pd.read_csv(ROOT / "subject_level_split.csv")
pairs = pd.read_csv(ROOT / "p3_calibration_target_pairs.csv")
record_split = final_cohort.merge(split, on="subject_id", how="left", validate="many_to_one")

expected = {
    "benchmark_records": 3265,
    "benchmark_subjects": 839,
    "development_records": 2615,
    "development_subjects": 671,
    "holdout_records": 650,
    "holdout_subjects": 168,
    "p3_holdout_targets": 482,
    "p3_holdout_subjects": 167,
}
observed = {
    "benchmark_records": len(final_cohort),
    "benchmark_subjects": final_cohort["subject_id"].nunique(),
    "development_records": int((record_split["split"] == "model_development").sum()),
    "development_subjects": record_split.loc[record_split["split"] == "model_development", "subject_id"].nunique(),
    "holdout_records": int((record_split["split"] == "independent_holdout").sum()),
    "holdout_subjects": record_split.loc[record_split["split"] == "independent_holdout", "subject_id"].nunique(),
    "p3_holdout_targets": int((pairs["split"] == "independent_holdout").sum()),
    "p3_holdout_subjects": pairs.loc[pairs["split"] == "independent_holdout", "subject_id"].nunique(),
}

if observed != expected:
    raise SystemExit(f"ERROR: benchmark counts differ from the manuscript. Expected {expected}; observed {observed}")

overlap = set(split.loc[split["split"] == "model_development", "subject_id"]) & set(
    split.loc[split["split"] == "independent_holdout", "subject_id"]
)
if overlap:
    raise SystemExit(f"ERROR: {len(overlap)} participants occur in both split subsets")

if (pairs["calibration_record_id"] == pairs["target_record_id"]).any():
    raise SystemExit("ERROR: a P3 calibration record is also used as its own target")

print("\nPASS: required benchmark files, manuscript counts, participant-disjoint split, and P3 pairs verified.")
