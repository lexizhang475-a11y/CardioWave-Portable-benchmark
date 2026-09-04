#!/usr/bin/env bash
set -euo pipefail

echo "============================================================"
echo "CardioWave-Portable full workflow: benchmark preparation and run_01 to run_05"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "============================================================"

# Limit CPU threads to avoid overloading local machines
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

mkdir -p logs
mkdir -p outputs

SUMMARY_LOG="logs/full_workflow_summary.log"

{
  echo "============================================================"
  echo "CardioWave-Portable full workflow summary"
  echo "Start time: $(date)"
  echo "Working directory: $(pwd)"
  echo "Python: $(which python)"
  python --version
  echo "============================================================"
  echo
} | tee "$SUMMARY_LOG"

echo "[PRECHECK] Checking required files..." | tee -a "$SUMMARY_LOG"

required_files=(
  "data/cardiowave_portable_dataset.pkl"
  "prepare_clean_benchmark_dataset.py"
  "check_benchmark_outputs.py"
  "run_01_extract_physio_features.py"
  "run_02_p1_p2_structured_benchmark.py"
  "run_03_p3_one_record_calibration.py"
  "run_04_make_final_protocol_package.py"
  "run_05_make_final_revision_outputs.py"
)

for f in "${required_files[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] Missing required file: $f" | tee -a "$SUMMARY_LOG"
    exit 1
  fi
  echo "[OK] $f" | tee -a "$SUMMARY_LOG"
done

echo | tee -a "$SUMMARY_LOG"

echo "[ENV] Saving pip freeze..." | tee -a "$SUMMARY_LOG"
pip freeze > logs/pip_freeze_full_workflow.txt

run_step() {
  local step_name="$1"
  local log_file="$2"
  shift 2

  echo "============================================================" | tee -a "$SUMMARY_LOG"
  echo "[START] $step_name" | tee -a "$SUMMARY_LOG"
  echo "Time: $(date)" | tee -a "$SUMMARY_LOG"
  echo "Log: $log_file" | tee -a "$SUMMARY_LOG"
  echo "Command: $*" | tee -a "$SUMMARY_LOG"
  echo "============================================================" | tee -a "$SUMMARY_LOG"

  local start_seconds
  start_seconds=$(date +%s)

  "$@" 2>&1 | tee "$log_file"

  local end_seconds
  end_seconds=$(date +%s)
  local elapsed=$((end_seconds - start_seconds))

  echo "============================================================" | tee -a "$SUMMARY_LOG"
  echo "[DONE] $step_name" | tee -a "$SUMMARY_LOG"
  echo "Elapsed seconds: $elapsed" | tee -a "$SUMMARY_LOG"
  echo "End time: $(date)" | tee -a "$SUMMARY_LOG"

  if grep -iE "traceback|error|failed|exception" "$log_file" > "logs/${step_name}_possible_errors.txt"; then
    echo "[WARN] Possible error keywords found in $log_file" | tee -a "$SUMMARY_LOG"
    echo "See: logs/${step_name}_possible_errors.txt" | tee -a "$SUMMARY_LOG"
  else
    echo "[OK] No obvious Traceback/Error/Failed/Exception keywords found." | tee -a "$SUMMARY_LOG"
  fi

  echo | tee -a "$SUMMARY_LOG"
}

run_step "prepare_clean_benchmark_dataset" "logs/prepare_clean_benchmark_dataset.log" \
  python prepare_clean_benchmark_dataset.py

run_step "check_benchmark_outputs" "logs/check_benchmark_outputs.log" \
  python check_benchmark_outputs.py

run_step "run_01_extract_physio_features" "logs/run_01_extract_physio_features.log" \
  python run_01_extract_physio_features.py \
    --data-path data/cardiowave_portable_dataset.pkl \
    --output-root outputs/01_physio_feature_ablation \
    --batch-id physio_feature_strict_v1 \
    --feature-sets meta_only,ecg_ppg_basic_meta,ecg_ppg_basic_physio,ecg_ppg_basic_physio_meta \
    --models catboost,randomforest \
    --run-cv true

run_step "run_02_p1_p2_structured_benchmark" "logs/run_02_p1_p2_structured_benchmark.log" \
  python run_02_p1_p2_structured_benchmark.py

run_step "run_03_p3_one_record_calibration" "logs/run_03_p3_one_record_calibration.log" \
  python run_03_p3_one_record_calibration.py

run_step "run_04_make_final_protocol_package" "logs/run_04_make_final_protocol_package.log" \
  python run_04_make_final_protocol_package.py

run_step "run_05_make_final_revision_outputs" "logs/run_05_make_final_revision_outputs.log" \
  python run_05_make_final_revision_outputs.py

run_step "check_benchmark_outputs_final" "logs/check_benchmark_outputs_final.log" \
  python check_benchmark_outputs.py

echo "============================================================" | tee -a "$SUMMARY_LOG"
echo "[POSTCHECK] Output directory sizes" | tee -a "$SUMMARY_LOG"
echo "============================================================" | tee -a "$SUMMARY_LOG"
du -sh outputs/* 2>/dev/null | tee -a "$SUMMARY_LOG" || true
echo | tee -a "$SUMMARY_LOG"

echo "============================================================" | tee -a "$SUMMARY_LOG"
echo "[POSTCHECK] Generated output files" | tee -a "$SUMMARY_LOG"
echo "============================================================" | tee -a "$SUMMARY_LOG"
find outputs -maxdepth 4 -type f | sort > logs/all_output_files.txt
cat logs/all_output_files.txt | tee -a "$SUMMARY_LOG"
echo | tee -a "$SUMMARY_LOG"

echo "============================================================" | tee -a "$SUMMARY_LOG"
echo "[POSTCHECK] Important CSV / JSON output preview" | tee -a "$SUMMARY_LOG"
echo "============================================================" | tee -a "$SUMMARY_LOG"

python - <<'PY' 2>&1 | tee logs/final_output_preview.log
from pathlib import Path
import json
import pandas as pd

paths = [
    Path("outputs/00_benchmark_cohort/benchmark_cohort_summary.json"),
    Path("outputs/01_physio_feature_ablation/physio_feature_strict_v1/physio_ablation_summary_physio_feature_strict_v1.csv"),
    Path("outputs/01_physio_feature_ablation/physio_feature_strict_v1/physio_ablation_holdout_metrics_physio_feature_strict_v1.csv"),
]

print("=== Existing selected outputs ===")
for p in paths:
    print(f"\n--- {p} ---")
    if not p.exists():
        print("MISSING")
        continue

    if p.suffix == ".json":
        with p.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        print(json.dumps(obj, ensure_ascii=False, indent=2)[:4000])
    elif p.suffix == ".csv":
        df = pd.read_csv(p)
        print("shape:", df.shape)
        print(df.head(20).to_string(index=False))

print("\n=== Search all final output csv/json files ===")
for p in sorted(Path("outputs").rglob("*")):
    if p.is_file() and p.suffix.lower() in [".csv", ".json"]:
        print(p)
PY

cat logs/final_output_preview.log >> "$SUMMARY_LOG"

echo | tee -a "$SUMMARY_LOG"
echo "============================================================" | tee -a "$SUMMARY_LOG"
echo "[ALL DONE] Full workflow completed successfully." | tee -a "$SUMMARY_LOG"
echo "End time: $(date)" | tee -a "$SUMMARY_LOG"
echo "Please send me: logs/full_workflow_summary.log and logs/final_output_preview.log" | tee -a "$SUMMARY_LOG"
echo "============================================================" | tee -a "$SUMMARY_LOG"
