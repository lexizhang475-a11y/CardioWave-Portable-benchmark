# CardioWave-Portable adult benchmark code

This directory contains the code and fixed participant-level split resources used to generate the adult-only benchmark results reported in the manuscript. Controlled waveform data are not included.

## Input

Obtain authorized access to the CardioWave-Portable PhysioNet release and place or reference its project root, which must contain `RECORDS`, `metadata/records.csv`, `subject-info.csv`, and the WFDB waveform shards.

## Environment

Python 3.10 was used for the reported run. Install the listed packages with:

```bash
python -m pip install -r requirements.txt
```

## Reproduction

From this directory, reconstruct the modelling input from an authorized local copy of the PhysioNet release:

```bash
python build_adult_modeling_input_from_physionet.py \
  --physionet-root /path/to/CardioWave-Portable \
  --output data/cardiowave_portable_dataset.pkl
```

Then run the complete benchmark workflow:

```bash
bash run_full_workflow_01_05.sh
```

The workflow first runs `prepare_clean_benchmark_dataset.py` to generate `outputs/00_benchmark_cohort/clean_record_table.csv`, `final_benchmark_cohort.csv`, `subject_level_split.csv`, and `p3_calibration_target_pairs.csv`. It then verifies those files, extracts physiological descriptors, runs the P1/P2 model-family benchmarks, runs P3 one-record calibration, and exports the final protocol, same-subset, Bland--Altman, proportional-bias, participant-cluster bootstrap, and subgroup results. The aggregate statistics reported in the revised manuscript are also supplied in `resources/additional_validation_statistics.json`; no participant-level predictions are included in that file.

## Data protection

Do not place controlled waveform data in a public repository or redistribute them with this code. Access and reuse remain governed by the PhysioNet credentialing and data-use agreement requirements.
