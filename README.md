# PTL water-quality forecasting

This repository contains the Python code used for the pretraining and progressive transfer learning (PTL) experiments on next-day multivariate water-quality forecasting.

The release is code-only. Monitoring data, processed datasets, trained weights, experiment outputs, and manuscript files are not included.

## Repository layout

- `src/PTL/`: cross-station masked pretraining, progressive transfer, fine-tuning, SHAP analysis, and experiment runners.
- `src/Base/models/`: MLP, CNN, LSTM, Bi-LSTM, CNN-LSTM, and Transformer baselines.
- `src/Base/benchmarks/`: station-level baseline training entry points.
- `src/Base/analysis/`: multi-station comparison, training-data-availability, summary, and figure scripts.
- `script/`: data preparation helpers and shell entry points.

## Environment

The code was tested with Python 3.13 and PyTorch 2.8. Install the remaining dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Expected data layout

Most experiment runners use repository-relative paths and expect locally prepared files under `data/`. A typical layout is:

```text
data/
├── data_cleaned/
│   └── yangzte/
└── water_quality_processed_2021_2024/
    ├── 4h/
    ├── daily/
    ├── weekly/
    └── station_meta.csv
```

Station CSV files use a timestamp column followed by `CODMn`, `DO`, `NH4N`, and `pH`. Data paths can be changed through the command-line arguments exposed by each runner.

## Main entry points

Run commands from the repository root. Use `--help` to inspect the complete options before launching an experiment.

```bash
# Cross-station masked pretraining
python src/PTL/Pretrain_cross_station_v2.py --help
python src/PTL/Pretrain_cross_station_v3.py --help

# Progressive target-station adaptation
python src/PTL/run_pearl_other_stations_core3_progressive_v2pretrain_v2_2021_2024.py --help

# Six-model baseline comparison
python src/Base/analysis/run_strict_six_model_baselines.py --help

# Contiguous training-history availability experiment
python src/Base/analysis/run_tail_training_availability_ptl_17stations.py --help

# Input-dependence analysis for trained daily PTL models
python src/PTL/run_shap_analysis_17stations.py --help
```

Training outputs are written below `results/` by default. That directory is ignored by Git.

## Data availability

No monitoring records are redistributed in this repository. The code can be run with data prepared in the layout above, subject to the access conditions and permissions of the original data provider.
