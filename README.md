# Physics-Informed Shuffle Reservoir Spiking Neural Network for Battery SOH Estimation

This repository contains a physics-informed shuffle reservoir spiking neural network (PI-SRSNN) framework for lithium-ion battery state-of-health (SOH) estimation. The project combines event-driven spiking reservoir computing, grouped channel-shuffled recurrent connectivity, physics-guided battery degradation features, and lightweight ridge-regression readouts to estimate relative capacity across heterogeneous battery datasets.

The goal is to keep SOH prediction accurate while reducing training complexity compared with fully trainable recurrent deep networks such as LSTM, GRU, and CNN-LSTM models.

## Highlights

- **Physics-informed SOH estimation** using impedance growth, square-root-time aging, Arrhenius effective age, Nernst/OCV consistency, and physically plausible SOH clipping.
- **Shuffle reservoir spiking core** with leaky integrate-and-fire neurons, grouped block-diagonal recurrent connectivity, and deterministic channel shuffling.
- **Lightweight training** because the recurrent reservoir is fixed and only the readout is trained.
- **Cross-dataset evaluation** on NASA PCoE, CALCE cylindrical SP cells, and LG MJ1 dynamic degradation data.
- **Sub-percent prediction error** with an overall row-level RMSE of **0.370%** and a balanced-dataset RMSE of **0.345%** across **1,363 held-out samples from 29 cells/groups**.

## Reported Results

The latest saved evaluation artifacts report the following held-out performance:

| Dataset | Evaluation protocol | Test samples/groups | RMSE (%) | MAE (%) |
|---|---:|---:|---:|---:|
| NASA PCoE | Leave-one-battery-out | 636 rows / 4 batteries | 0.536 | 0.443 |
| CALCE SP cells | Hold-out battery | 14 rows / 2 cells | 0.468 | 0.216 |
| LG MJ1 | 70/30 cell-level split | 713 rows / 23 cells | 0.031 | 0.025 |
| **Overall row-level** | Pooled NASA + CALCE + LG | **1,363 rows / 29 groups** | **0.370** | **0.222** |
| **Balanced dataset** | Mean of dataset-level metrics | NASA + CALCE + LG equally weighted | **0.345** | **0.228** |

These values are stored in:

- `grandpipeline_summary.csv`
- `grandpipeline_crossdataset_summary.csv`
- `final_tune_best_config.json`
- `nasa_holdout_summary.csv`
- `calce_holdout_SP20_1_summary.csv`
- `calce_holdout_SP20_3_summary.csv`
- `lg_train70_summary.csv`

## Method Overview

The proposed PI-SRSNN pipeline has four main components.

### 1. Input Representation

Each battery record is converted into:

- a fixed-length temporal sequence of voltage, current, temperature or power;
- summary features such as duration, throughput, voltage/current statistics, temperature behavior, impedance indicators, and drift from earlier operating states.

### 2. Shuffle Reservoir Spiking Core

The temporal sequence is processed by a leaky integrate-and-fire reservoir. Instead of a fully dense recurrent matrix, the model uses grouped block-diagonal recurrence. A fixed channel-shuffle permutation mixes spike activity between groups at every recurrent update. This keeps the reservoir expressive while lowering recurrent synaptic cost.

### 3. Physics-Informed Guidance

Physics-guided features are concatenated with the reservoir representation. The implemented priors include:

- impedance/ECM SOH prior for NASA;
- square-root-time degradation prior;
- Arrhenius-weighted effective age for temperature-sensitive CALCE data;
- Nernst/OCV-consistent voltage-capacity estimate;
- final SOH range clipping and Coulomb-consistency style post-processing.

### 4. Lightweight Readout

The readout is a ridge-regression ensemble trained on standardized reservoir, summary, and physics features. Since the recurrent spiking reservoir is fixed, training is much faster and simpler than backpropagation through time.

## Repository Structure

```text
.
├── nasa_rsnn_pipeline.py          # Physics-informed shuffle RSNN pipeline for NASA PCoE
├── calce_rsnn_pipeline.py         # Physics-informed shuffle RSNN pipeline for CALCE SP cells
├── lg_rsnn_pipeline.py            # Physics-informed shuffle RSNN pipeline for LG MJ1
├── physics_layer.py               # Battery physics priors and output constraints
├── physics_readout.py             # Optional physics-loss linear readout
├── grandpipeline.py               # Cross-dataset aggregation and final metrics
├── apply_final_tune.py            # Regenerates tuned summaries and final grand metrics
├── final_tune_best_config.json    # Best saved tuning configuration
├── *_summary.csv                  # Saved evaluation summaries
├── *_predictions.csv              # Saved prediction outputs
└── requirements.txt               # Python dependencies
```

Raw datasets are intentionally not committed because they are large and should be downloaded from their original sources.

## Datasets

This project uses three public battery datasets:

1. **NASA PCoE Li-ion battery dataset**
   - Cells: B0005, B0006, B0007, B0018
   - Protocol: charge/discharge/impedance cycling
   - Evaluation: leave-one-battery-out

2. **CALCE cylindrical SP cell dataset**
   - Cells: SP20-1 and SP20-3
   - Protocol: low-current OCV, incremental OCV, and initial-capacity tests at multiple temperatures
   - Evaluation: hold-out cell

3. **LG MJ1 dynamic degradation dataset**
   - Dynamic and path-dependent drive-cycle profiles
   - Evaluation: 70/30 split at cell level

## Installation

Create a Python environment and install the dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the Saved Metrics

After placing the raw datasets in the expected local paths, run:

```bash
python apply_final_tune.py
```

or run the cross-dataset aggregation from existing artifacts:

```bash
python grandpipeline.py
```

The main output files are:

- `grandpipeline_predictions.csv`
- `grandpipeline_summary.csv`
- `grandpipeline_crossdataset_summary.csv`
- `grandpipeline_tuning_summary.csv`

## Key Result

The best saved configuration is:

```json
{
  "nasa_ridge": 50.0,
  "calce_ridge": 10.0,
  "n_seeds": 10.0,
  "aggregator": "mean",
  "coulomb_tol": 0.08,
  "overall_row_rmse_percent": 0.36984682395168267,
  "balanced_dataset_rmse_percent": 0.34509391900738645
}
```

## Notes on Computation and Energy

The model reduces training cost because the reservoir weights are fixed and only a ridge readout is trained. The grouped recurrent matrix reduces recurrent synaptic count by a factor related to the number of groups, while the channel shuffle preserves cross-group information flow without adding trainable parameters. This makes the architecture attractive for embedded battery-management applications where memory, computation, and switching activity matter.



