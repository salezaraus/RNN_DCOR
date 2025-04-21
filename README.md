A Distance Correlation–Based Toolkit for Characterizing RNN Effectiveness in Time Series Forecasting

This repository implements the methodology from:

A Distance Correlation-Based Approach to Characterize the Effectiveness of Recurrent Neural Networks for Time Series ForecastingChristopher Salazar, Ashis G. Banerjee (University of Washington) citeturn1file0

📋 Highlights

Evaluates internal RNN behavior using distance correlation to track information flow through activation layers.

Diagnoses memory loss: shows how RNNs learn and gradually forget lag structures over layers.

Benchmarks synthetic & real data: AR, MA, ARMA, GARCH processes; ETTh1, OT, Solar‑energy, NASDAQ indices.

Hyperparameter insights: input window size dominates over activation functions or hidden unit count.

🛠️ Features

Modular codebase in my_timeseries_pkg/:

generator.py – Synthetic & real‑data series generator (TSGenerator).

acfs.py – ACF vs. ADCF computations & plotting utilities.

rnn.py – RNN model build, training, activation‑extraction routines.

metrics.py – MSE, SMAPE, WAPE, Relative MSE implementations.

utils.py – I/O helpers (CSV export, results saving).

Orchestration via run_experiments.py: loops over multiple processes, runs simulations, generates plots & CSV summaries.

Configurable via command‑line flags: series length, window, epochs, hyperparameters.

🚀 Installation

Clone this repo:

git clone https://github.com/salezaraus/RNN_DCOR.git
cd rnn_dc_simulation

Set up a virtual environment (optional but recommended):

python3 -m venv venv
source venv/bin/activate  # on Windows: venv\\Scripts\\activate

Install dependencies:

pip install -r requirements.txt

🎯 Usage

Run the default suite of experiments (AR(1,5,10,20)):

python run_experiments.py --output_dir results

Key arguments:

--num_runs (number of Monte Carlo runs)

--window / --overlap / --horizon (sliding‑window settings)

--epochs / --batch_size / --hidden_units (RNN hyperparameters)

--mean / --std / --frequency / --drift / --omega (TS generator settings)

Results will be saved under results/:

results/figures/ – ACF/ADCF barplots & time‐series forecasts

results/summary.csv – aggregated MSE, SMAPE, WAPE, Relative MSE per process

🔍 Repository Structure

my_timeseries_project/
├── config.py             # (optional) global hyperparameter definitions
├── run_experiments.py    # CLI entry‐point for experiments
├── requirements.txt      # Python package dependencies
├── my_timeseries_pkg/    # core modules
│   ├── __init__.py
│   ├── generator.py
│   ├── acfs.py
│   ├── rnn.py
│   ├── metrics.py
│   └── utils.py
└── tests/                # (optional) unit tests
    ├── test_generator.py
    ├── test_acfs.py
    └── test_metrics.py

📖 Paper Summary

The paper introduces a distance correlation–based framework to:

Quantify how RNN activation layers capture lag dependencies in time series.

Reveal that RNNs, while learning short‐range lags well, lose memory across ~5–6 layers, harming forecasts on high‐lag processes.

Demonstrate RNN struggles on MA and heteroskedastic (GARCH) series.

Visualize model similarities via heatmaps—showing input window size drives performance more than activation or hidden units.

These insights help practitioners pre‑assess RNN suitability for new time series, guide hyperparameter choices, and interpret internal network dynamics without exhaustive tuning.

📜 Citation

If you use this code, please cite:

Salazar, C. & Banerjee, A. G. “A Distance Correlation–Based Approach to Characterize the Effectiveness of Recurrent Neural Networks for Time Series Forecasting”. Neurocomputing, 2025. 

📄 License

This project is licensed under the MIT License. See LICENSE for details.

Developed by Christopher Salazar, University of Washington.

