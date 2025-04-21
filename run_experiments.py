"""
run_experiments.py

Orchestrates forecasting experiments across multiple time series processes.
"""

import os
import argparse
import numpy as np
from statsmodels.tsa.stattools import acf
import dcor

from my_timeseries_pkg.generator import TSGenerator
from my_timeseries_pkg.rnn import build_rnn, train_and_extract, plot_loss
from my_timeseries_pkg.acfs import plot_mean_corrs, plot_ts
from my_timeseries_pkg.metrics import smape, wape, relative_mse
from my_timeseries_pkg.utils import save_results_to_csv


def parse_args():
    parser = argparse.ArgumentParser(description="Run time series forecasting experiments")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory for outputs")
    parser.add_argument("--num_runs", type=int, default=35, help="Number of runs per process")
    parser.add_argument("--length", type=int, default=4000, help="Length of time series")
    parser.add_argument("--window", type=int, default=20, help="Window size for samples")
    parser.add_argument("--overlap", type=int, default=19, help="Overlap between windows")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--hidden_units", type=int, default=64, help="RNN hidden units")
    parser.add_argument("--mean", type=float, default=0.0, help="TS generator mean")
    parser.add_argument("--std", type=float, default=1.0, help="TS generator std")
    parser.add_argument("--frequency", type=float, default=1.0, help="TS generator noise frequency")
    parser.add_argument("--drift", type=float, default=0.0, help="TS generator drift")
    parser.add_argument("--omega", type=float, default=0.5, help="GARCH omega parameter")
    parser.add_argument("--activation", type=str, default="relu", help="RNN activation")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split fraction")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    TS_PROCESSES = ["AR_1", "AR_5", "AR_10", "AR_20"]
    C_LAGS = [1, 5, 10, 20]
    B_LAGS = [1, 5, 10, 20]

    summary = []

    for ts_name, ar_lag, ma_lag in zip(TS_PROCESSES, C_LAGS, B_LAGS):
        adcf_all = np.zeros((args.num_runs, args.window))
        acf_all = np.zeros((args.num_runs, args.window))
        metrics_runs = []

        for run in range(args.num_runs):
            # prepare AR coefficients
            coef_lag_list = [(1.0 if i+1 == ar_lag else 0.0, i+1) for i in range(args.window)]
            beta_list = [(0.0, i+1) for i in range(args.window)]  # zero MA for pure AR

            # generate time series
            gen = TSGenerator(dataset="ARMA", length=args.length, num_ts=1,
                              mu=args.mean, sigma=args.std,
                              coef_lag_list=coef_lag_list,
                              beta_list=beta_list,
                              omega=args.omega,
                              noise_frequency=args.frequency,
                              drift=args.drift)

            # create sliding windows and targets
            X, y = gen.create_windows(args.window, args.horizon, args.overlap)
            # split into train/val
            val_size = int(len(X) * args.val_split)
            X_train, y_train = X[:-val_size], y[:-val_size]
            X_val, y_val = X[-val_size:], y[-val_size:]
            # reshape for RNN input
            X_train = X_train[..., np.newaxis]
            X_val = X_val[..., np.newaxis]

            # build, train, extract activations
            model = build_rnn(args.window, args.hidden_units, args.horizon, args.activation)
            history, activations = train_and_extract(model, X_train, y_train,
                                                     X_val, y_val,
                                                     args.epochs, args.batch_size)

            # compute ACF and ADCF for this run
            raw_series = gen.ts[0].numpy()
            acf_vals = acf(raw_series, nlags=args.window)[1:]
            adcf_vals = [dcor.distance_correlation(y_val, activations[-1][:, lag, 0])
                         for lag in range(args.window)]

            adcf_all[run, :] = adcf_vals
            acf_all[run, :] = acf_vals

            # forecast and metrics
            preds = model.predict(X_val).flatten()
            m = {
                "process": ts_name,
                "mse": float(np.mean((preds - y_val)**2)),
                "smape": smape(y_val, preds),
                "wape": wape(y_val, preds),
                "rel_mse": relative_mse(y_val, preds)
            }
            metrics_runs.append(m)

        # save correlation plots
        plot_mean_corrs(adcf_all, acf_all, ts_name, fig_dir)
        # time series plot for last run
        plot_ts(gen.ts[0].numpy(),
                np.arange(args.length - len(y_val), args.length),
                preds, ts_name, fig_dir)

        # aggregate metrics across runs
        agg = {"process": ts_name}
        for key in ["mse", "smape", "wape", "rel_mse"]:
            agg[key] = float(np.mean([r[key] for r in metrics_runs]))
        summary.append(agg)

    # write summary CSV
    save_results_to_csv(summary, os.path.join(args.output_dir, "summary.csv"))


if __name__ == "__main__":
    main()
