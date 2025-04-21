"""
Auto‐ and distance‐correlation analysis and plotting utilities.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import dcor


def plot_mean_corrs(adcf_all: np.ndarray, acf_all: np.ndarray, ts_process: str, fig_path: str) -> None:
    """
    Plot mean ADCF vs ACF across runs and save the figure.

    :param adcf_all: array of shape (runs, lags) of distance correlations
    :param acf_all: array of shape (runs, lags) of autocorrelations
    :param ts_process: name of the time series process (for filename)
    :param fig_path: directory to save figures
    """
    os.makedirs(fig_path, exist_ok=True)

    bar_width = 0.3
    mean_adcf = np.mean(adcf_all, axis=0)
    mean_acf = np.mean(acf_all, axis=0)
    lags = np.arange(1, mean_adcf.size + 1)

    idx1 = np.arange(len(lags))
    idx2 = idx1 + bar_width

    plt.figure(figsize=(8, 5))
    plt.bar(idx1, mean_adcf, width=bar_width, label="Distance Corr")
    plt.bar(idx2, mean_acf, width=bar_width, label="ACF")
    plt.xticks(idx1 + bar_width/2, lags)
    plt.xlabel("Lag")
    plt.ylabel("Correlation")
    plt.title(f"Mean Correlations: {ts_process}")
    plt.legend()
    plt.tight_layout()
    out_file = os.path.join(fig_path, f"{ts_process}_mean_corrs.png")
    plt.savefig(out_file, dpi=300)
    plt.close()


def plot_ts(original: np.ndarray, time_idx: np.ndarray, pred: np.ndarray, ts_process: str, fig_path: str) -> None:
    """
    Plot a time series with its forecasted segment.

    :param original: full series (1D array)
    :param time_idx: indices of forecasted horizon
    :param pred: predicted values (1D array)
    :param ts_process: name for saving
    :param fig_path: save directory
    """
    os.makedirs(fig_path, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(len(original)), original, label="Original")
    plt.plot(time_idx, pred, label="Forecast")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(ts_process)
    plt.legend()
    plt.tight_layout()
    out_file = os.path.join(fig_path, f"{ts_process}_forecast.png")
    plt.savefig(out_file, dpi=300)
    plt.close()