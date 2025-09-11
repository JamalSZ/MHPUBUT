import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries

from typing import Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from sklearn.neighbors import KDTree, NearestNeighbors
from sklearn.metrics import mutual_info_score
from statsmodels.tsa.seasonal import STL

from config import *
from statsmodels.tsa.stattools import acf

def load_series(dataset_name: str, dataset_config) -> pd.Series:
    """Load time series data"""
    dataset = dataset_config["loader"]().load()
    if isinstance(dataset, TimeSeries):
        target_series = dataset
    else:
        target_series = TimeSeries.from_series(dataset[dataset_config["target"]])
    
    if target_series.n_components != 1:
        target_series = target_series.univariate_component(0)
    
    target_series = target_series.values().astype(np.float32)
    train_size = int(len(target_series) * 0.8)
    train = target_series[:train_size, 0]
    return train[~np.isnan(train)]

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# === STL Decomposition ===
def extract_residual_stl(series: pd.Series, period: int, robust: bool = True, plot: bool = False
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    stl = STL(series, period=period, robust=robust)
    result = stl.fit()
    if plot:
        result.plot()
        plt.suptitle("STL Decomposition")
        plt.show()
    return result.trend, result.seasonal, result.resid



def estimate_seasonal_period_fft(series: np.ndarray, max_period: int = 1000):
    series = np.asarray(series)
    n = len(series)
    freqs = np.fft.rfftfreq(n, d=1.0)
    spectrum = np.abs(np.fft.rfft(series - np.mean(series)))
    freqs = freqs[1:]
    spectrum = spectrum[1:]
    periods = 1 / freqs
    valid = periods < max_period
    if np.any(valid):
        return int(round(periods[valid][np.argmax(spectrum[valid])]))
    return None

def estimate_seasonal_period_acf(series: np.ndarray, max_lag: int = 500):
    acf_vals = acf(series, nlags=max_lag, fft=True)
    for lag in range(1, len(acf_vals) - 1):
        if acf_vals[lag] > 0.5 and acf_vals[lag] > acf_vals[lag - 1] and acf_vals[lag] > acf_vals[lag + 1]:
            return lag
    return None

def estimate_seasonal_period(series: np.ndarray, max_period: int = 1000, method="hybrid"):
    fft_period = estimate_seasonal_period_fft(series, max_period=max_period)
    acf_period = estimate_seasonal_period_acf(series, max_lag=max_period)
    if method == "hybrid":
        if fft_period and acf_period:
            return min(fft_period, acf_period)
        return fft_period or acf_period
    elif method == "fft":
        return fft_period
    elif method == "acf":
        return acf_period
    raise ValueError(f"Unknown method: {method}")

# === False Nearest Neighbors ===
def false_nearest_neighbors(
    ts: np.ndarray,
    d_max: int = 20,
    tau: int = 1,
    threshold: float = 0.01,
    verbose: bool = False
) -> int:
    ts = np.asarray(ts)
    N = len(ts)
    fnn_rel_change_threshold = 10
    fnn_distance_threshold = 2 * np.std(ts)
    perc_fnn = []

    for d in range(1, d_max + 1):
        n_vectors = N - (d + 1) * tau + 1
        if n_vectors < 1:
            break

        x_d = np.array([ts[i: i + n_vectors * tau: tau] for i in range(d)]).T
        x_d1 = np.array([ts[i: i + n_vectors * tau: tau] for i in range(d + 1)]).T

        tree = KDTree(x_d)
        _, indices = tree.query(x_d, k=2)
        neighbors = indices[:, 1]

        dist_d = np.linalg.norm(x_d - x_d[neighbors], axis=1)
        delta = np.abs(x_d1[:, -1] - x_d1[neighbors, -1])

        with np.errstate(divide='ignore', invalid='ignore'):
            rel_change = delta / dist_d
            fnn_mask = (rel_change > fnn_rel_change_threshold) | (delta > fnn_distance_threshold)

        perc = np.mean(fnn_mask) * 100
        perc_fnn.append(perc)

        if verbose:
            logging.info(f"d={d}: {perc:.2f}% false neighbors")

        if perc < threshold * 100:
            return d

    return d_max

# === Time-delay Embedding ===
def time_delay_embedding(series: np.ndarray, d: int, tau: int = 1) -> np.ndarray:
    N = len(series)
    return np.array([series[i: N - (d - 1) * tau + i: tau] for i in range(d)]).T

# === AMI & Theiler Estimation ===
def compute_average_mutual_information(ts: np.ndarray, max_lag: int = 100) -> np.ndarray:
    ts = (ts - np.mean(ts)) / np.std(ts)
    bins = int(np.sqrt(len(ts)))
    ami_values = []

    for lag in range(1, max_lag + 1):
        x, y = ts[:-lag], ts[lag:]
        c_xy = np.histogram2d(x, y, bins=bins)[0]
        mi = mutual_info_score(None, None, contingency=c_xy)
        ami_values.append(mi)

    return np.array(ami_values)

def find_first_minimum(ami: np.ndarray) -> int:
    for i in range(1, len(ami) - 1):
        if ami[i - 1] > ami[i] < ami[i + 1]:
            return i + 1
    return 1

def estimate_theiler_window(ts: np.ndarray, max_lag: int = 100, max_d: int = 10) -> int:
    ami = compute_average_mutual_information(ts, max_lag=max_lag)
    tau_opt = find_first_minimum(ami)
    d_opt = false_nearest_neighbors(ts, d_max=max_d)
    return d_opt + tau_opt

# === Lyapunov Spectrum ===
def estimate_lyapunov_spectrum(
    embedded: np.ndarray,
    theiler_window: int,
    min_neighbors: Optional[int] = None,
    max_iter: Optional[int] = None,
    use_radius: bool = False,
    epsilon: Optional[float] = None
) -> np.ndarray:
    M, d = embedded.shape
    if min_neighbors is None:
        min_neighbors = max(2 * d, 20)
    if max_iter is None:
        max_iter = M - 2 - theiler_window
    if use_radius and epsilon is None:
        raise ValueError("epsilon must be specified when use_radius=True")

    Q = np.eye(d)
    lyap_sum = np.zeros(d)
    valid_steps = 0

    nbrs = NearestNeighbors(n_neighbors=M - 1, algorithm="kd_tree")
    nbrs.fit(embedded[:-1])

    for t in range(theiler_window, min(theiler_window + max_iter, M - 1)):
        x_t, x_tp1 = embedded[t], embedded[t + 1]
        distances, indices = nbrs.kneighbors([x_t], n_neighbors=M - 1)
        indices, distances = indices[0], distances[0]

        mask = np.abs(indices - t) > theiler_window
        valid_indices = indices[mask]
        valid_distances = distances[mask]

        if use_radius:
            radius = epsilon * np.sqrt(d)
            valid_indices = valid_indices[valid_distances < radius]

        if len(valid_indices) < min_neighbors:
            continue

        selected = valid_indices[:min_neighbors]
        X, Y = embedded[selected], embedded[selected + 1]
        Xc, Yc = X - x_t, Y - x_tp1

        try:
            A = np.linalg.lstsq(Xc, Yc, rcond=None)[0].T
            V = A @ Q
            Q, R = np.linalg.qr(V)
            lyap_sum += np.log(np.abs(np.diag(R)))
            valid_steps += 1
        except np.linalg.LinAlgError:
            continue

    if valid_steps < 10:
        return np.full(d, np.nan)
    return lyap_sum / valid_steps

def save_lyapunov_exponents(dataset_key: str, lyap_exponents: np.ndarray):
    result_dir = os.path.join("results", "ResultsLyap")
    os.makedirs(result_dir, exist_ok=True)

    df = pd.DataFrame({
        "lyap_index": np.arange(1, len(lyap_exponents)+1),
        "exponent": lyap_exponents
    })
    csv_path = os.path.join(result_dir, f"{dataset_key}_lyapunov_exponents.csv")
    df.to_csv(csv_path, index=False)
    logging.info(f"Saved Lyapunov exponents for {dataset_key} to {csv_path}")

# === Pipeline ===
def run_pipeline(dataset_name: str, plot_decomposition: bool = False):
    config = DATASETS_CONFIG[dataset_name]
    series = load_series(dataset_name, config)

    seasonal_period = estimate_seasonal_period(series, method="hybrid")
    logging.info(f"[{dataset_name}] Seasonal period: {seasonal_period}")

    trend, seasonal, residual = extract_residual_stl(series, period=seasonal_period, plot=plot_decomposition)

    optimal_d = false_nearest_neighbors(series, d_max=500, tau=1, verbose=True) * 2
    ami = compute_average_mutual_information(series, max_lag=500)
    tau_opt = find_first_minimum(ami)
    theiler_window = optimal_d + tau_opt

    logging.info(f"[{dataset_name}] Embedding dimension: {optimal_d}, Tau: {tau_opt}, Theiler window: {theiler_window}")

    embedded = time_delay_embedding(series, d=optimal_d, tau=1)

    lyap_exponents = estimate_lyapunov_spectrum(
        embedded=embedded,
        theiler_window=theiler_window
    )

    save_lyapunov_exponents(dataset_name, lyap_exponents)

    positive_lyap = [l for l in lyap_exponents if l > 0]
    logging.info(f"[{dataset_name}] Positive Lyapunov exponents: {positive_lyap}")

# === Runner ===
def run_all_pipelines_parallel():
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(run_pipeline, name): name for name in DATASETS_CONFIG.keys()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
                logging.info(f"[✓] Completed: {name}")
            except Exception as e:
                logging.error(f"[✗] Failed: {name} with error: {e}")

if __name__ == "__main__":
    run_all_pipelines_parallel()
    logging.info("All datasets processed.")
