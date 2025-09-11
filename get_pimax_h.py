import os
import csv
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Any
from scipy.optimize import fsolve
from statsmodels.tsa.seasonal import STL
from multiprocessing import Pool, cpu_count
from darts import TimeSeries
from sortedcontainers import SortedDict
from collections import defaultdict
from config import *
from statsmodels.tsa.stattools import acf
import random


# Constants
TEST_SIZE = 0.2
seed=2025
np.random.seed(seed)


class IFI:
    def __init__(self, T: np.ndarray, epsilon: float, h: int):
        self.T = np.asarray(T)  # ✅ Ensure T is a NumPy array and assign it to self.T
        self.epsilon = epsilon  
        self.n = len(T)
        self.h = h

    def bisect_left(self,arr, x, lo=0, hi=None):
        """
        Locate the insertion point for x in a sorted sequence arr.
        The parameters lo and hi may be used to specify a subset of the sequence
        to search.
        """
        if hi is None:
            hi = len(arr)

        while lo < hi:
            mid = (lo + hi) // 2
            if arr[mid] < x:
                lo = mid + 1
            else:
                hi = mid

        return lo

    def bisect_right(self, arr, x, lo=0, hi=None):
        """
        Return the index after the last occurrence of x in a sorted sequence arr.
        The parameters lo and hi may be used to specify a subset of the sequence
        to search.
        """
        if hi is None:
            hi = len(arr)

        while lo < hi:
            mid = (lo + hi) // 2
            if x < arr[mid]:
                hi = mid
            else:
                lo = mid + 1

        return lo

    def LZ2lookup_ifi(self,keys, D, s, j, e):
        """Find indices i < j where T[j] is within e-range of T[i]."""
        matches = set()
        
        left, right = self.bisect_left(keys, s - e), self.bisect_right(keys, s + e)
        
        for key in keys[left:right]:
            values = D.get(key, [])
            for k in values:
                if k < j:
                    matches.add(k)
                else:
                    break

        return matches

    def LZ2build_IIdx_ifi(self):
        """Build an inverted index for quick lookup."""
        index = {}
        for i, char in enumerate(self.T):
            index.setdefault(char, []).append(i)
        
        return SortedDict(index)  # Keeps keys sorted for fast range queries

    def LZ2_ifi(self):
        """Compute pairs (cur, m) where T[cur] is within tolerance e of T[m]."""
        Idx = self.LZ2build_IIdx_ifi()
        keys = list(Idx.keys())
        result = []
        ee = 2 * self.epsilon
        
        lzc = np.zeros(self.n-self.h, dtype=int)
        lzc_zeros = np.ones(self.n-self.h, dtype=int)  # Start assuming all rows will have non-zero values
        # Use two dictionaries to store the current row and the row below it
        current_row_dict = defaultdict(int)
        next_row_dict = defaultdict(int)

        for cur in range(self.n-1-self.h,-1,-1):
            curr_row = cur + self.h
            matches = self.LZ2lookup_ifi(keys, Idx, self.T[curr_row], cur, ee)
            
            if len(matches) > 0:
                next_row_dict = current_row_dict
                current_row_dict = defaultdict(int)
                for col in matches:
                    if curr_row+1 <self.n and col+1 <self.n:
                        diagonal_value = next_row_dict.get(col + 1, 0)
                        current_row_dict[col] = min(diagonal_value + 1, cur - col)

                        if current_row_dict[col] + curr_row <= self.n - 1 - self.h:
                            lzc[cur] = max(current_row_dict[col] + 1, lzc[cur])
                        else:
                            #print(row,col, diagonal_value,current_row_dict[col])
                            lzc_zeros[cur] = 0  # Mark for zero in the final pass
                    else:
                        lzc_zeros[cur] = 0  # Mark for zero in the final pass
                        current_row_dict[col] = 1
            else:
                lzc_zeros[cur] = 0  # Mark for zero in the final pass
                current_row_dict = defaultdict(int) 
                next_row_dict = defaultdict(int)
        for i in range(self.n-1-self.h,-1,-1):
            if lzc_zeros[i] != 0:
                break
            lzc[i]=0

        return lzc.sum()
    
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

def entropy_gap_equation(x, H: float, N: float) -> float:
    """Entropy balance equation used to solve for pi_max"""
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x) + (1 - x) * np.log2(N - 2) - H


def get_pi_max(H: float, N: float) -> float:
    """Compute pi_max using numerical solver"""
    pi_max = fsolve(lambda x: entropy_gap_equation(x, H, N), 0.999999)[0]
    return pi_max


def compute_pi_max_for_series(series: np.ndarray, epsilon: float, period: int, horizon: int) -> List[float]:
    """Compute pi_max values for different horizons"""
    stl = STL(series, period=period).fit()
    trend, seasonal, residual = stl.trend, stl.seasonal, stl.resid

    # Variance ratio of residuals
    var_total = np.var(series)
    var_r = np.var(residual)
    var_s = np.var(seasonal)
    var_t = np.var(trend)
    v_r = var_r / var_total

    lzc_values = [IFI(series, epsilon, h).LZ2_ifi() for h in range(horizon)]
    pi_max_values = []
    pi_max_values_uncorrected = []

    N = (max(series) - min(series) + 2 * epsilon) / epsilon

    for lzc in lzc_values:
        n = len(series)
        H_R = n * np.log2(n) / lzc
        pi_h = get_pi_max(H_R, N) 
        pi_max_values.append(pi_h* (1 - v_r))
        pi_max_values_uncorrected.append(pi_h)

    return pi_max_values, pi_max_values_uncorrected


def process_task(args: Tuple[str, np.ndarray, int, float, int]) -> Dict[str, Any]:
    """Worker function to compute pi_max for a dataset/epsilon combo"""
    dataset_name, series, period, epsilon, horizon = args
    start = time.time()
    pi_max_H, pi_max_H_uncorrected = compute_pi_max_for_series(series, epsilon, period, horizon)
    return {
        'dataset': dataset_name,
        'epsilon': epsilon,
        'pi_max_H': pi_max_H,
        'pi_max_H_uncorrected': pi_max_H_uncorrected,
        'time_sec': time.time() - start
    }


def save_results_to_csv(dataset_results: List[Dict[str, Any]]) -> None:
    """Save pi_max results to per-dataset CSV files"""
    grouped_results = defaultdict(list)
    for res in dataset_results:
        grouped_results[res['dataset']].append(res)

    for dataset_name, results in grouped_results.items():
        output_dir = Path(f"results/ResultsPimax")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{dataset_name}_pimax_h.csv"
        output_file1 = output_dir / f"{dataset_name}_pimax_h_uncorrected.csv"

        with output_file.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['epsilon', 'pi_max_H'])
            writer.writeheader()
            for res in sorted(results, key=lambda x: x['epsilon']):
                writer.writerow({
                    'epsilon': res['epsilon'],
                    'pi_max_H': res['pi_max_H']
                })
        with output_file1.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['epsilon', 'pi_max_H_uncorrected'])
            writer.writeheader()
            for res in sorted(results, key=lambda x: x['epsilon']):
                writer.writerow({
                    'epsilon': res['epsilon'],
                    'pi_max_H_uncorrected': res['pi_max_H_uncorrected']
                })


def load_dataset_series(dataset_name: str, config: Dict) -> np.ndarray:
    """Load and preprocess the dataset as a univariate float32 series"""
    dataset = config["loader"]().load()

    if isinstance(dataset, TimeSeries):
        if dataset_name == "TaxiNewYork":
            series = dataset
        else:
            series = dataset.univariate_component(config["target"])
    else:
        series = TimeSeries.from_series(dataset[config["target"]])

    if series.n_components != 1:
        print(f"[{dataset_name}] Warning: using first component of multivariate series.")
        series = series.univariate_component(0)

    values = series.values().astype(np.float32)
    values = values[:, 0] if values.ndim == 2 else values
    values = values[~np.isnan(values)]

    split_idx = int(len(values) * (1 - TEST_SIZE))
    return values[:split_idx]


def main():
    dataset_info = {}
    for name, config in DATASETS_CONFIG.items():
        series = load_dataset_series(name, config)
        period = estimate_seasonal_period(series)
        print(f"[{name}] Estimated Seasonal Period: {period}")
        dataset_info[name] = (series, period, config['epsilons'], config['horizon'])

    all_tasks = [
        (name, series, period, eps, horizon)
        for name, (series, period, epsilons, horizon) in dataset_info.items()
        for eps in epsilons
    ]

    num_workers = min(cpu_count() * 2, 32)
    print(f"Processing {len(all_tasks)} tasks using {num_workers} workers...")

    with Pool(processes=num_workers) as pool:
        results = pool.imap_unordered(process_task, all_tasks, chunksize=2)
        final_results = list(results)

    save_results_to_csv(final_results)
    print("All tasks completed and results saved.")


if __name__ == "__main__":
    main()


