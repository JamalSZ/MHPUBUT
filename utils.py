import numpy as np
from scipy.optimize import fsolve
from sortedcontainers import SortedDict  # Efficient sorted dictionary
from collections import defaultdict
import pandas as pd
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
from config import *
from scipy.signal import find_peaks, detrend
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

def load_series(dataset_cls, target_column: str = None, size: int = None) -> pd.Series:
    ts = dataset_cls().load()
    df = ts.to_dataframe()
    length = len(df) if size is None else size
    if target_column is not None:
        return df[target_column].iloc[:length]
    return df.iloc[:length, 0]

def estimate_period_fft(series, min_prominence=0.1, return_float=False, apply_window=True):
    series = np.asarray(series)
    n = len(series)

    # Remove linear trend
    series_detrended = detrend(series)

    # Apply window to reduce spectral leakage
    if apply_window:
        series_detrended *= np.hamming(n)

    # FFT
    ft = np.abs(np.fft.rfft(series_detrended))
    freqs = np.fft.rfftfreq(n)

    # Find peaks
    peaks, _ = find_peaks(ft, height=min_prominence * np.max(ft))
    valid = peaks[freqs[peaks] > 1e-6]
    if len(valid) == 0:
        return None

    dominant_freq = freqs[valid[np.argmax(ft[valid])]]
    period = 1 / dominant_freq

    return period if return_float else int(round(period))

def estimate_period_stl(series, candidate_periods):
    best_score = -np.inf
    best_period = None
    for period in candidate_periods:
        try:
            res = STL(series, period=period).fit()
            seasonal_strength = max(0, 1 - np.var(res.resid)/np.var(res.seasonal + res.resid))
            if seasonal_strength > best_score:
                best_score = seasonal_strength
                best_period = period
        except:
            continue
    return best_period if best_score > 0.1 else None  # Threshold for significance

def estimate_period_acf(series, max_lag=100):
    plot_acf(series, lags=max_lag)
    plt.show()
    # Manually inspect plot for peaks at multiples of candidate period
    # Or automate peak detection:
    acf = np.correlate(series - np.mean(series), series - np.mean(series), mode='full')
    acf = acf[len(acf)//2:]
    peaks = np.argsort(acf)[-5:][::-1]  # Top 5 candidate lags
    return peaks[peaks > 0][0]  # First non-zero peak


def auto_detect_period(series, candidate_periods=None):
    """Automatically estimate seasonality period"""
    # Step 1: Get candidate periods
    if candidate_periods is None:
        max_period = min(100, len(series)//2)  # Practical upper limit
        candidate_periods = list(range(2, max_period+1))
    
    # Step 2: FFT-based estimation
    fft_period = estimate_period_fft(series)
    
    # Step 3: ACF-based estimation
    acf_period = estimate_period_acf(series, max_lag=max(candidate_periods))
    
    # Step 4: STL-based validation
    candidates = list(set([p for p in [fft_period, acf_period] if p is not None]))
    stl_period = estimate_period_stl(series, candidates)
    if not candidates:
        return None
    return fft_period, acf_period, stl_period

def plot_combined_results(dataset_name,epsilon, results, original_series):
    """Combine decomposition and predictability plots into one figure"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(4, 2, width_ratios=[3, 1], height_ratios=[1,1,1,1])
    
    # Decomposition Plots (Left Column)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
    ax3 = fig.add_subplot(gs[3, 0], sharex=ax0)
    
    # Original Series
    ax0.plot(original_series, color='#1f77b4')
    ax0.set_title(f'Original Time Series ({dataset_name})')
    ax0.grid(True, alpha=0.3)
    
    # Trend Component
    ax1.plot(results['components']['trend'], color='#2ca02c')
    ax1.set_title('Trend Component')
    ax1.grid(True, alpha=0.3)
    
    # Seasonal Component
    ax2.plot(results['components']['seasonal'], color='#d62728')
    ax2.set_title('Seasonal Component')
    ax2.grid(True, alpha=0.3)
    
    # Residual Component
    ax3.plot(results['components']['residuals'], color='#7f7f7f')
    ax3.set_title(f'Residuals (ε={results["epsilon"]:.3f}, N={results["N"]:.1f})')
    ax3.grid(True, alpha=0.3)
    
    # Analysis Plots (Right Column)
    ax_pie = fig.add_subplot(gs[0:2, 1])
    ax_bar = fig.add_subplot(gs[2:4, 1])
    
    # Variance Proportions Pie
    vp = results['variance_proportions']
    ax_pie.pie([vp['trend'], vp['seasonality'], vp['residual']],
               labels=['Trend', 'Seasonality', 'Residual'],
               colors=['#2ca02c', '#d62728', '#7f7f7f'],
               autopct='%1.1f%%')
    ax_pie.set_title('Variance Proportions')
    
    # Predictability Breakdown Bar
    breakdown = {
        'Trend': vp['trend'],
        'Seasonality': vp['seasonality'],
        'Residual\nPredictability': vp['residual'] * results['pi_max_residual']
    }
    ax_bar.bar(breakdown.keys(), breakdown.values(),
               color=['#2ca02c', '#d62728', '#1f77b4'])
    ax_bar.set_title('Predictability Contribution')
    ax_bar.set_ylim(0, 1)
    ax_bar.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'results/{dataset_name}/combined_plot_{results["epsilon"]}.png')
    plt.close()

# ==== 2. Estimate seasonal period ====
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
