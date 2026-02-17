import numpy as np
import pandas as pd

from statsmodels.tsa.stattools import adfuller
from joblib import Parallel, delayed

def get_weights_ffd(d, thres, size):
    w = [1.0]
    for k in range(1, size):
        w_ = -w[-1] * (d - k + 1) / k
        if abs(w_) < thres:
            break
        w.append(w_)
    return np.array(w[::-1]).reshape(-1, 1)

def frac_diff_ffd(series, d, thres = 1e-4):
    series = series.dropna()
    w = get_weights_ffd(d, thres, len(series))
    width = len(w) - 1

    series_f = series.ffill().dropna()
    res = []
    for i in range(width, series_f.shape[0]):
        res.append(np.dot(w.T, series_f.iloc[i-width:i+1])[0])

    return pd.Series(res, index=series_f.index[width:])

def find_optimal_d_and_diff(series, name):
    for d in np.linspace(0, 1, 21):
        diff_series = frac_diff_ffd(series, d)
        if diff_series.empty: continue

        p_val = adfuller(diff_series, maxlag=1, regression='c', autolag=None)[1]
        if p_val < 0.05:
            return name, d, diff_series

    return name, 1.0, frac_diff_ffd(series, 1.0)

def parallel_frac_diff(df, n_jobs=-1) -> tuple:
    print(f"Starting parallel FracDiff on {len(df.columns)} assets...")

    results = Parallel(n_jobs=n_jobs)(
        delayed(find_optimal_d_and_diff)(df[col], col) for col in df.columns
    )
    d_stats = {}
    diff_dfs = []

    for name, d, series in results:
        d_stats[name] = round(float(d),2)
        diff_dfs.append(series)

    final_df = pd.concat(diff_dfs, axis=1)
    final_df.columns = df.columns # rename columns

    return final_df, d_stats

def get_representative_d(df, quantile=0.9, n_jobs=-1):
    """
    Step 1: Find the minimum required 'd' for each column to achieve stationarity.
    Step 2: Select a representative 'd' based on the given quantile.
    """
    def find_min_d(series):
        # Scan 'd' from 0 to 1 with 0.05 step
        for d in np.linspace(0, 1, 21):
            diff_series = frac_diff_ffd(series, d)
            if diff_series.empty: continue

            # Use maxlag=1 for consistency in p-value testing
            p_val = adfuller(diff_series, maxlag=1, regression='c', autolag=None)[1]
            if p_val < 0.05:
                return d
        return 1.0

    print(f"🔍 Searching for minimum stationarity 'd' per asset...")
    individual_ds = Parallel(n_jobs=n_jobs)(
        delayed(find_min_d)(df[col]) for col in df.columns
    )

    # Select representative d (e.g., 90th percentile to ensure most assets are stationary)
    rep_d = np.percentile(individual_ds, quantile * 100)
    print(f"🎯 Representative d selected: {rep_d:.2f} (Quantile: {quantile})")

    return rep_d, individual_ds

def apply_uniform_frac_diff(df, d_value, n_jobs=-1):
    """
    Apply a single 'd' value to all columns in the DataFrame.
    """
    print(f"🚀 Applying uniform FracDiff (d={d_value:.2f}) to all columns...")

    results = Parallel(n_jobs=n_jobs)(
        delayed(frac_diff_ffd)(df[col], d_value) for col in df.columns
    )

    final_df = pd.concat(results, axis=1)
    final_df.columns = df.columns
    return final_df

