import numpy as np
import pandas as pd
from scipy.signal import detrend

def aggregate_hit_rate(df: pd.DataFrame, n_bins: int = 90) -> tuple:
    """
    Returns (soa_centers_ms, hit_rate) as np.ndarrays.
    Bins trials by SOA into n_bins equal-width bins.
    Raises ValueError if mean trials per bin < 5.
    """
    if df.empty:
        raise ValueError("DataFrame is empty")
        
    if len(df) / n_bins < 5:
        raise ValueError("Mean trials per bin < 5")
        
    min_soa = df['soa_ms'].min()
    max_soa = df['soa_ms'].max()
    
    if min_soa == max_soa:
        # Avoid zero division or identical edges
        bins = np.linspace(min_soa - 1, max_soa + 1, n_bins + 1)
    else:
        bins = np.linspace(min_soa, max_soa, n_bins + 1)
        
    soa_centers_ms = (bins[:-1] + bins[1:]) / 2.0
    
    df_copy = df.copy()
    df_copy['bin'] = pd.cut(df_copy['soa_ms'], bins=bins, include_lowest=True, labels=False)
    
    hit_rate_series = df_copy.groupby('bin')['hit'].mean()
    hit_rate_series = hit_rate_series.reindex(range(n_bins))
    
    # Interpolate any Empty bins
    hit_rate_series = hit_rate_series.interpolate(method='linear', limit_direction='both')
    hit_rate = hit_rate_series.fillna(0).values # Fallback if interpolation fails
    
    return soa_centers_ms, hit_rate

def detrend_signal(signal: np.ndarray) -> np.ndarray:
    """Linear detrend via scipy.signal.detrend. Mean-centers afterward."""
    d = detrend(signal, type='linear')
    d = d - np.mean(d)
    return d

def apply_hanning_window(signal: np.ndarray) -> np.ndarray:
    """Multiply by Hanning window to reduce spectral leakage. Same length."""
    win = np.hanning(len(signal))
    return signal * win

def zero_pad_to_power_of_two(signal: np.ndarray, min_length: int = 4096) -> np.ndarray:
    """Zero-pad to next power of 2 >= max(len(signal), min_length)."""
    target_len = max(len(signal), min_length)
    if target_len <= 0:
        power = 0
    else:
        power = int(np.ceil(np.log2(target_len)))
    pad_len = (2 ** power) - len(signal)
    return np.pad(signal, (0, pad_len), 'constant')
