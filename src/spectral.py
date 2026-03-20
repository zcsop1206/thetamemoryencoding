import logging
import numpy as np
import pandas as pd
from scipy import signal
from fooof import FOOOF

def compute_power_spectrum(signal_arr: np.ndarray, dt_ms: float) -> tuple:
    """
    Compute one-sided PSD via scipy.signal.periodogram.
    dt_ms: sampling interval in milliseconds. fs = 1000 / dt_ms Hz.
    Returns (freqs_hz, power), both trimmed to 0-30 Hz.
    """
    fs = 1000.0 / dt_ms
    freqs, power = signal.periodogram(signal_arr, fs)
    mask = (freqs >= 0) & (freqs <= 30)
    return freqs[mask], power[mask]

def fit_fooof(freqs: np.ndarray, power: np.ndarray, freq_range: list = [1, 30], peak_width_limits: list = [0.5, 6.0]) -> dict:
    """
    Fit specparam (FOOOF) model to separate periodic from aperiodic power.
    """
    fm = FOOOF(peak_width_limits=peak_width_limits, max_n_peaks=4, min_peak_height=0.05, verbose=False)
    fm.fit(freqs, power, freq_range)
    
    res = {
        'model': fm,
        'aperiodic_fit': fm._ap_fit,
        'periodic_peaks': [],
        'theta_peak_cf': None,
        'theta_peak_pw': None,
        'aperiodic_exp': fm.aperiodic_params_[1] if len(fm.aperiodic_params_) > 1 else 0.0,
        'r_squared': fm.r_squared_
    }
    
    if fm.r_squared_ < 0.85:
        logging.warning("FOOOF fit quality poor (R^2 < 0.85). Setting theta_peak_cf = None")
    else:
        peaks = fm.get_params('peak_params')
        if peaks.ndim == 1 and len(peaks) == 3:
            peaks = [peaks]
            
        for peak in peaks:
            if len(peak) == 3:
                cf, pw, bw = peak
                res['periodic_peaks'].append({'cf': cf, 'pw': pw, 'bw': bw})
                if 3.0 <= cf <= 10.0:
                    if res['theta_peak_pw'] is None or pw > res['theta_peak_pw']:
                        res['theta_peak_cf'] = cf
                        res['theta_peak_pw'] = pw
                        
    return res

def get_theta_peak_power(df: pd.DataFrame, theta_band: tuple = (3.0, 10.0)) -> float:
    """
    Single-call convenience for permutation test inner loop.
    Runs: aggregate -> detrend -> window -> pad -> FFT -> FOOOF.
    Returns theta peak power, or 0.0 if no peak found or FOOOF fit poor.
    """
    from .preprocessing import aggregate_hit_rate, detrend_signal, apply_hanning_window, zero_pad_to_power_of_two
    
    try:
        soas, hit_rate = aggregate_hit_rate(df)
    except ValueError:
        return 0.0
        
    if len(soas) < 2:
        return 0.0
        
    dt_ms = soas[1] - soas[0]
    detrended = detrend_signal(hit_rate)
    windowed = apply_hanning_window(detrended)
    padded = zero_pad_to_power_of_two(windowed)
    
    freqs, power = compute_power_spectrum(padded, dt_ms)
    fooof_res = fit_fooof(freqs, power)
    
    pw = fooof_res['theta_peak_pw']
    cf = fooof_res['theta_peak_cf']
    
    if pw is not None and cf is not None and theta_band[0] <= cf <= theta_band[1]:
        return float(pw)
    return 0.0
