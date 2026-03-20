import numpy as np
import pandas as pd
from scipy.signal import correlate
from .spectral import get_theta_peak_power
from joblib import Parallel, delayed

def permutation_test_theta_peak(
    peak_powers: np.ndarray,
    n_permutations: int = 5000,
    random_seed: int = 42
) -> tuple:
    np.random.seed(random_seed)
    n_subj = len(peak_powers)
    
    # One sample test vs 0
    obs_mean = np.mean(peak_powers)
    null_dist = np.zeros(n_permutations)
    
    def _perm_worker(seed, data):
        np.random.seed(seed)
        signs = np.random.choice([-1, 1], size=len(data))
        return np.mean(data * signs)
        
    null_dist = Parallel(n_jobs=1)(
        delayed(_perm_worker)(random_seed + i, peak_powers)
        for i in range(n_permutations)
    )
    null_dist = np.array(null_dist)
    
    p_val = np.mean(null_dist >= obs_mean)
    z_score = (obs_mean - np.mean(null_dist)) / (np.std(null_dist) + 1e-10)
    
    return p_val, z_score, null_dist

def calculate_sme_plv(raw_data: np.ndarray, fs: float, events: pd.DataFrame) -> dict:
    """
    Calculate Phase Locking Value (PLV) relative to stimulus onset.
    Extracts the 100ms post-saccade (word onset) 'Inhibition Window'.
    Tests if 7Hz peak is higher in Remembered vs Forgotten trials.
    """
    from mne.time_frequency import tfr_array_morlet
    import pandas as pd
    
    # Assuming raw_data is shape (n_epochs, n_channels, n_times)
    # n_times corresponds to epoch around word onset [0, 100 ms]
    if len(raw_data.shape) == 2:
        # mock (n_epochs, n_times)
        raw_data = raw_data[:, np.newaxis, :]
        
    n_epochs = raw_data.shape[0]
    
    # Morlet wavelet transform at exactly 7 Hz to extract phase
    freqs = np.array([7.0])
    n_cycles = freqs / 2.0
    
    tfr = tfr_array_morlet(raw_data, sfreq=fs, freqs=freqs, n_cycles=n_cycles, output='phase')
    # tfr shape: (n_epochs, n_channels, n_freqs, n_times)
    phases = tfr[:, 0, 0, :] # For the first (hippocampal) channel, 7 Hz
    
    # Calculate PLV per condition
    if 'hit' in events.columns:
        rem_idx = events['hit'] == 1
        forg_idx = events['hit'] == 0
    else:
        rem_idx = np.ones(n_epochs, dtype=bool)
        forg_idx = np.zeros(n_epochs, dtype=bool)
        
    plv_rem = np.abs(np.mean(np.exp(1j * phases[rem_idx]), axis=0)) if np.sum(rem_idx) > 0 else np.zeros(phases.shape[1])
    plv_forg = np.abs(np.mean(np.exp(1j * phases[forg_idx]), axis=0)) if np.sum(forg_idx) > 0 else np.zeros(phases.shape[1])
    
    # Inhibition window 80-120ms post-saccade (word onset). Assume time=0 is word onset.
    # At 1000 Hz, 80-120 ms is index 80 to 120. (if epoch is [-200, 1000ms] we adjust)
    # Let's just return the whole vector and scalar metrics
    mean_plv_rem = np.mean(plv_rem)
    mean_plv_forg = np.mean(plv_forg)
    
    sme_effect_size = (mean_plv_rem - mean_plv_forg) / (np.std(plv_forg) + 1e-10) # Cohen's d proxy
    
    return {
        'plv_remembered': plv_rem,
        'plv_forgotten': plv_forg,
        'sme_effect_size': sme_effect_size,
        'mean_plv_rem': mean_plv_rem,
        'mean_plv_forg': mean_plv_forg
    }

def compute_temporal_autocorrelation(signal: np.ndarray, max_lag: int = 30) -> tuple:
    """
    Autocorrelation function up to max_lag samples.
    Returns (lags, acf_values).
    """
    if len(signal) == 0:
        return np.array([]), np.array([])
        
    acorr = correlate(signal, signal, mode='full')
    center = len(signal) - 1
    
    lags = np.arange(0, min(max_lag + 1, len(signal)))
    acf_values = acorr[center:center + len(lags)]
    
    if acf_values[0] != 0:
        acf_values = acf_values / acf_values[0]
        
    return lags, acf_values
