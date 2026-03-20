import logging
from pathlib import Path
import warnings
import json

from src.data_io import load_bids_ieeg
from src.spectral import fit_fooof
from src.stats import calculate_sme_plv
from scipy import signal

def main():
    warnings.filterwarnings('ignore')
    
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    log = logging.getLogger('ThetaMemory')
    
    config = {
        'bids_root': 'data/raw/bids/ds004100',
        'fooof_freq_range': [3, 10], # strictly theta constraints for FOOOF
        'output_dir': 'results/'
    }
    
    Path(config['output_dir']).mkdir(parents=True, exist_ok=True)
    
    log.info('Phase 1 START — Data acquisition (ds004100 FR1)')
    subject_dirs = [p for p in Path(config['bids_root']).glob('sub-*') if p.is_dir()]
    subject_ids = [p.name for p in subject_dirs]
    
    results_engram = {
        'metadata': {'dataset': 'ds004100', 'task': 'FR1'},
        'subjects': {}
    }
    
    processed_count = 0
    
    for i, sub in enumerate(subject_ids):
        try:
            log.info(f"Processing {sub}...")
            data = load_bids_ieeg(config['bids_root'], sub)
            events = data['events']
            raw = data['raw']
            fs = data['sfreq']
            
            log.info(f"Phase 2 — FOOOF Spectral Parameterization for {sub}")
            raw_data = raw.get_data()
            if raw_data.shape[1] > 2000:
                raw_data = raw_data[:, :2000] # Limit size for quick test
                
            freqs, psd = signal.periodogram(raw_data[0, :], fs=fs)
            mask = (freqs >= 1) & (freqs <= 30)
            
            # Use broader range for FOOOF parameterization to isolate 1/f noise accurately
            fooof_res = fit_fooof(freqs[mask], psd[mask], freq_range=[1, 30])
            
            log.info(f"Phase 3 — SME & PLV for {sub}")
            plv_res = calculate_sme_plv(raw_data, fs, events)
            
            # FOOOF R-squared extraction and rule check
            r2 = fooof_res.get('r_squared', 0.0)
            theta_cf = fooof_res.get('theta_peak_cf', None)
            
            if r2 > 0.90:
                results_engram['subjects'][sub] = {
                    'fooof_r2': r2,
                    'theta_center_freq': theta_cf,
                    'sme_effect_size': plv_res['sme_effect_size'],
                    'mean_plv_remembered': plv_res['mean_plv_rem'],
                    'mean_plv_forgotten': plv_res['mean_plv_forg']
                }
                processed_count += 1
            else:
                log.warning(f"{sub} failed spectral integrity check (R2={r2:.2f})")
                
        except Exception as e:
            log.debug(f"Skipped {sub}: {e}")
            
    if processed_count == 0:
        log.warning("No subjects passed the pipeline (either missing FR1 data or failed R2). Writing empty engram.")
        
    engram_path = Path(config['output_dir']) / 'ds004100_ram_engram.json'
    with open(engram_path, 'w') as f:
        json.dump(results_engram, f, indent=4)
        
    log.info(f"COMPLETE — Encoded {processed_count} subjects to {engram_path}")

if __name__ == '__main__':
    main()
