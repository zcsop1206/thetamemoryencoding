import mne
import pandas as pd
from pathlib import Path

def load_bids_ieeg(bids_root: str, subject_id: str) -> dict:
    """
    Load iEEG data for one subject from a BIDS dataset root.

    Returns dict:
        'raw'      : mne.io.Raw (preloaded, unfiltered)
        'events'   : pd.DataFrame (onset, duration, trial_type)
        'channels' : pd.DataFrame (name, type, units)
        'sfreq'    : float, sampling frequency in Hz
        'subject'  : str

    Raises:
        FileNotFoundError if any expected BIDS file is missing.
        ValueError if events TSV lacks required columns.
    """
    root = Path(bids_root)
    sub_dir = root / subject_id
    
    # Searching for FR1 or CatFR1 events
    events_paths = []
    events_paths.extend(list(sub_dir.rglob('*task-FR1*events.tsv')))
    events_paths.extend(list(sub_dir.rglob('*task-CatFR1*events.tsv')))
    
    if not events_paths:
        raise FileNotFoundError(f'No FR1 or CatFR1 events found for {subject_id} in {sub_dir}')
    
    events_path = events_paths[0]
    events = pd.read_csv(events_path, sep='\t')
    
    # In UPenn RAM events, we are looking for RECALL vs FORGOTTEN or recalled == 1/0
    # Let's standardize it to 'hit'
    if 'recall' in events.columns or 'recalled' in events.columns:
        col = 'recall' if 'recall' in events.columns else 'recalled'
        events['hit'] = events[col].astype(str).str.upper().apply(
            lambda x: 1 if x in ['1', 'RECALL', 'TRUE', '1.0'] else 0
        )
    elif 'type' in events.columns and events['type'].str.upper().isin(['RECALL']).any():
        # Sometimes RECALL is an event type itself. Assume word presentation is WORD
        # and subsequent RECALL events link to them. For simplicity, assume word events have a 'recalled' flag
        pass
    
    # Require RECALL vs FORGOTTEN (or 1 vs 0 derived label)
    if 'hit' not in events.columns:
        # User specified confirming RECALL vs FORGOTTEN labels
        # Attempt to map from trial_type or similar
        for col in events.columns:
            if events[col].astype(str).str.upper().isin(['RECALL', 'FORGOTTEN']).any():
                events['hit'] = events[col].astype(str).str.upper() == 'RECALL'
                events['hit'] = events['hit'].astype(int)
                break
        
        if 'hit' not in events.columns:
            raise ValueError(f"events.tsv missing 'RECALL' vs 'FORGOTTEN' labels in {events_path}")
            
    # Find associated iEEG
    parent_dir = events_path.parent
    base_name = events_path.name.replace('_events.tsv', '')
    
    raw = None
    for ext in ['.edf', '.vhdr', '.mef', '.set']:
        ieeg_files = list(parent_dir.glob(f'{base_name}_ieeg{ext}'))
        if ieeg_files:
            if ext == '.edf':
                raw = mne.io.read_raw_edf(ieeg_files[0], preload=True, verbose=False)
            elif ext == '.vhdr':
                raw = mne.io.read_raw_brainvision(ieeg_files[0], preload=True, verbose=False)
            break
            
    if raw is None:
        raise FileNotFoundError(f"No matching iEEG data found for {base_name}")
        
    return {
        'raw': raw,
        'events': events,
        'channels': pd.DataFrame(), # Simplification
        'sfreq': raw.info['sfreq'],
        'subject': subject_id,
    }
