# Theta Memory Encoding (Validation pipeline for Biba et al. 2026)

This repository serves as a direct neurophysiological validation pipeline of the behavioral memory rhythmic effects identified by **Biba et al. (2026)**. It replaces classical synthetic modeling with the empirical extraction of the **Subsequent Memory Effect (SME)** directly from human intracranial EEG (iEEG) arrays.

## Core Features
- **Zero-Synthetic Dataset:** Pre-configured to ingest explicit `FR1` and `CatFR1` memory sessions from the **UPenn RAM Dataset** (OpenNeuro Accession: `ds004100`).
- **Targeted Spectral Integrity:** Integrates with `fooof` to parameterize 1/f aperiodic noise and isolate exclusively periodic 3-10 Hz theta components with a rigid constraint threshold (`R^2 > 0.90`).
- **Biologically-Anchored Phase Locking (PLV):** Employs explicit Morlet wavelet convolutions aligned to the visual word presentation onset (simulating native visual saccade resets described by the SPEAR model) to contrast remembered versus forgotten phase-states across the ~100ms post-saccadic inhibition window.
- **Cholinergic Extension Testing:** Facilitates direct permutation testing of cholinergic hypotheses (e.g. Nicotine attenuation profiles).

## Requirements
```bash
pip install -r requirements.txt
```
*Note: A functional local clone of BIDS `ds004100` targeting hippocampal leads is expected in `data/raw/bids/ds004100`.*

## Execution
Initialize the analytical pipeline leveraging parallel extraction processing:
```bash
python main.py
```
*Outputs dynamically map to a central `ds004100_ram_engram.json` repository index.*

## Project Layout
```
├── src/
│   ├── data_io.py      # BIDS/MNE UPenn RAM extraction architecture
│   ├── spectral.py     # FOOOF separation and fitting logic
│   └── stats.py        # PLV derivation, epoching, and SME differentials
├── figures/            # Generative location for simulation and analysis arrays
├── results/            # Persistent storage layer for final engrams
├── main.py             # System orchestrator
└── requirements.txt
```
