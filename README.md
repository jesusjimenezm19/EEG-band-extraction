# EEG preprocessing and band-power extraction (MNE-Python)

`eeg_band_extraction.py` processes raw BrainVision EEG recorded during a probabilistic go/no-go task and extracts time-resolved oscillatory power in the theta, alpha and beta bands for each trial category.

## Processing steps

1. **Trial classification** from the event markers: target codes define go/no-go trials and task difficulty (high, medium, low go probability); outcome codes classify trials as correct, false alarm, late response or omission.
2. **Filtering**: 0.75–40 Hz band-pass.
3. **Epoching**: −3.0 to +2.3 s around the target marker.
4. **Noisy channels**: detected from the robust per-epoch standard deviation and interpolated (spherical splines).
5. **Re-referencing** to the common average.
6. **ICA** (Picard, 15 components) with Fp1/Fp2 as EOG proxies to remove ocular components.
7. **Artefact rejection**: epochs exceeding ±90 µV (excluding Fp1/Fp2).
8. **Surface Laplacian** (current source density, Legendre order 10).
9. **Time–frequency decomposition**: Morlet wavelets, 4–25 Hz in 1-Hz steps, 3 to 10 cycles.
10. **Band power by region of interest**: midfrontal theta (4–7 Hz; Fz, FC1, FC2), posterior alpha (8–12 Hz; P7, P8, P3, P4, O1, O2) and left-central beta (15–25 Hz; CP5, CP1, C3), averaged across trials for each trial category.
11. **Target-onset estimate**: occipital ERP peak (O1, Oz, O2) between 0.5 and 0.8 s after the marker.

## Requirements

Python ≥ 3.10 with `mne`, `numpy`, `pandas` and `python-picard`.

## Usage

Place the BrainVision files (`.vhdr`, `.vmrk`, `.eeg`) in `data/raw/` or set the environment variable `EEG_RAW_DIR`, and provide `tf_bands_meta.csv` (one row per recording, with columns `Subject`, `BaseSubject` and `Fase`) in the same folder as the script. Then run:

```bash
python eeg_band_extraction.py
```

Output: `bandas_realineado_raw.npz` (recording × trial category × band × time, absolute power) and `bandas_realineado_meta.csv` (trials retained per category and quality indicators).

No participant data are included in this repository.
