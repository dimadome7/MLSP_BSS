## Overview
This repository contains a blind source separation (BSS) pipeline utilizing multi-channel non-negative matrix factorization techniques (MC-NMF). We introduce several extensions of a MC-NMF baseline to improve separation performance. These include replacing hard component assignment rules with a weighted scoring framework, introducing adaptive per-track component selection, and refining the reconstruction stage using soft Wiener masking.

## Results
Across 100 MUSDB18 tracks, our proposed MC-NMF pipeline achieves a mean SDR of ***-6.64 dB***, representing an average improvement of ***5.71 dB*** over the raw mixture and ***3.65 dB*** over the HPSS baseline. The method improves SDR relative to the mixture on ***92%*** of tracks and outperforms HPSS on ***88%*** of tracks, indicating consistent gains across a variety of musical material.

Our method also improves Source-to-Artifacts Ratio (SAR) by ***5.23 dB*** relative to the mixture and ***3.70 dB*** relative to HPSS. This suggests that the proposed soft scoring and masking strategy reduces musical noise compared to simpler baselines. Despite these improvements, performance remains limited by the fully blind nature of the approach, and some residual accompaniment leakage persists, particularly in dense mixes with strong harmonic instruments.

## Instructions
1.  Download the MUSDB18 dataset at: https://zenodo.org/records/1117372
2.  Install dataset Python library: ```pip install musdb```
3.  Convert dataset to .wav: ```musdbconvert path/to/musdb-stems-root musdb18_wav```
4.  Visit sigstep-mus-db Github for more info: https://github.com/sigsep/sigsep-mus-db
5.  Ensure file format follows this structure:

```text
musdb18_wav/
├── train/
│   ├── A Classic Education - NightOwl/
│   │   ├── accompaniment.wav
│   │   ├── bass.wav
│   │   ├── drums.wav
│   │   ├── linear_mixture.wav
│   │   ├── mixture.wav
│   │   ├── other.wav
│   │   └── vocals.wav
│   └── ...
└── test/
│   ├── ...
```

6.  Ensure ```python>=3.11```
7.  Use ```./run.sh``` to install requirements and perform vocal separation on the dataset.
8.  To compute vocal for a subset of tracks, edit ```config['n_samples']``` and ```config['use_subset']``` in ```main.py```
