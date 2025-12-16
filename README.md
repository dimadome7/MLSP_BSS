MLSP Project: Blind Sourse Separation

This repository contains code attemtping blind source separation (BSS) with multi-channel non-negative matrix factorization techniques (MC-NMF).

The MUSDB118 Dataset used is available for download at: https://zenodo.org/records/1117372

Code instructions:
1.  Download the dataset from the link above.
2.  Install dataset Python library: pip install musdb
3.  Convert dataset to .wav: musdbconvert path/to/musdb-stems-root musdb18_wav
5.  Visit sigstep-mus-db Github for more info: https://github.com/sigsep/sigsep-mus-db
6.  Ensure file format follows this structure:

musdb18_wav:
    train:
        A Classic Education - NightOwl:
            accompaniment.wav
            bass.wav
            drums.wav
            linear_mixture.wav
            mixture.wav
            other.wav
            vocals.wav
        .
        .
        .
    test:
        .
        .
        .

7.  Ensure python>=3.11
8.  Use ./run.sh to install requirements and perform vocal separation on the dataset.
9.  To compute vocal for a subset of tracks, edit config['n_samples'] and config['use_subset'] in main.py
