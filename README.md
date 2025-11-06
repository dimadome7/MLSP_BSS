MLSP Project: Blind Sourse Separation

This repository contains code attemtping blind source separation (BSS) with multi-channel non-negative matrix factorization techniques (MC-NMF).

The MUSDB118 Dataset used is available for download at: https://zenodo.org/records/1117372

Code instructions:
1. Download the dataset from the link above.
2. Install dataset Python library: pip install musdb
3. Convert dataset to .wav: musdbconvert path/to/musdb-stems-root musdb18_wav
5. Visit sigstep-mus-db Github for more info: https://github.com/sigsep/sigsep-mus-db
6. Recover vocal recordings from different songs in the dataset.
