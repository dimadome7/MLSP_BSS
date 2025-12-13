import numpy as np
import os
import random
import tqdm
from sklearn.decomposition import NMF
import scipy.ndimage as ndi
import soundfile as sf
import librosa
import museval
import warnings
import csv

warnings.filterwarnings('ignore')


####################################################################################################
# Config
####################################################################################################

config = {

    # dataset
    'root': 'musdb18_wav/train',
    'out_dir': 'outputs',
    'use_subset': True,
    'n_samples': 2,
    'n_fft': 2048,
    'hop': 512,
    'random_seed': 100,

    # NMF
    'n_components': 30,
    'max_iter': 450,

    # scoring:
    # scale for panning requirement (larger means allowing more off-center energy)
    'pan_norm': 0.6,
    # vocal frequency range (Hz)
    'freq_low': 150.0,
    'freq_high': 5000.0,
    # minimum vocal-band ratio to be considered vocal component
    'freq_ratio_cutoff': 0.3,

    # weights of score contributions / penalties
    'pan_weight': 2.0,
    'temp_weight': 2.5,
    'freq_weight': 1.0,
    'flat_weight': 1.0,

    # number of components to use in vocal reconstruction
    'K': 10,

    # mask exponent
    'p': 2.0

}


####################################################################################################
# Helper Functions
####################################################################################################


'''
function: spectral flatness
param: mag_X magnitude spectrum)
return: flatness (spectral flatness of signal)
'''
def spectral_flattness(mag_X):
    eps = 1e-8
    flatness = np.exp(np.mean(np.log(mag_X + eps))) / (np.mean(mag_X) + eps)

    return flatness

'''
function: compute bss metrics during evaluation
param: v_ref (reference track), v_est (estimation of reference track)
return: bss metrics
'''
def compute_bss_metrics(v_ref, v_est):

    # make sure shape is [T,2]
    if v_ref.shape[0] == 2 and v_ref.ndim == 2:
        v_ref = v_ref.T

    if v_est.shape[0] == 2 and v_est.ndim == 2:
        v_est = v_est.T

    # museval wants[n_src, n_samples, n_channels]
    v_ref = v_ref[np.newaxis, :, :]
    v_est = v_est[np.newaxis, :, :]

    # crop to common length
    min_len = min(v_ref.shape[1], v_est.shape[1])
    v_ref = v_ref[:, :min_len, :]
    v_est = v_est[:, :min_len, :]

    SDR, ISR, SIR, SAR, perm = museval.metrics.bss_eval(v_ref, v_est)

    # return metrics
    return {
        "SDR": float(np.nanmean(SDR)),
        "ISR": float(np.nanmean(ISR)),
        "SIR": float(np.nanmean(SIR)),
        "SAR": float(np.nanmean(SAR)),
    }

'''
function: return safe differences, prevent inf - inf and nan
return a - b or nan
'''
def safe_delta(a, b):
        if np.isfinite(a) and np.isfinite(b):
            return float(a - b)
        else:
            return np.nan
        

# function: check energy at harmonic multiples of the peak
def get_harmonicity_score(spectrum, n_harmonics=5, min_fundamental_bin=5):

    # normalize spectrum
    spec = spectrum / (np.max(spectrum) + 1e-9)

    # fundamental frequency, ignore fundamentals that are too low
    f0 = np.argmax(spec)
    if f0 < min_fundamental_bin:
        return 0.0

    score = 0.0

    # check multiples of fundamental to add to score
    for h in range(2, n_harmonics + 1):
        idx = f0 * h
        if idx < len(spec):
            score += spec[idx]
        else:
            break
    
    # normalize the scores with fundamental
    return score / (spec[f0] + 1e-9)


####################################################################################################
# Importing Data
####################################################################################################


# import sample tracks from MUSDB18 dataset:
# safely load all tracks that have a mixture file and are formatted correctly as a folder
print('Loading wav files...')
data = []
for d in sorted(os.listdir(config['root'])):
    track_dir = os.path.join(config['root'], d)
    mix_path = os.path.join(track_dir, 'mixture.wav')
    if os.path.isdir(track_dir) and os.path.isfile(mix_path):
        data.append(d)

data = np.array(data)
data_len = len(data)

# set random seed
np.random.seed(config['random_seed'])

# load specified number of sample tracks
if config['use_subset'] == True:
    n_samples = config['n_samples']
    sample_idxs = np.random.randint(0, data_len, size=n_samples)
    sample_paths = data[sample_idxs]
    print(f'samples selected: {sample_paths}')
else:
    n_samples = data_len
    sample_paths = data

# prepent sample paths for loading
full_sample_paths = [os.path.join(config['root'], path, 'mixture.wav') for path in sample_paths]
vocal_sample_paths = [os.path.join(config['root'], path, 'vocals.wav') for path in sample_paths]

# extract stereo file, and isolated channels
samples = []
samples_L = []
samples_R = []

for path in full_sample_paths:
    y, sr = librosa.load(path, mono=False, sr=None)
    y = y / np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else y
    samples.append(y)
    samples_L.append(y[0,:])
    samples_R.append(y[1,:])

print("Audio loaded.")


####################################################################################################
# STFT
####################################################################################################


mag_mono = []
mag_L = []
mag_R = []
phase_L = []
phase_R = []
stft_L = []
stft_R = []

# for each sample: compute and store magnitude spectrum, phase, stft for each channel (as well as mono)
print('Computing STFTs...')
for i in tqdm.tqdm(range(n_samples)):

    stft_left = librosa.stft(samples_L[i], n_fft=config['n_fft'], hop_length=config['hop'])
    stft_right = librosa.stft(samples_R[i], n_fft=config['n_fft'], hop_length=config['hop'])

    mag_left = np.abs(stft_left)
    mag_right = np.abs(stft_right)
    phase_left = np.angle(stft_left)
    phase_right = np.angle(stft_right)
    mag = mag_left + mag_right

    mag_mono.append(mag)
    mag_L.append(mag_left)
    mag_R.append(mag_right)
    phase_L.append(phase_left)
    phase_R.append(phase_right)
    stft_L.append(stft_left)
    stft_R.append(stft_right)

print("STFT done.")


####################################################################################################
# NMF
####################################################################################################


# lists to append W and H's of each sample track
W = []
H_L = []
H_R = []
epsilon = 1e-9

# create NMF model
model = NMF(n_components=config['n_components'], init='nndsvda', random_state=0, max_iter=config['max_iter'], solver='mu', beta_loss='itakura-saito', l1_ratio=0.0)

print('Computing NMF...')

for i in tqdm.tqdm(range(n_samples)):
    
    # NMF of mono signal
    X_mono = mag_mono[i].T + epsilon
    model.fit(X_mono)
    print(f"NMF iterations used for sample {i}: {model.n_iter_}")

    W_learned = model.components_.T 
    H_left_T = model.transform(mag_L[i].T + epsilon)
    H_right_T = model.transform(mag_R[i].T + epsilon)
      
    H_left = H_left_T.T
    H_right = H_right_T.T

    W.append(W_learned)
    H_L.append(H_left)
    H_R.append(H_right)

# With H_left and right, we now have activations derived jointly from the specific channel data
print("NMF factorization done.")


####################################################################################################
# Channel Analysis
####################################################################################################


# outcomes
y_vocals_final = []

# perform analysis on each sample track
for i in tqdm.tqdm(range(n_samples)):

    print(f'Analysing Sample {i+1}')

    stft_left  = stft_L[i]
    stft_right = stft_R[i]

    # mono magnitude for this track
    mag_mono_i = mag_mono[i]

    # frequency bins
    freq_bins = librosa.fft_frequencies(sr=sr, n_fft=config['n_fft'])

    # vocal band indices
    vocal_band_idx = np.where((freq_bins >= config['freq_low']) & (freq_bins <= config['freq_high']))[0]

    # time varying energy in vocal band from mixture
    vocal_band_energy_t = np.sum(mag_mono_i[vocal_band_idx, :], axis=0)

    # normalize and zero mean
    eps = 1e-9
    vocal_activity = vocal_band_energy_t - np.mean(vocal_band_energy_t)
    vocal_activity /= (np.linalg.norm(vocal_activity) + eps)

    # component scores
    scores = np.full(config['n_components'], -1e5, dtype=np.float32)

    for k in range(config['n_components']):
        Hk_L = H_L[i][k, :]
        Hk_R = H_R[i][k, :]
        Hk_total = Hk_L + Hk_R

        # panning score
        energy_left  = np.sum(Hk_L)
        energy_right = np.sum(Hk_R)
        denom = energy_left + energy_right + eps
        panning_diff = np.abs(energy_left - energy_right) / denom
        pan_score = 1.0 - (panning_diff / config['pan_norm'])
        pan_score = float(np.clip(pan_score, 0.0, 1.0))

        # temporal similarity
        Hk_centered = Hk_total - np.mean(Hk_total)
        Hk_centered /= (np.linalg.norm(Hk_centered) + eps)
        temporal_sim = float(np.dot(Hk_centered, vocal_activity))
        temp_score = max(0.0, temporal_sim)

        # vocal band ratio
        basis_spectrum = W[i][:, k]
        total_energy = np.sum(basis_spectrum) + eps
        energy_vocal_band = np.sum(basis_spectrum[vocal_band_idx])
        vocal_band_ratio = float(energy_vocal_band / total_energy)

        # reject if not enough energy in vocal band
        if vocal_band_ratio < config['freq_ratio_cutoff']:
            continue

        # flatness score
        flat = spectral_flattness(basis_spectrum)
        flat_score = max(0.0, (0.5 - flat) / 0.5)

        # final score
        scores[k] = (
            config['pan_weight']  * pan_score +
            config['temp_weight'] * temp_score +
            config['freq_weight'] * vocal_band_ratio +
            config['flat_weight'] * flat_score
        )

    # take top-K 'vocal' components
    K = config['K']
    vocal_indices = np.argsort(scores)[-K:]
    instrumental_indices = [idx for idx in range(config['n_components']) if idx not in vocal_indices]

    # Reconstruct vocal magnitudes
    Wi = W[i]
    HL = H_L[i]
    HR = H_R[i]
    V_L = Wi[:, vocal_indices] @ HL[vocal_indices, :] 
    V_R = Wi[:, vocal_indices] @ HR[vocal_indices, :]
    I_L = Wi[:, instrumental_indices] @ HL[instrumental_indices, :]
    I_R = Wi[:, instrumental_indices] @ HR[instrumental_indices, :]

    # Reconstruct mixture magnitudes
    X_L = Wi @ HL
    X_R = Wi @ HR

    # Soft masks
    p = config['p']
    mask_L = (V_L**p) / ((V_L**p) + (I_L**p) + eps)
    mask_R = (V_R**p) / ((V_R**p) + (I_R**p) + eps)

    # clip masks
    mask_L = np.clip(mask_L, 0.0, 1.0)
    mask_R = np.clip(mask_R, 0.0, 1.0)

    # mask smoothing
    mask_L = ndi.uniform_filter1d(mask_L, size=5, axis=1, mode="nearest")
    mask_R = ndi.uniform_filter1d(mask_R, size=5, axis=1, mode="nearest")

    # clip masks again
    mask_L = np.clip(mask_L, 0.0, 1.0)
    mask_R = np.clip(mask_R, 0.0, 1.0)
    

    # apply masks:
    # trim to min length
    min_frames = min(stft_left.shape[1], mask_L.shape[1], mask_R.shape[1])
    stft_left  = stft_left[:, :min_frames]
    stft_right = stft_right[:, :min_frames]
    mask_L     = mask_L[:, :min_frames]
    mask_R     = mask_R[:, :min_frames]

    # apply masks
    stft_left_vocal  = stft_left  * mask_L
    stft_right_vocal = stft_right * mask_R

    # reconstruct vocal
    y_left_vocal  = librosa.istft(stft_left_vocal, hop_length=config['hop'], length=len(samples_L[i]))
    y_right_vocal = librosa.istft(stft_right_vocal, hop_length=config['hop'], length=len(samples_R[i]))

    y_vocal_stereo = np.vstack((y_left_vocal, y_right_vocal))
    y_vocals_final.append(y_vocal_stereo)

    os.makedirs(config['out_dir'], exist_ok=True)
    output_path = os.path.join(config['out_dir'], f'0{i}_vocal_separated_v2.wav')
    sf.write(output_path, y_vocal_stereo.T, sr)
    print(f"Saved to {output_path}")


####################################################################################################
# Evaluation
####################################################################################################


results = []

# load samples from musdb
vocal_refs = []
for path in vocal_sample_paths:
    y, sr = librosa.load(path, mono=False, sr=None)
    vocal_refs.append(y)

# assert we are comparing the correct tracks
for i in range(n_samples):
    
    track_name = sample_paths[i]
    print(f'Evaluating track {i+1}: {track_name}...')

    # gather estimate
    v_ref = vocal_refs[i]
    v_hat = y_vocals_final[i]

    # skip if reference is too quiet
    if np.sum(v_ref ** 2) < 1e-6:
        print(f'Reference vocal nearly silent. Skipping track {i+1}.')
        continue

    # compute our model metrics
    model_metrics = compute_bss_metrics(v_ref, v_hat)
    print(f"model metrics:  SDR={model_metrics['SDR']:.3f}, SAR={model_metrics['SAR']:.3f}")

    # performance using mixture as estimate
    mix_stereo, _ = librosa.load(full_sample_paths[i], mono=False, sr=None)
    mixture_metrics = compute_bss_metrics(v_ref, mix_stereo)
    print(f"mixture metrics:  SDR={mixture_metrics['SDR']:.3f}, SAR={mixture_metrics['SAR']:.3f}")

    # performance of hpss baseline
    harm = np.zeros_like(mix_stereo)
    for ch in range(mix_stereo.shape[0]):
        y_ch = mix_stereo[ch]  # (T,)
        y_harm, y_perc = librosa.effects.hpss(y_ch)
        harm[ch] = y_harm

    hpss_metrics = compute_bss_metrics(v_ref, harm)
    print(f"hpss metrics:  SDR={hpss_metrics['SDR']:.3f}, SAR={hpss_metrics['SAR']:.3f}\n")

    # log all statistics for this track:
    results.append({
        'track': track_name,

        'model_SDR': model_metrics['SDR'],
        'model_ISR': model_metrics['ISR'],
        'model_SIR': model_metrics['SIR'],
        'model_SAR': model_metrics['SAR'],

        'mix_SDR': mixture_metrics['SDR'],
        'mix_ISR': mixture_metrics['ISR'],
        'mix_SIR': mixture_metrics['SIR'],
        'mix_SAR': mixture_metrics['SAR'],

        'hpss_SDR': hpss_metrics['SDR'],
        'hpss_ISR': hpss_metrics['ISR'],
        'hpss_SIR': hpss_metrics['SIR'],
        'hpss_SAR': hpss_metrics['SAR'],

        'delta_SDR_vs_mix': safe_delta(model_metrics['SDR'], mixture_metrics['SDR']),
        'delta_SDR_vs_hpss': safe_delta(model_metrics['SDR'], hpss_metrics['SDR']),
        'delta_SAR_vs_mix': safe_delta(model_metrics['SAR'], mixture_metrics['SAR']),
        'delta_SAR_vs_hpss': safe_delta(model_metrics['SAR'], hpss_metrics['SAR']),

    })

# save results
print(f'Evaluated {len(results)} tracks.')

# summary
model_SDR = np.array([r['model_SDR'] for r in results])
mix_SDR = np.array([r['mix_SDR'] for r in results])
hpss_SDR = np.array([r['hpss_SDR'] for r in results])
delta_SDR_mix = np.array([r['delta_SDR_vs_mix'] for r in results])
delta_SDR_hpss = np.array([r['delta_SDR_vs_hpss'] for r in results])
delta_SAR_mix = np.array([r['delta_SAR_vs_mix'] for r in results])
delta_SAR_hpss = np.array([r['delta_SAR_vs_hpss'] for r in results])

print('global SDR summary:')
print(f'Mean model SDR:   {np.nanmean(model_SDR):.3f} dB')
print(f'Mean mix SDR:    {np.nanmean(mix_SDR):.3f} dB')
print(f'Mean hpss SDR:   {np.nanmean(hpss_SDR):.3f} dB')
print(f'Mean ΔSDR vs mix:  {np.nanmean(delta_SDR_mix):.3f} dB')
print(f'Mean ΔSDR vs HPSS: {np.nanmean(delta_SDR_hpss):.3f} dB')
print(f'Mean ΔSAR vs mix:  {np.nanmean(delta_SAR_mix):.3f} dB')
print(f'Mean ΔSAR vs HPSS: {np.nanmean(delta_SAR_hpss):.3f} dB')

print('\nTracks where we beat HPSS (SDR): '
      f"{np.sum(delta_SDR_hpss > 0)} / {len(results)}")
print('Tracks where we beat MIX (SDR):  '
      f"{np.sum(delta_SDR_mix > 0)} / {len(results)}")

print('\nTracks where we beat HPSS (SAR): '
      f"{np.sum(delta_SAR_hpss > 0)} / {len(results)}")
print('Tracks where we beat MIX (SAR):  '
      f"{np.sum(delta_SDR_mix > 0)} / {len(results)}")

# write CSV
csv_path = os.path.join(config['out_dir'],'musdb18_eval_results.csv')
fieldnames = list(results[0].keys())

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f'\nSaved detailed per-track results to: {csv_path}')