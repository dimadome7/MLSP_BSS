import numpy as np
import os
import random
import tqdm
from sklearn.decomposition import NMF
import soundfile as sf
import scipy
import librosa
import dsdtools
import musdb
import museval
import warnings

warnings.filterwarnings('ignore')

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


####################################################################################################
# Importing Data
####################################################################################################

# import sample tracks from MUSDB18 dataset
data_root = 'musdb18_wav/train'
data = np.array(sorted(os.listdir(data_root)))
data_len = len(data)

# load specified number of sample tracks
np.random.seed(6)
n_samples = 5
sample_idxs = np.random.randint(0, data_len, size=n_samples)
sample_paths = data[sample_idxs]
print(f'samples selected: {sample_paths}')

# prepent sample paths for loading
full_sample_paths = [os.path.join(data_root, path, 'mixture.wav') for path in sample_paths]
vocal_sample_paths = [os.path.join(data_root, path, 'vocals.wav') for path in sample_paths]

# extract stereo file, and isolated channels
samples = []
samples_L = []
samples_R = []

for path in full_sample_paths:
    y, sr = librosa.load(path, mono=False, sr=None)
    samples.append(y)
    samples_L.append(y[0,:])
    samples_R.append(y[1,:])

print("Audio loaded.")

####################################################################################################
# STFT
####################################################################################################

# stft parameters
n_fft_default = 2048
HOP_LENGTH = 512

mag_mono = []
mag_L = []
mag_R = []
phase_L = []
phase_R = []

# for each sample: compute and store magnitude spectrum and phase for each channel (as well as mono)
for i in tqdm.tqdm(range(n_samples)):

    stft_left = librosa.stft(samples_L[i], n_fft=n_fft_default, hop_length=HOP_LENGTH)
    stft_right = librosa.stft(samples_R[i], n_fft=n_fft_default, hop_length=HOP_LENGTH)

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
n_components = 30 # between 30-60
model = NMF(n_components=n_components, init='random', random_state=0, max_iter=1000, solver='mu', beta_loss='itakura-saito')

for i in tqdm.tqdm(range(n_samples)):

    X_mono = mag_mono[i].T + epsilon
    X_left = mag_L[i].T + epsilon
    X_right = mag_R[i].T + epsilon

    # perform NMF on mono signal
    model.fit(X_mono)

    # As we're assuming no elements that phase cancel out completely
    # aka no 'close to negative 1' correlation for any elements of the track.
    W_learned = model.components_.T 

    H_left_T = model.transform(X_left)
    H_right_T = model.transform(X_right)

    H_left = H_left_T.T
    H_right = H_right_T.T

    W.append(W_learned)
    H_L.append(H_left)
    H_R.append(H_right)

# so with H_left and right, I'm saying that we're using the bases in W learned from the mono mix
# and we're seeing how much of those components are present in the left and right channels respectively.
print("NMF factorization done.")

####################################################################################################
# Channel Analysis
####################################################################################################

# How close the energy in both channels should be for a component to be centered in the stereo field
panning_threshold = 0.02 

# Maybe we can do male vs female ranges later?
freq_range_hz = (150, 5000)

# FFT parameters
n_fft_default_forbins = 2048
HOP_LENGTH_forbins = 512

# outcomes
y_vocals_final = []

# perform analysis on each sample track
for i in tqdm.tqdm(range(n_samples)):

    print(f'Analysing Sample {i+1}')
    
    vocal_components = []
    instrumental_components = []

    freq_bins = librosa.fft_frequencies(sr=sr, n_fft=n_fft_default_forbins)
    vocal_freq_indices = np.where((freq_bins >= freq_range_hz[0]) & (freq_bins <= freq_range_hz[1]))[0] # bins of vocal presence

    for k in range(n_components):
        # Stereo field
        energy_left = np.sum(H_L[i][k, :])
        energy_right = np.sum(H_R[i][k, :])
        panning_diff = np.abs(energy_left - energy_right) / (energy_left + energy_right + 1e-6) # to prevent a 
        # division by zero if that element has complete negative phase correlation.

        # Dividing by mono energy normalzes panning diff to between 0 and 1.
        is_center_panned = panning_diff < panning_threshold

        # Frequency filtering
        basis_spectrum = W[i][:, k]
        spectral_centroid_hz = np.sum(freq_bins * basis_spectrum) / (np.sum(basis_spectrum) + 1e-6)
        is_vocal_frequency = freq_range_hz[0] <= spectral_centroid_hz <= freq_range_hz[1]

        # spectral flatness tends to be low for speech (peaky / noisy)
        flatness = spectral_flattness(W[i][:,k])
        is_not_flat = flatness < 0.4

        # I'm still very unhappy with the frequency cue implementation - I want to find ways to see if I can capture how
        # the energy of a vocal changes over the frequency spectrum over time, rather than a static centroid. 

        # Decision
        if is_center_panned and is_vocal_frequency:
            vocal_components.append(k)
        else:
            instrumental_components.append(k)

    print(f"Found {len(vocal_components)} vocal components and {len(instrumental_components)} instrumental components.")

    W_vocal = W[i][:, vocal_components]
    H_left_vocal = H_L[i][vocal_components, :]
    H_right_vocal = H_R[i][vocal_components, :]

    W_instrumental = W[i][:, instrumental_components]
    H_left_instrumental = H_L[i][instrumental_components, :]
    H_right_instrumental = H_R[i][instrumental_components, :]

    mag_left_vocal_recon = W_vocal @ H_left_vocal
    mag_right_vocal_recon = W_vocal @ H_right_vocal
    mag_left_instrumental_recon = W_instrumental @ H_left_instrumental
    mag_right_instrumental_recon = W_instrumental @ H_right_instrumental

    # I'm reconstructing instruments as well just for checking results to optimize our model. 

    # creating soft masks (Wiener-style) - basically minimizing MSE
    mag_total_vocal = mag_left_vocal_recon**2 + mag_right_vocal_recon**2
    mag_total_inst = mag_left_instrumental_recon**2 + mag_right_instrumental_recon**2

    # apply smoothing to mask
    vocal_mask = mag_total_vocal / (mag_total_vocal + mag_total_inst + epsilon)
    vocal_mask = scipy.ndimage.gaussian_filter(vocal_mask, sigma=1.2)

    # determine min frames to match mask and stft
    min_frames = min(stft_left.shape[1], vocal_mask.shape[1])
    stft_left = stft_left[:,:min_frames]
    stft_right = stft_right[:,:min_frames]
    vocal_mask = vocal_mask[:,:min_frames]
    
    stft_left_vocal = stft_left * vocal_mask
    stft_right_vocal = stft_right * vocal_mask

    y_left_vocal = librosa.istft(stft_left_vocal)
    y_right_vocal = librosa.istft(stft_right_vocal)

    # stereo output
    y_vocal_stereo = np.vstack((y_left_vocal, y_right_vocal))
    y_vocals_final.append(y_vocal_stereo)

    output_path = f'0{i}_vocal_separated_v2.wav'
    sf.write(output_path, y_vocal_stereo.T, sr)

    print(f"Saved to {output_path}")

####################################################################################################
# Evaluation
####################################################################################################

# load samples from musdb
vocal_refs = []
for path in vocal_sample_paths:
    y, sr = librosa.load(path, mono=False, sr=None)
    vocal_refs.append(y)

# assert we are comparing the correct tracks
for i in range(n_samples):
    
    print(f'sample name: {sample_paths[i]}')
    print(f'track name: {vocal_sample_paths[i]}')

    v_ref = vocal_refs[i].T
    v_hat = y_vocals_final[i].T

    # adjust shape to match expecation [n_sources, n_samples, n_channels]
    v_ref = v_ref[np.newaxis, :, :]
    v_hat = v_hat[np.newaxis, :, :]

    # trim to same length to ger rid of extra frames
    min_len = min(v_ref.shape[1], v_hat.shape[1])
    v_ref = v_ref[:, :min_len, :]
    v_hat = v_hat[:, :min_len, :]

    # ignore nearly silent vocal cases
    if np.sum(v_ref**2) < 1e-6:
        print('skipping evaluation, reference vocal nearly silent.')
        continue

    # compute BSS performance metrics
    (SDR, ISR, SIR, SAR, perm) = museval.metrics.bss_eval(v_ref, v_hat)

    print(f'results (nan-mean) for sample {i+1}:')
    print(f'SDR: {np.nanmean(SDR)}')
    print(f'ISR: {np.nanmean(ISR)}')
    print(f'SIR: {np.nanmean(SIR)}')
    print(f'SAR: {np.nanmean(SAR)}')