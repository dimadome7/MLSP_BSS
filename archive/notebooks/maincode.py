import librosa
import numpy as np
from sklearn.decomposition import NMF
import soundfile as sf
import dsdtools


#testing with one song

audio_path = 'Hips_Dont_Lie_Original.mp3'
y, sr = librosa.load(audio_path, sr=None, mono=False)

print(f'y shape: {y.shape}')

y_left = y[0, :]
y_right = y[1, :]

print("Audio loaded.")

#-----------------------

# Playing around parameters ----------
n_fft_default = 2048
HOP_LENGTH = 512
# ---------------
stft_left = librosa.stft(y_left, n_fft=n_fft_default, hop_length=HOP_LENGTH)
stft_right = librosa.stft(y_right, n_fft=n_fft_default, hop_length=HOP_LENGTH)

mag_left = np.abs(stft_left)
mag_right = np.abs(stft_right)
phase_left = np.angle(stft_left)
phase_right = np.angle(stft_right)
mag_mono = mag_left + mag_right

print("STFT done.")

#-----------------------
n_components = 50 # between 30-60 to test out maybe

X_mono = mag_mono.T
X_left = mag_left.T
X_right = mag_right.T


model = NMF(n_components=n_components, init='random', random_state=0, max_iter=300, solver='mu', beta_loss='kullback-leibler')

model.fit(X_mono)
# As we're assuming no elements that phase cancel out completely
# aka no 'close to negative 1' correlation for any elements of the track.
W = model.components_.T 

H_left_T = model.transform(X_left)
H_right_T = model.transform(X_right)

H_left = H_left_T.T
H_right = H_right_T.T
# so with H_left and right, I'm saying that we're using the bases in W learned from the mono mix
# and we're seeing how much of those components are present in the left and right channels respectively.
print("NMF factorization done.")


#------------------------
vocal_components = []
instrumental_components = []

panning_threshold = 0.02 # How close the energy in both channels should be for a component to be centered in the stereo field
# we can try see if adjusting this helps
freq_range_hz = (150, 5000) # Maybe we can do male vs female ranges later?

# Playing around parameters ----------
n_fft_default_forbins = 2048
HOP_LENGTH_forbins = 512
# --------------

freq_bins = librosa.fft_frequencies(sr=sr, n_fft=n_fft_default_forbins)
vocal_freq_indices = np.where((freq_bins >= freq_range_hz[0]) & (freq_bins <= freq_range_hz[1]))[0] # bins of vocal presence

for k in range(n_components):
    # Stereo field
    energy_left = np.sum(H_left[k, :])
    energy_right = np.sum(H_right[k, :])
    panning_diff = np.abs(energy_left - energy_right) / (energy_left + energy_right + 1e-6) # to prevent a 
    # division by zero if that element has complete negative phase correlation.

    # Dividing by mono energy normalzes panning diff to between 0 and 1.
    is_center_panned = panning_diff < panning_threshold

    # Frequency filtering
    basis_spectrum = W[:, k]
    spectral_centroid_hz = np.sum(freq_bins * basis_spectrum) / (np.sum(basis_spectrum) + 1e-6)
    is_vocal_frequency = freq_range_hz[0] <= spectral_centroid_hz <= freq_range_hz[1]

    # I'm still very unhappy with the frequency cue implementation - I want to find ways to see if I can capture how
    # the energy of a vocal changes over the frequency spectrum over time, rather than a static centroid. 

    # Decision
    if is_center_panned and is_vocal_frequency:
        vocal_components.append(k)
    else:
        instrumental_components.append(k)

print(f"Found {len(vocal_components)} vocal components and {len(instrumental_components)} instrumental components.")

# ------------------------

W_vocal = W[:, vocal_components]
H_left_vocal = H_left[vocal_components, :]
H_right_vocal = H_right[vocal_components, :]

W_instrumental = W[:, instrumental_components]
H_left_instrumental = H_left[instrumental_components, :]
H_right_instrumental = H_right[instrumental_components, :]

mag_left_vocal_recon = W_vocal @ H_left_vocal
mag_right_vocal_recon = W_vocal @ H_right_vocal
mag_left_instrumental_recon = W_instrumental @ H_left_instrumental
mag_right_instrumental_recon = W_instrumental @ H_right_instrumental

# I'm reconstructing instruments as well just for checking results to optimize our model. 

# creating soft masks (Wiener-style) - basically minimizing MSE
epsilon = 1e-9
total_mag_left = mag_left_vocal_recon + mag_left_instrumental_recon + epsilon
total_mag_right = mag_right_vocal_recon + mag_right_instrumental_recon + epsilon

vocal_mask_left = mag_left_vocal_recon / total_mag_left
vocal_mask_right = mag_right_vocal_recon / total_mag_right

stft_left_vocal = stft_left * vocal_mask_left
stft_right_vocal = stft_right * vocal_mask_right

y_left_vocal = librosa.istft(stft_left_vocal)
y_right_vocal = librosa.istft(stft_right_vocal)

# stereo output
y_vocal_stereo = np.vstack((y_left_vocal, y_right_vocal))

output_path = 'vocal_separated.wav'
sf.write(output_path, y_vocal_stereo.T, sr)

print(f"Saved to {output_path}")