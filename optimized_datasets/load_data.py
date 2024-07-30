"""
This file contains the function that loads the features and labels from the updated dataset
"""


import os
import h5py
import numpy as np
import librosa
from scipy.stats import skew, kurtosis, entropy, gmean
from scipy.signal import find_peaks
from librosa.feature import (
    spectral_centroid, spectral_bandwidth, spectral_rolloff, zero_crossing_rate
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.utils import to_categorical
import joblib


def load_data(h5_folder):
    """
    Load and process data from H5 files in the specified folder.

    :param h5_folder: Path to the folder containing H5 files
    :return: Tuple of scaled features and one-hot encoded labels
    """
    features = []
    labels = []
    feature_length = 102

    for filename in os.listdir(h5_folder):
        if filename.endswith('.h5'):
            file_path = os.path.join(h5_folder, filename)
            with h5py.File(file_path, 'r') as f:
                try:
                    # Extract basic song information
                    songs = f['analysis/songs'][()]
                    duration = songs['duration'][0] if 'duration' in songs.dtype.names else 0
                    tempo = songs['tempo'][0] if 'tempo' in songs.dtype.names else 0
                    key = songs['key'][0] if 'key' in songs.dtype.names else 0
                    mode = songs['mode'][0] if 'mode' in songs.dtype.names else 0
                    time_signature = songs['time_signature'][0] if 'time_signature' in songs.dtype.names else 0

                    rhythmic_complexity = (tempo * time_signature) / 120

                    # Extract timbre, pitches, and loudness data
                    timbre = (np.array(f['analysis/segments_timbre'][()])
                              if 'analysis/segments_timbre' in f else np.zeros((1, 12)))
                    pitches = (np.array(f['analysis/segments_pitches'][()])
                               if 'analysis/segments_pitches' in f else np.zeros((1, 12)))
                    loudness = (np.array(f['analysis/segments_loudness_max'][()])
                                if 'analysis/segments_loudness_max' in f else np.zeros(1))

                    # Enhanced feature extraction
                    timbre_mean = np.mean(timbre, axis=0)
                    timbre_std = np.std(timbre, axis=0)
                    timbre_skew = skew(timbre, axis=0)
                    timbre_kurtosis = kurtosis(timbre, axis=0)
                    timbre_max = np.max(timbre, axis=0)
                    timbre_min = np.min(timbre, axis=0)
                    timbre_range = timbre_max - timbre_min
                    timbre_median = np.median(timbre, axis=0)
                    timbre_variance = np.var(timbre, axis=0)
                    timbre_q1 = np.percentile(timbre, 25, axis=0)
                    timbre_q3 = np.percentile(timbre, 75, axis=0)
                    timbre_iqr = timbre_q3 - timbre_q1
                    timbre_entropy = entropy(np.abs(timbre) + 1e-10, axis=0)
                    timbre_energy = np.sum(timbre**2, axis=0)
                    timbre_flux = np.mean(np.diff(timbre, axis=0)**2)
                    timbre_flatness = gmean(np.abs(timbre) + 1e-10, axis=0) / (np.mean(np.abs(timbre), axis=0) + 1e-10)

                    pitches_mean = np.mean(pitches, axis=0)
                    pitches_std = np.std(pitches, axis=0)
                    pitches_skew = skew(pitches, axis=0)
                    pitches_kurtosis = kurtosis(pitches, axis=0)
                    pitches_max = np.max(np.where(np.isnan(pitches), -np.inf, pitches), axis=0)
                    pitches_min = np.min(pitches, axis=0)
                    pitches_range = pitches_max - pitches_min
                    pitches_median = np.median(pitches, axis=0)
                    pitches_q1 = np.percentile(pitches, 25, axis=0)
                    pitches_q3 = np.percentile(pitches, 75, axis=0)
                    pitches_iqr = pitches_q3 - pitches_q1
                    pitches_entropy = entropy(pitches + 1e-10, axis=0)
                    pitches_energy = np.sum(pitches**2, axis=0)
                    pitches_flux = np.mean(np.diff(pitches, axis=0)**2)
                    pitches_flatness = gmean(pitches + 1e-10, axis=0) / (np.mean(pitches + 1e-10, axis=0))

                    loudness_mean = np.mean(loudness)
                    loudness_std = np.std(loudness)
                    loudness_skew = skew(loudness)
                    loudness_kurtosis = kurtosis(loudness)
                    loudness_max = np.max(loudness)
                    loudness_min = np.min(loudness)
                    loudness_range = loudness_max - loudness_min
                    loudness_median = np.median(loudness)
                    loudness_q1 = np.percentile(loudness, 25)
                    loudness_q3 = np.percentile(loudness, 75)
                    loudness_iqr = loudness_q3 - loudness_q1
                    loudness_entropy = entropy(np.abs(loudness) + 1e-10)
                    loudness_energy = np.sum(loudness**2)
                    loudness_flux = np.mean(np.diff(loudness)**2)

                    # Adjust n_fft based on the input signal length
                    n_fft = min(2048, timbre.shape[0])

                    # Additional spectral features with adjusted n_fft
                    spec_cent = np.mean(spectral_centroid(y=timbre.T, sr=22050, n_fft=n_fft)[0])
                    spec_bw = np.mean(spectral_bandwidth(y=timbre.T, sr=22050, n_fft=n_fft)[0])
                    spec_rolloff = np.mean(spectral_rolloff(y=timbre.T, sr=22050, n_fft=n_fft)[0])
                    zcr = np.mean(zero_crossing_rate(y=timbre.T)[0])

                    # Compute RMS - Root Mean Square
                    rms_mean = np.mean(np.sqrt(np.mean(timbre**2, axis=0)))

                    # Compute RMS for each segment using librosa
                    rms_values = librosa.feature.rms(y=timbre.T)[0]
                    num_non_zero_rms_segments = np.sum(rms_values > 1e-10)

                    # Brightness and roughness
                    brightness = np.nanmean(np.sum(timbre[:, 1:], axis=1) / (np.sum(timbre, axis=1) + 1e-10))
                    roughness = np.nanmean(np.abs(np.diff(loudness)))

                    # Melodic features
                    melodic_contour = np.argmax(pitches, axis=1)
                    melodic_contour_direction = np.nanmean(np.diff(melodic_contour))
                    melodic_contour_interval = np.nanmean(np.abs(np.diff(melodic_contour)))

                    # Rhythmic complexity features
                    onset_env = np.sum(np.diff(pitches, axis=0) > 0, axis=1)
                    peaks, _ = find_peaks(onset_env)
                    if len(peaks) > 1:
                        ioi = np.diff(peaks)
                        rhythmic_entropy = entropy(ioi)
                        rhythmic_irregularity = np.std(ioi) / np.mean(ioi)
                    else:
                        rhythmic_entropy = 0
                        rhythmic_irregularity = 0

                    # Concatenate all features
                    feature = np.concatenate([
                        np.atleast_1d(duration), np.atleast_1d(tempo), np.atleast_1d(key),
                        np.atleast_1d(mode), np.atleast_1d(time_signature),
                        np.atleast_1d(rhythmic_complexity),
                        np.atleast_1d(loudness_mean), np.atleast_1d(loudness_std),
                        np.atleast_1d(loudness_skew), np.atleast_1d(loudness_kurtosis),
                        np.atleast_1d(loudness_max), np.atleast_1d(loudness_min),
                        np.atleast_1d(loudness_range), np.atleast_1d(loudness_median),
                        np.atleast_1d(loudness_q1), np.atleast_1d(loudness_q3),
                        np.atleast_1d(loudness_iqr), np.atleast_1d(loudness_entropy),
                        np.atleast_1d(loudness_energy), np.atleast_1d(loudness_flux),
                        np.atleast_1d(timbre_mean), np.atleast_1d(timbre_std),
                        np.atleast_1d(timbre_skew), np.atleast_1d(timbre_kurtosis),
                        np.atleast_1d(timbre_max), np.atleast_1d(timbre_min),
                        np.atleast_1d(timbre_range), np.atleast_1d(timbre_median),
                        np.atleast_1d(timbre_variance),
                        np.atleast_1d(timbre_q1), np.atleast_1d(timbre_q3),
                        np.atleast_1d(timbre_iqr), np.atleast_1d(timbre_entropy),
                        np.atleast_1d(timbre_energy), np.atleast_1d(timbre_flux),
                        np.atleast_1d(timbre_flatness),
                        np.atleast_1d(pitches_mean), np.atleast_1d(pitches_std),
                        np.atleast_1d(pitches_skew), np.atleast_1d(pitches_kurtosis),
                        np.atleast_1d(pitches_max), np.atleast_1d(pitches_min),
                        np.atleast_1d(pitches_range), np.atleast_1d(pitches_median),
                        np.atleast_1d(pitches_q1), np.atleast_1d(pitches_q3),
                        np.atleast_1d(pitches_iqr), np.atleast_1d(pitches_entropy),
                        np.atleast_1d(pitches_energy), np.atleast_1d(pitches_flux),
                        np.atleast_1d(pitches_flatness),
                        np.atleast_1d(spec_cent), np.atleast_1d(spec_bw),
                        np.atleast_1d(spec_rolloff), np.atleast_1d(zcr),
                        np.atleast_1d(rms_mean), np.atleast_1d(num_non_zero_rms_segments),
                        np.atleast_1d(brightness), np.atleast_1d(roughness),
                        np.atleast_1d(melodic_contour_direction),
                        np.atleast_1d(melodic_contour_interval),
                        np.atleast_1d(rhythmic_entropy), np.atleast_1d(rhythmic_irregularity)
                    ])

                    # Ensure feature vector has consistent length
                    if len(feature) < feature_length:
                        feature = np.pad(feature, (0, feature_length - len(feature)))
                    elif len(feature) > feature_length:
                        feature = feature[:feature_length]

                    features.append(feature)
                    labels.append(f.attrs['genre'])
                except Exception as e:
                    print(f"Error processing file {filename}: {str(e)}")

    features = np.array(features)
    labels = np.array(labels)

    # Encode labels and scale features
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    labels_onehot = to_categorical(labels_encoded)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # save the labels for future use
    joblib.dump(label_encoder, 'label_encoder.pkl')

    return features_scaled, labels_onehot
