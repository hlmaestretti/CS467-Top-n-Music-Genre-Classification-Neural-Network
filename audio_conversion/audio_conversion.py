import os
import shutil
import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import skew, kurtosis, entropy, gmean
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from librosa.feature import spectral_bandwidth, zero_crossing_rate, rms
from pydub import AudioSegment
import tkinter as tk
from tkinter import filedialog


# Function to convert audio files to a supported format
def convert_audio(input_file, output_format):
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"File {input_file} does not exist.")

    file_name, _ = os.path.splitext(input_file)

    try:
        audio = AudioSegment.from_file(input_file)
    except Exception as e:
        raise ValueError(f"Could not load audio file {input_file}: {e}")

    output_file = f"{file_name}.{output_format}"

    try:
        with open(output_file, 'wb') as f:
            audio.export(f, format=output_format)
        print(f"File has been converted to {output_file}")
    except Exception as e:
        raise ValueError(
          f"Could not export audio file to {output_format}: {e}")

    return output_file


# Function to find all H5 files in a directory
def find_h5_files(directory):
    return [os.path.join(
      root, file) for root, _, files in os.walk(directory)
            for file in files if file.endswith('.h5')]


# Function to clean filenames
def clean_filename(filename):
    return os.path.basename(filename).strip()


# Function to safely get a value from an H5 file
def safe_get(f, key, default_value):
    try:
        value = f[key]
        if isinstance(value, h5py.Dataset):
            if value.shape == ():  # scalar dataset
                return value[()]
            else:
                return value[:]  # return as numpy array
        else:
            return value
    except (KeyError, ValueError):
        return default_value


# Function to extract features from an H5 file
def extract_features(f):
    features = {}
    songs = safe_get(f, 'analysis/songs', None)
    if songs is not None:
        if isinstance(songs, np.ndarray) and songs.dtype.names:
            features['duration'] = (songs['duration'][0] if 'duration' in
                                    songs.dtype.names else 0)
            features['tempo'] = (songs['tempo'][0] if 'tempo' in songs.dtype.
                                 names else 0)
            features['key'] = (int(songs['key'][0]) if 'key' in songs.dtype.
                               names else 0)
            features['mode'] = (int(songs['mode'][0]) if 'mode' in songs.
                                dtype.names else 0)
            features['time_signature'] = (int(songs['time_signature'][0]) if
                                          'time_signature' in songs.dtype.
                                          names else 4)
        else:
            features['duration'] = safe_get(f, 'analysis/songs/duration', 0)
            features['tempo'] = safe_get(f, 'analysis/songs/tempo', 0)
            features['key'] = int(safe_get(f, 'analysis/songs/key', 0))
            features['mode'] = int(safe_get(f, 'analysis/songs/mode', 0))
            features['time_signature'] = (int(safe_get(f, 'analysis/songs/\
time_signature', 4)))

        features['rhythmic_complexity'] = (features['tempo'] *
                                           features['time_signature'] / 120)

    segments_timbre = safe_get(f, 'analysis/segments_timbre', None)
    if segments_timbre is not None:
        timbre = np.array(segments_timbre)
        if timbre.ndim == 2 and timbre.shape[0] > 0:
            timbre_normalized = timbre / (np.sum(np.abs(timbre), axis=1,
                                                 keepdims=True) + 1e-10)
            features['spectral_centroid_mean'] = np.nanmean(timbre[:, 0])
            features['spectral_rolloff_mean'] = np.nanmean(timbre[:, 1])
            features['mfcc1'] = np.nanmean(timbre[:, 0])
            features['mfcc2'] = np.nanmean(timbre[:, 1])
            features['mfcc3'] = np.nanmean(timbre[:, 2])
            features['timbre_std'] = np.nanmean(np.nanstd(timbre, axis=0))
            features['timbre_skew'] = np.nanmean(skew(timbre, axis=0,
                                                      nan_policy='omit'))
            features['timbre_kurtosis'] = np.nanmean(kurtosis(
              timbre, axis=0, nan_policy='\
omit'))

            n_fft = min(2048, timbre.shape[0])
            features['spec_bw'] = np.nanmean(
              spectral_bandwidth(y=timbre.T, sr=22050, n_fft=n_fft)[0])
            features['zcr'] = np.nanmean(zero_crossing_rate(y=timbre.T)[0])
            features['rms'] = np.nanmean(rms(y=timbre.T)[0])

            features['timbre_max'] = np.nanmax(timbre, axis=0).mean()
            features['timbre_min'] = np.nanmin(timbre, axis=0).mean()
            features['timbre_range'] = (features['timbre_max'] -
                                        features['timbre_min'])
            features['timbre_median'] = np.nanmedian(timbre, axis=0).mean()
            features['timbre_variance'] = np.nanmean(
              np.nanvar(timbre, axis=0))
            features['timbre_mad'] = np.nanmean(np.abs(np.diff(timbre,
                                                               axis=0)))
            timbre_sum = np.nansum(np.nanmean(timbre, axis=1))
            features['timbre_temporal_centroid'] = (
              np.nanmean(np.arange(len(timbre))
                         * np.nanmean(timbre, axis=1)) /
                        (timbre_sum + 1e-10))

            features['timbre_q1'] = np.nanpercentile(
              timbre, 25, axis=0).mean()
            features['timbre_q3'] = np.nanpercentile(
              timbre, 75, axis=0).mean()
            features['timbre_iqr'] = features[
              'timbre_q3'] - features['timbre_q1']
            features['timbre_entropy'] = entropy(np.abs(timbre_normalized) +
                                                 1e-10, axis=0).mean()
            features['timbre_energy'] = np.nansum(timbre**2, axis=0).mean()
            features['timbre_flux'] = np.nanmean(np.diff(timbre, axis=0)**2)
            features['timbre_flatness'] = (
              gmean(np.abs(timbre) + 1e-10,
                    axis=0).mean() /
                   (np.nanmean(np.abs(timbre), axis=0).mean()
                    + 1e-10))

            rms_values = rms(y=timbre.T)[0]
            non_zero_rms_segments = np.sum(rms_values > 1e-10)
            features['num_non_zero_rms_segments'] = non_zero_rms_segments

            features['brightness'] = (
              np.nanmean(np.sum(timbre[:, 1:], axis=1) /
                         (np.sum(timbre, axis=1)
                          + 1e-10)))

    segments_loudness = safe_get(f, 'analysis/segments_loudness_max', None)
    if segments_loudness is not None:
        loudness = np.array(segments_loudness)
        features['loudness_mean'] = np.nanmean(loudness)
        features['loudness_std'] = np.nanstd(loudness)
        features['loudness_skew'] = skew(loudness, nan_policy='omit')
        features['loudness_kurtosis'] = kurtosis(loudness, nan_policy='omit')
        features['loudness_max'] = np.nanmax(loudness)
        features['loudness_min'] = np.nanmin(loudness)
        features['loudness_range'] = features[
          'loudness_max'] - features['loudness_min']
        features['loudness_median'] = np.nanmedian(loudness)

        features['loudness_q1'] = np.nanpercentile(loudness, 25)
        features['loudness_q3'] = np.nanpercentile(loudness, 75)
        features['loudness_iqr'] = features[
          'loudness_q3'] - features['loudness_q1']
        features['loudness_entropy'] = entropy(np.abs(loudness) + 1e-10)
        features['loudness_energy'] = np.nansum(loudness**2)
        features['loudness_flux'] = np.nanmean(np.diff(loudness)**2)

        features['roughness'] = np.nanmean(np.abs(np.diff(loudness)))

    segments_pitches = safe_get(f, 'analysis/segments_pitches', None)
    if segments_pitches is not None:
        pitches = np.array(segments_pitches)
        pitches = np.maximum(pitches, 0)
        features['chroma_mean'] = np.nanmean(pitches)
        features['chroma_std'] = np.nanstd(pitches)
        features['chroma_skew'] = np.nanmean(skew(pitches, axis=0,
                                                  nan_policy='omit'))
        features['chroma_kurtosis'] = np.nanmean(kurtosis(pitches, axis=0,
                                                          nan_policy='omit'))
        features['chroma_max'] = np.max(np.mean(pitches, axis=0))
        features['chroma_min'] = np.min(np.mean(pitches, axis=0))
        features['chroma_range'] = features[
          'chroma_max'] - features['chroma_min']
        features['chroma_median'] = np.nanmedian(pitches, axis=0).mean()

        pitch_diff = np.diff(pitches, axis=0)
        features['zcr'] = np.nanmean(
          np.sum(np.abs(np.sign(
            pitch_diff)), axis=1) / (2 * pitches.shape[1]))

        features['chroma_q1'] = np.nanpercentile(pitches, 25, axis=0).mean()
        features['chroma_q3'] = np.nanpercentile(pitches, 75, axis=0).mean()
        features['chroma_iqr'] = features['chroma_q3'] - features[
          'chroma_q1']
        features['chroma_entropy'] = entropy(pitches + 1e-10, axis=0).mean()
        features['chroma_energy'] = np.nansum(pitches**2, axis=0).mean()
        features['chroma_flux'] = np.nanmean(np.diff(pitches, axis=0)**2)
        features['chroma_flatness'] = (gmean(pitches + 1e-10, axis=0).mean() /
                                       (np.nanmean(
                                         pitches + 1e-10, axis=0).mean()))

        melodic_contour = np.argmax(pitches, axis=1)
        features['melodic_contour_direction'] = np.nanmean(
          np.diff(melodic_contour))
        features['melodic_contour_interval'] = np.nanmean(np.abs(
          np.diff(melodic_contour)))

        onset_env = np.sum(np.diff(pitches, axis=0) > 0, axis=1)
        peaks, _ = find_peaks(onset_env)
        if len(peaks) > 1:
            ioi = np.diff(peaks)
            features['rhythmic_entropy'] = entropy(ioi)
            features['rhythmic_irregularity'] = np.std(
              ioi) / np.mean(ioi)

    return features


# Function to process and copy H5 files
def process_and_copy_h5(input_files, output_dir, unified_df=None):
    os.makedirs(output_dir, exist_ok=True)
    data = []

    files_to_process = input_files if unified_df is None else unified_df[
      'filename']

    for filename in tqdm(files_to_process, desc="Processing files"):
        clean_name = clean_filename(filename)
        input_file = next(
          (f for f in input_files if clean_filename(f) == clean_name), None)
        if input_file:
            output_file = os.path.join(output_dir, clean_name)
            shutil.copy2(input_file, output_file)

            with h5py.File(output_file, 'r+') as f:
                genre = unified_df.loc[unified_df[
                  'filename'] == filename, 'genre'].iloc[
                    0] if unified_df is not None else f.attrs.get('genre')
                f.attrs['genre'] = genre

            with h5py.File(output_file, 'r') as f:
                summary = {
                    'filename': clean_name,
                    'genre': f.attrs['genre'],
                }
                features = extract_features(f)
                summary.update(features)

            data.append(summary)
        else:
            print(f"File not found: {filename}")

    return pd.DataFrame(data)


# Function to balance the dataset
def balance_dataset(df, n_samples=500):
    return df.groupby('genre').apply(
      lambda x: x.sample(
        n=min(len(x), n_samples), random_state=42)).reset_index(drop=True)


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    input_file = filedialog.askopenfilename(title="Select input audio file")
    if not input_file:
        print("No file selected.")
    else:
        supported_formats = ['wav', 'mp3', 'au', 'ogg', 'flac']
        output_format = input(
          "Enter desired output format (example: wav, mp3, au, ogg, flac):"
          ).strip().lower()

    if output_format not in supported_formats:
        supported = ', '.join(supported_formats)
        print(f"Unsupported format: {output_format}. "
              f"Supported formats are: {supported}")
    else:
        converted_file = convert_audio(input_file, output_format)
        print(f"Converted file: {converted_file}")

        # Load unified dataset to match filenames with millionsongsubset
        unified_df = pd.read_csv('unified_preprocessed_dataset.csv')

        unified_df['filename'] = unified_df['filename'].apply(clean_filename)
        unified_df['genre'] = unified_df['genre'].str.lower()

        genre_counts = unified_df['genre'].value_counts()
        genres_to_keep = genre_counts[genre_counts >= 70].index
        unified_df = unified_df[unified_df['genre'].isin(genres_to_keep)]

        gtzan_files = find_h5_files('./gtzan_msd_structure')
        msd_files = find_h5_files('./millionsongsubset')
        all_files = gtzan_files + msd_files

        combined_summary = process_and_copy_h5(
          all_files, './processed_h5', unified_df)
        balanced_summary = balance_dataset(combined_summary)

        numeric_columns = balanced_summary.select_dtypes(
          include=[np.number]).columns
        non_numeric_columns = balanced_summary.select_dtypes(
          exclude=[np.number]).columns

        balanced_summary[numeric_columns] = balanced_summary[
          numeric_columns].abs()
        balanced_summary[numeric_columns] = balanced_summary[
          numeric_columns].replace([np.inf, -np.inf], np.nan)
        balanced_summary[numeric_columns] = balanced_summary[
          numeric_columns].fillna(balanced_summary[
            numeric_columns].median())

        feature_columns = [
          col for col in balanced_summary.columns if col not in [
            'filename', 'genre', 'duration', 'key', 'mode', 'time_signature']]

        for column in feature_columns:
            if column in numeric_columns:
                balanced_summary[column] = balanced_summary[
                  column].clip(-1e15, 1e15)

        scaler = StandardScaler()
        balanced_summary[feature_columns] = scaler.fit_transform(
          balanced_summary[feature_columns])

        X = balanced_summary[[
          'duration', 'rhythmic_complexity'] + feature_columns + [
            'key', 'mode', 'time_signature']]

        print(f"Number of features: {X.shape[1]}")

        y = balanced_summary['genre']
        X_train, X_test, y_train, y_test = train_test_split(
          X, y, test_size=0.2, random_state=42, stratify=y)

        balanced_summary.to_csv(
          'processed_dataset_summary.csv', index=False)
        np.save('X_train.npy', X_train)
        np.save('X_test.npy', X_test)
        np.save('y_train.npy', y_train)
        np.save('y_test.npy', y_test)

        print("Processing complete. H5 files/summary saved: './processed_h5'.")
        print(f"Number of genres in the final dataset: "
              f"{balanced_summary['genre'].nunique()}")
        print(f"Genre distribution:\n{balanced_summary[
          'genre'].value_counts()}")
