import os
import h5py
import librosa
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import ast


def convert_gtzan_to_h5(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for genre in os.listdir(input_dir):
        genre_path = os.path.join(input_dir, genre)
        if os.path.isdir(genre_path):
            for track in os.listdir(genre_path):
                if track.endswith('.au'):
                    track_path = os.path.join(genre_path, track)
                    audio, sr = librosa.load(track_path, sr=None)
                    output_file = os.path.join(output_dir, f"{genre}_{track[:-3]}.h5")
                    with h5py.File(output_file, 'w') as hf:
                        hf.create_dataset('audio', data=audio)
                        hf.create_dataset('sr', data=sr)
                        hf.attrs['genre'] = genre
    print("GTZAN conversion to .h5 format complete.")


# Convert GTZAN to .h5
convert_gtzan_to_h5('./genres', './gtzan_h5')

# Load the extracted features
gtzan_df = pd.read_csv('gtzan_features.csv')
msd_df = pd.read_csv('msd_features.csv')

# Update GTZAN filenames to .h5 and convert genres to lowercase
gtzan_df['filename'] = gtzan_df['filename'].apply(lambda x: x[:-3] + '.h5')
gtzan_df['genre'] = gtzan_df['genre'].str.lower()
msd_df['genre'] = msd_df['genre'].str.lower()

# Combine the datasets
combined_df = pd.concat([gtzan_df, msd_df], ignore_index=True)

# Get the most common genre
most_common_genre = combined_df['genre'].value_counts().index[0]
print(f"Most common genre: {most_common_genre}")

# Limit the most common genre to 500 songs
common_genre_df = combined_df[combined_df['genre'] == most_common_genre].sample(
    n=min(500, combined_df['genre'].value_counts()[most_common_genre]), random_state=42
)
other_genres_df = combined_df[combined_df['genre'] != most_common_genre]
combined_df = pd.concat([common_genre_df, other_genres_df], ignore_index=True)

# Remove genres with fewer than 70 occurrences
genre_counts = combined_df['genre'].value_counts()
genres_to_keep = genre_counts[genre_counts >= 70].index
combined_df = combined_df[combined_df['genre'].isin(genres_to_keep)]


def safe_float_convert(x):
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list):
                return parsed if len(parsed) > 1 else parsed[0]
            return float(parsed)
        except ValueError:
            return np.nan
    elif isinstance(x, list):
        return x if len(x) > 1 else x[0]
    elif isinstance(x, (int, float)):
        return float(x)
    else:
        return np.nan


# Apply safe conversion to all columns except 'filename' and 'genre'
for col in combined_df.columns:
    if col not in ['filename', 'genre']:
        combined_df[col] = combined_df[col].apply(safe_float_convert)

# Ensure tempo is positive and within a reasonable range
combined_df['tempo'] = combined_df['tempo'].apply(lambda x: abs(x) if isinstance(x, (int, float)) else np.nan)
combined_df['tempo'] = combined_df['tempo'].clip(20, 300)  # Clip tempo between 20 and 300 BPM

# Handle missing values
imputer = SimpleImputer(strategy='median')
columns_to_impute = combined_df.select_dtypes(include=[np.number]).columns
combined_df[columns_to_impute] = imputer.fit_transform(combined_df[columns_to_impute])

# Ensure duration is positive
if 'duration' in combined_df.columns:
    combined_df['duration'] = np.abs(combined_df['duration'])

# Normalize numeric features
features_to_normalize = [
    'spectral_centroid_mean', 'spectral_rolloff_mean', 'tempo',
    'mfcc1', 'mfcc2', 'mfcc3', 'loudness_mean'
]
scaler = MinMaxScaler()
combined_df[features_to_normalize] = scaler.fit_transform(combined_df[features_to_normalize])

# Select final features
final_features = [
    'filename', 'genre', 'duration', 'spectral_centroid_mean', 'spectral_rolloff_mean',
    'tempo', 'mfcc1', 'mfcc2', 'mfcc3', 'loudness_mean',
    'key', 'mode', 'time_signature', 'chroma_mean'
]
final_features = [col for col in final_features if col in combined_df.columns]

unified_df = combined_df[final_features]

# Sort the unified dataset by genre
unified_df = unified_df.sort_values('genre')

# Save the unified dataset
unified_df.to_csv('unified_preprocessed_dataset.csv', index=False)

print("GTZAN conversion and feature preprocessing complete. Unified dataset saved as 'unified_preprocessed_dataset.csv'.")
print(f"Number of genres in the final dataset: {unified_df['genre'].nunique()}")
print(f"Genre distribution:\n{unified_df['genre'].value_counts()}")
for col in unified_df.columns:
    if col not in ['filename', 'genre', 'chroma_mean']:
        print(f"{col} range: {unified_df[col].min()} to {unified_df[col].max()}")
print(f"Columns in the final dataset: {', '.join(unified_df.columns)}")
print("\nSample of the first few rows:")
print(unified_df.head().to_string())
