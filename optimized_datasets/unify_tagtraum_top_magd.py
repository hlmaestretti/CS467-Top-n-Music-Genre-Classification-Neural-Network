"""
This file was used to create a genre-annotated dataset by combining information from the Top-MAGD and Tagtraum datasets.
It reads genre information from both datasets, processes the Million Song Dataset (MSD) subset, and creates a unified
dataset with genre annotations from both sources. This unified dataset serves as the basis for further processing and
analysis in the music genre classification project.
"""

import os
import pandas as pd


def read_genre_file(file_path):
    """
    This function reads a genre file and creates a dictionary mapping track IDs to genres.

    :param file_path: Path to the genre file
    :return: Dictionary of track IDs and their corresponding genres
    """
    genres = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            track_id = parts[0]
            genre = '_'.join(parts[1:])
            genres[track_id] = genre
    return genres


def process_msd_subset(msd_path, tagtraum_genres, topmagd_genres):
    """
    This function processes the MSD subset and creates a DataFrame with genre annotations.

    :param msd_path: Path to the MSD subset
    :param tagtraum_genres: Dictionary of Tagtraum genre annotations
    :param topmagd_genres: Dictionary of Top-MAGD genre annotations
    :return: DataFrame with processed data
    """
    data = []
    total_files = 0
    matched_files = 0

    for root, _, files in os.walk(msd_path):
        for file in files:
            if file.endswith('.h5'):
                total_files += 1
                track_id = file[:-3]
                tagtraum_genre = tagtraum_genres.get(track_id, '')
                topmagd_genre = topmagd_genres.get(track_id, '')

                if tagtraum_genre or topmagd_genre:
                    matched_files += 1
                    data.append({
                        'track_id': track_id,
                        'tagtraum_genre': tagtraum_genre,
                        'topmagd_genre': topmagd_genre,
                        'filename': file
                    })
                else:
                    print(f"Unmatched file: {file}")

    print(f"Total .h5 files in MSD subset: {total_files}")
    print(f"Matched files: {matched_files}")
    return pd.DataFrame(data)


def main():
    tagtraum_path = 'msd-tagtraum.txt'
    topmagd_path = 'msd-topmagd.txt'
    msd_subset_path = './millionsongsubset'

    tagtraum_genres = read_genre_file(tagtraum_path)
    topmagd_genres = read_genre_file(topmagd_path)

    print(f"Total entries in tagtraum: {len(tagtraum_genres)}")
    print(f"Total entries in top-magd: {len(topmagd_genres)}")

    unified_df = process_msd_subset(
        msd_subset_path, tagtraum_genres, topmagd_genres)

    unified_df.to_csv('unified_music_dataset.csv', index=False)

    print(f"Unified dataset created with {len(unified_df)} tracks.")
    print(unified_df.head())

    print("\nTagtraum genre distribution:")
    print(unified_df['tagtraum_genre'].value_counts())

    print("\nTop-MAGD genre distribution:")
    print(unified_df['topmagd_genre'].value_counts())

    unified_df['genres_match'] = unified_df['tagtraum_genre'] == unified_df['topmagd_genre']
    match_percentage = (
        unified_df['genres_match'].sum() / len(unified_df)) * 100

    print(f"\nPercentage of matching genres: {match_percentage:.2f}%")


if __name__ == "__main__":
    main()
