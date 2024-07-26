import os
import h5py
import librosa


def convert_gtzan_to_h5(input_dir, output_dir):
    """
    The convert_gtzan_to_h5 file converts the downloaded files from gtzan into a single h5 files. It follows the path
    of the downloaded folders and copies it over to the output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    for genre in os.listdir(input_dir):
        genre_path = os.path.join(input_dir, genre)
        if os.path.isdir(genre_path):
            for track in os.listdir(genre_path):
                if track.endswith('.au'):
                    track_path = os.path.join(genre_path, track)
                    audio, sr = librosa.load(track_path, sr=None)
                    output_file = os.path.join(output_dir, f"{track[:-3]}.h5")
                    with h5py.File(output_file, 'w') as hf:
                        hf.create_dataset('audio', data=audio)
                        hf.create_dataset('sr', data=sr)
                        hf.attrs['genre'] = genre
    print("GTZAN conversion to .h5 format complete.")


# Convert GTZAN to .h5
# convert_gtzan_to_h5('./genres', './gtzan_h5')


if __name__ == '__main__':
    ans = input("Are you sure you want to do this? (y/n)")
    if ans == 'y':
        inp = ('C:/Users/wwwhu/PycharmProjects/CS467-Top-n-Music-Genre-Classification-Neural-Network/Github-ignore'
               '/unified dataset/gtzan-input')
        out = ('C:/Users/wwwhu/PycharmProjects/CS467-Top-n-Music-Genre-Classification-Neural-Network/Github-ignore'
               '/unified dataset/gtzan-output')
        convert_gtzan_to_h5(inp, out)
