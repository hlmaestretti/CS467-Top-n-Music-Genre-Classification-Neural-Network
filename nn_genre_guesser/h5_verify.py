"""
This file contains functions that can be used to check and verify the contents of an h5 file
"""

import h5py


def print_h5_structure(file_path):
    """
    The print_h5_structure function can print the structure of most h5 files

    :param file_path: h5 file
    :return: None, but prints the structure of the h5 file
    """
    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"Group: {name}")

    with h5py.File(file_path, 'r') as h5file:
        print(h5file.attrs['genre'])
        h5file.visititems(print_structure)


if __name__ == "__main__":
    # Example usage
    h5_file_path = "../nn_genre_guesser/blues.00000.h5"

    print_h5_structure(h5_file_path)
    # print(extract_mfcc_features_from_simple_h5(h5_file_path))
