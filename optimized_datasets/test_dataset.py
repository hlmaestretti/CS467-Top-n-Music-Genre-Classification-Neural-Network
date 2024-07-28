"""
THis file was made to test the optimized dataset's performance
"""

from nn_training import nn_training


if __name__ == "__main__":
    h5_folder = "./processed_h5"
    nn_training.train_nn(h5_folder)
