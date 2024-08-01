"""
This file contains the primary functions that allow the NN to guess what genres the inputted
song might be related to.
"""

from keras.models import load_model
from optimized_datasets import load_data, gtzan_to_h5
import h5py
import joblib
import numpy as np


def interpret_predictions(predictions, label_encoder):
    """
    Interpret model predictions to map them to genre labels.

    :param predictions: Array of model predictions (probability distributions).
    :param label_encoder: LabelEncoder used to decode genre labels.
    :return: Dictionary with genre labels and their corresponding probabilities.
    """
    genre_probabilities = predictions[0]

    # Get the genre labels from the label_encoder
    genre_labels = label_encoder.classes_

    # Create a dictionary to map genre labels to their probabilities
    genre_dict = {genre: prob for genre, prob in zip(genre_labels, genre_probabilities)}

    return genre_dict


def genre_guesser(model, file):
    """
    The genre_guesser guesses what genres are related to the given song data.

    :param model:
    :param h5_song_file:
    :return: A dictionary with the percentile relationship to each genre in the NN
    """
    # Load model
    model = load_model(model)

    # extract features from h5
    file_name = file.split('.')[:-1]
    h5_file_name = ".".join(file_name) + '.h5'

    gtzan_to_h5.convert_gtzan_to_msd_structure(file, h5_file_name)

    with h5py.File(h5_file_name, 'r') as f:
        try:
            feature = load_data.extract_features(f)
            feature = feature.reshape((1, 102, 1))
        except Exception as e:
            print(f"Error processing file {h5_file_name}: {str(e)}")

    prediction = model.predict(feature)
    # read the vector, for now just assume it out
    print(prediction)
    return prediction


if __name__ == '__main__':
    model = "../nn_training/trained_model.keras"
    input_file = ('C:/Users/wwwhu/PycharmProjects/CS467-Top-n-Music-Genre-Classification-Neural-Network'
                  '/nn_genre_guesser/country.00000.au')

    results = genre_guesser(model, input_file)

    # Load the LabelEncoder
    label_encoder = joblib.load('../nn_training/label_encoder.pkl')

    print(interpret_predictions(results, label_encoder))
