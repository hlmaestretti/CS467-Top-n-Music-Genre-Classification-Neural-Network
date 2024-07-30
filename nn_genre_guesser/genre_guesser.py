"""
This file contains the primary functions that allow the NN to guess what genres the inputted
song might be related to.
"""

from keras.models import load_model


def extract_features(file):
    """
    The convert_file_to h5 receives an audio file and then extracts various features from it so that it can be passed
    through the neural network.

    :param file: An audio file. Supported files include [list here]
    :return: The extracted features in the same format as Corey's work
    """
    return file


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
    song_features = extract_features(file)

    # do model.predict
    prediction = model.predict(song_features)

    # read the vector, for now just assume it out
    return prediction
