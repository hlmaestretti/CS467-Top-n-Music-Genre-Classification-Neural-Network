"""
This file contains the primary functions that allow the NN to guess what genres the inputted
song might be related to.
"""

from keras.models import load_model


def genre_guesser(model, h5_song_file):
    """
    The genre_guesser guesses what genres are related to the given song data.

    :param model:
    :param h5_song_file:
    :return: A dictionary with the percentile relationship to each genre in the NN
    """
    # Load model
    model = load_model(model)

    # extract features from h5
    song_features = h5_song_file

    # do model.predict
    prediction = model.predict(song_features)

    # read the vector, for now just assume it out
    return prediction
