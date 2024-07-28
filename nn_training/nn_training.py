"""
This file contains the contents necessary to train the neural network

This code was inspired by the wandb_example.py file which can be found in the following link:
https://wandb.ai/mostafaibrahim17/ml-articles/reports/An-Introduction-to-Audio-Classification-with-Keras--Vmlldzo0MDQzNDUy
"""


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from keras.callbacks import EarlyStopping
import pandas as pd
import joblib
from optimized_datasets import load_data


def train_nn(data_path):
    """
    The train_nn function takes a csv file containing the features and labels of a song
    and feeds it to a new neural network. The neural network is designed to be a 1D
    Convolution Neural Network.
    :param data_path: The csv file containing the labels and features of each song in the dataset.
    :return: None, but creates a .keras files that holds the nn
    """
    # Get the features and labels for the NN training
    features, labels = load_data.load_data(data_path)

    # Get the number of genres
    num_genres = labels.shape[1]

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(features,
                                                        labels,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        stratify=labels
                                                        )

    # Create and compile the NN
    input_shape = (x_train.shape[1], 1)
    model = Sequential()
    model.add(Conv1D(64, 3, padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Conv1D(128, 3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_genres, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Reshape the x data from array to Conv1D so that we can feed data into model
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    # Setting batch size and epoch
    batch_size = 32
    epochs = 20

    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_data=(x_test, y_test), verbose=1, callbacks=[early_stopping])

    # Save the training
    model.save('trained_model.keras')


if __name__ == "__main__":
    data_path = ("C:/Users/wwwhu/PycharmProjects/CS467-Top-n-Music-Genre-Classification-Neural-Network"
                 "/Datasets/unified_preprocessed_dataset.csv")
    train_nn(data_path)
