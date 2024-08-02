"""
This file contains the contents necessary to train the neural network

This code was inspired by the wandb_example.py file which can be found in the following link:
https://wandb.ai/mostafaibrahim17/ml-articles/reports/An-Introduction-to-Audio-Classification-with-Keras--Vmlldzo0MDQzNDUy
"""


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from keras.models import Sequential
from keras.optimizers import Adagrad
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, BatchNormalization
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from optimized_datasets import load_data


def train_nn(h5_folder, dataset_file):
    """
    The train_nn function takes a csv file containing the features and labels of a song
    and feeds it to a new neural network. The neural network is designed to be a 1D
    Convolution Neural Network.
    :param h5_folder: The h5 folder containing each song in the dataset.
    :param dataset_file: csv file containing info of the desired dataset
    :return: None, but creates a .keras files that holds the nn
    """
    # Get the features and labels for the NN training
    features, labels = load_data.load_data(h5_folder, dataset_file)

    # Get the number of genres
    num_genres = labels.shape[1]

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(features,
                                                        labels,
                                                        test_size=0.2,
                                                        random_state=41,
                                                        stratify=np.argmax(labels, axis=1)
                                                        )

    # Create and compile the NN
    input_shape = (x_train.shape[1], 1)
    model = Sequential()

    # layer 1
    model.add(Conv1D(64, 3, padding='same', activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))

    # layer 2
    model.add(Conv1D(128, 3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))

    # layer 3
    model.add(Conv1D(256, 3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    # layer 4
    model.add(Conv1D(512, 3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    # Flatten and Dense Layers
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(num_genres, activation='softmax'))

    learning_rate = 0.02
    optimizer = Adagrad(learning_rate=learning_rate)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Reshape the x data from array to Conv1D so that we can feed data into model
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    # Setting batch size and epoch
    batch_size = 64
    epochs = 500

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

    # Compute class weights
    # Count the number of samples per class
    class_counts = np.bincount(np.argmax(y_train, axis=1))
    total_samples = len(y_train)

    # Compute inverse frequency weights
    class_weights = {i: (total_samples / count) ** 1.5 for i, count in enumerate(class_counts)}

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_data=(x_test, y_test), verbose=1, callbacks=[early_stopping, reduce_lr],
              class_weight=class_weights)

    # Save the training
    model.save('trained_model.keras')
    model.summary()


if __name__ == "__main__":
    data_path = ("C:/Users/wwwhu/PycharmProjects/CS467-Top-n-Music-Genre-Classification-Neural-Network/"
                 "optimized_datasets/processed_h5")
    dataset = ("C:/Users/wwwhu/PycharmProjects/CS467-Top-n-Music-Genre-Classification-Neural-Network/"
               "optimized_datasets/processed_dataset_summary.csv")
    train_nn(data_path, dataset)
