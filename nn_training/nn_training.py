"""
This file contains the contents necessary to train the neural network

This code was inspired by the wandb_example.py file which can be found in the following link:
https://wandb.ai/mostafaibrahim17/ml-articles/reports/An-Introduction-to-Audio-Classification-with-Keras--Vmlldzo0MDQzNDUy
"""

import joblib
import numpy as np
from sklearn.utils import class_weight
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, BatchNormalization
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.utils import to_categorical


def train_nn():
    """
    The train_nn function takes a csv file containing the features and labels of a song
    and feeds it to a new neural network. The neural network is designed to be a 1D
    Convolution Neural Network.

    :return: None, but creates a .keras files that holds the nn and a .pkl file to hold the unique labels
    """
    # Load training data
    x_train = np.load('./optimized_datasets/X_train.npy', allow_pickle=True)
    x_test = np.load('./optimized_datasets/X_test.npy', allow_pickle=True)
    y_train = np.load('./optimized_datasets/y_train.npy', allow_pickle=True)
    y_test = np.load('./optimized_datasets/y_test.npy', allow_pickle=True)

    # Initialize LabelEncoder and fit on y_train
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    joblib.dump(label_encoder, './nn_training/label_encoder.pkl')

    # Get unique class labels
    unique_labels = np.unique(y_train)
    num_genres = len(unique_labels)

    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, num_classes=num_genres)
    y_test = to_categorical(y_test, num_classes=num_genres)

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

    # Flatten and Dense Layers
    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.02)))
    model.add(Dropout(0.5))
    model.add(Dense(num_genres, activation='softmax'))

    optimizer = Adam(learning_rate=.0001)
    # optimizer = AdamW(learning_rate=0.001, weight_decay=0.01)
    # optimizer = Nadam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Reshape the x data from array to Conv1D so that we can feed data into model
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    # Setting batch size and epoch
    batch_size = 256
    epochs = 400

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.000005)
    model_checkpoint = ModelCheckpoint('./nn_training/best_model.keras', monitor='val_loss',
                                       save_best_only=True, save_weights_only=False, mode='min', verbose=1)

    # Compute class weights
    scaling_factor = 1
    class_weights = class_weight.compute_class_weight('balanced',
                                                      classes=np.unique(np.argmax(y_train, axis=1)),
                                                      y=np.argmax(y_train, axis=1))
    class_weights = dict(enumerate(class_weights))

    # Scale class weights by the scaling factor
    class_weights = {k: v * scaling_factor for k, v in class_weights.items()}

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_data=(x_test, y_test), verbose=1, callbacks=[early_stopping, reduce_lr, model_checkpoint],
              class_weight=class_weights)

    # Save the training
    model.save('./nn_training/trained_model.keras')
    model.summary()
    print(x_test.shape, x_train.shape)
    print(unique_labels)


if __name__ == "__main__":
    train_nn()
