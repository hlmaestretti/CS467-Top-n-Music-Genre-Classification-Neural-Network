# CS467-Top-n-Music-Genre-Classification-Neural-Network

## General Overview

## Quick Start
(Only if we plan to make this easier usable to other people)
## Explanation of the Functions
### Neural Network Training
The nn_training folder contains the main files used when creating a neural network.
The nn_training.py file contains the functions to create a trained model and uses various
optimization techniques to enhance the models. The .keras and .pkl files are the outputs
of the trained neural network and are used in the genre guesser to make predictions and
interpret them. 
#### nn_training.py
The nn_training.py file contains the bulk of the nn_training work. It creates a 1D
convolution neural network that uses the following parameters:
- Activation Function: Relu
- Loss Function: categorical_crossentropy
- Optimization Model: Adam
- Layers: 4 (3 Convolutional layers and 1 flatten/dense layer)

It utilizes various callback features and class weighting to optimize the system and 
are: 
- Early Stopping - Stops once value losses have plateau/increased for too long
- Reducing LR on plateau - Lowers the learning rate when the validation loss plateaus to minimize overfitting
- Model Checkpoints - Saves a copy of the model each time the validation loss hits a new minimum
- Class Weighting: Took an augmented balanced approach to make it so our imbalanced dataset learns more from correctly
    guessing underrepresented genres than it does overrepresented ones.
#### .keras and .pkl files
These files are the outputs of nn_training. The .keras file represents the trained model 
that can then be loaded into other programs to act as the neural network. The .pkl 
file is a downloaded form of the unique labels found when training the neural network. 
These are used to help other programs identify what genres are being considered in the 
neural network.

### Prediction Program
The predict folder contains the files used to create the genre guessing functionality of
our system. The genre_guess.py file contains the bulk of the content for the genre guessing and
the h5_verify.py file was used to verify that various components leading up to this worked correctly.
#### genre_guesser.py
The genre_guesser.py file contains two functions: genre_guesser and interpret_predictions.
The genre_guesser function feeds a song into the trained neural network and guesses what genre it may be.
It does this by outputting a list of numbers that add up to one. Each number represents how related are song is
to a particular genre. The interpret_prediction function then receives this info and adds genre labels to it, making
it easier to parse.
#### h5_verify.py
The h5_verify.py file contains a function that can print out the structure of an h5 file.


