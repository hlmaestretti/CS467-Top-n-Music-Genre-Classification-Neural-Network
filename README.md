# CS467-Top-n-Music-Genre-Classification-Neural-Network

## General Overview

## Installation
#### Install the Python Packages
To install the python packages, use the following command once you have copied the repo:
```commandline
pip install -r requirements.txt
```

#### Installation Guide for FFmpeg
To run the audio conversion successfully, you need to have FFmpeg installed on your system. Below are the instructions for installing FFmpeg.

1. Download the latest FFmpeg release from the official website: [FFmpeg Downloads](https://ffmpeg.org/download.html)
2. Extract the downloaded ZIP file to a folder on your computer.
3. Add FFmpeg to your system PATH:
   - Open the Start Menu, search for "Environment Variables" and open it.
   - Under the "System variables" section, find the `Path` variable and select "Edit".
   - Click "New" and add the path to the `bin` folder inside the extracted FFmpeg directory (e.g., `C:\ffmpeg\bin`).
   - Click "OK" to close all windows.
4. Open a command prompt and type `ffmpeg -version` to verify the installation.

## Quick Start
To test the individual files, use the following structure to test various files:

- From the root repo directory use the following:
```py -m directory.file```
  - An example would be ```py -m nn_training.nn_training```


#### Using with the UI
1. Open https://genreclassifier.pythonanywhere.com/ in your browser.
2. Click the "Choose File" button to select an audio file from your computer. The supported file types are MP3, WAV, and OGG. Once you've selected a file, its name will appear under the button.
3. Click the "Classify" button to start the genre classification process. While the file is being processed, you'll see a loading animation.
4. After processing, the results will be displayed on the screen. You'll see a list of genres with their corresponding confidence scores, sorted from highest to lowest. These are insights into the musical characteristics of your file.
5. You can repeat the process with different audio files to classify multiple songs.

The app.py file serves as the backend for the web interface. The index.html file is the template for the web interface. Run py -m app from the root repo directory to run app.py locally.


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

### Dataset Creation and Optimization
The optimized_datasets folder contains the main files used when creating and optimizing our dataset.

The unify-tagtraum-top-magd.py file combines genre information from the Top-MAGD and Tagtraum datasets to create a unified genre-annotated dataset. The process_and_copy.py file combines the GTZAN and MSD H5 files into one dataset and optimizes it by balancing genre representation. The load_data.py file contains the function that loads the features and labels from the updated dataset.

#### unify-tagtraum-top-magd.py
The unify-tagtraum-top-magd.py file contains the functions to create a unified dataset from the Top-MAGD and Tagtraum datasets. It processes the Million Song Dataset (MSD) subset and matches track IDs with genre labels from both sources. The main function in this file reads the genre files, processes the MSD subset, and creates a CSV file with the unified dataset.

#### process_and_copy.py
The process_and_copy.py file contains everything needed to combine the GTZAN and MSD H5 files into one dataset. It also optimizes the dataset by making it more balanced using data augmentation and setting limits on the size of a genre set. The main function in this file processes H5 files, extracts features, and creates a balanced dataset for training.

#### load_data.py
The load_data.py file contains the function that loads the features and labels from the updated dataset. It includes data augmentation and feature selection processes. The load_data function in this file prepares the data for input into our neural network training pipeline.

#### gtzan_to_h5.py
The gtzan_to_h5.py file contains functions to convert GTZAN audio files to the MSD H5 file structure. It analyzes the MSD structure and creates H5 files from GTZAN audio files that match this structure. This file ensures compatibility between the GTZAN dataset and our MSD-based processing pipeline.

### User Interface
The ui folder contains the files used to create the user interface for our music genre classification system.

#### genre_classifier_cli.py
The genre_classifier_cli.py file contains the functions to create a command-line interface for our genre classification system. It allows users to input audio files and receive genre predictions. The main function in this file handles user input, processes audio files, and displays genre predictions using the trained neural network model.

### Audio Conversion
The audio conversion folder contains two sets of files. They aim to convert audio from one supported format to another with user inputs. It is aided by a testing suite to check its functionality. It contains an audio_conversion.py file and a tests.py file.

### audio_conversion.py
The audio_converion.py file is a script for converting audio files from one format to another, using the Pydub/FFmpeg libraries. The main function takes an input file and the desired output format, first verifying it exists, then loading its audio content. If the output format is 'h5', it converts the audio to an HDF5 file, storing audio data, sample rate, and channel information. On the other hand, if it is all other audio formats, it exports the audio. Afterward, the script uses the feature extraction function from optimized_dataset to extract features from the converted audio file.

### tests.py
The tests.py file is a set of unittest. They include creating a 20-second audio clip in mp3 converted to wav. It tests for successful audio conversion, if the file does not exist, correct file format, and supported format.
