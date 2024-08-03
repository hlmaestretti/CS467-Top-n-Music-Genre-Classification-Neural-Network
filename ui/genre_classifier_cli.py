"""
This file contains the primary functions for the Top-n Music Genre Classification Neural Network UI.
It includes functionality for genre classification, user interaction, and result display.
"""

from nn_genre_guesser.genre_guesser import genre_guesser, interpret_predictions
import os
import sys
import logging
import threading
import time
import joblib

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
class NullWriter:
    def write(self, s):
        pass
sys.stderr = NullWriter()
logging.getLogger('tensorflow').disabled = True


def classify_genre(model_path: str, file_path: str):
    """
    Classifies the genre of a given audio file using a pre-trained model.

    :param model_path: Path to the trained model file.
    :param file_path: Path to the audio file to be classified.
    :return: Predicted genre probabilities.
    """
    result = None
    stop_event = threading.Event()

    def process():
        nonlocal result
        result = genre_guesser(model_path, file_path)
        stop_event.set()

    processing_thread = threading.Thread(target=process)
    processing_thread.start()

    animate_processing(stop_event)

    processing_thread.join()
    return result


def animate_processing(stop_event):
    """
    Displays an animated processing indicator while classification is ongoing.

    :param stop_event: Event to signal when processing is complete.
    """
    chars = "/-\\|"
    i = 0
    while not stop_event.is_set():
        sys.stdout.write(f"\rProcessing... {chars[i % len(chars)]}")
        sys.stdout.flush()
        time.sleep(0.1)
        i += 1
    sys.stdout.write("\r" + " " * 20 + "\r")  # Clear the processing message
    sys.stdout.flush()


def get_file_path() -> str:
    """
    Prompts the user to input a valid file path for audio classification.

    :return: Valid file path entered by the user.
    """
    while True:
        file_path = input(
            "Please input the file you would like to have tested: ")
        if os.path.isfile(file_path):
            return file_path
        else:
            print("Error: File not found. Please enter a valid file path.")


def get_user_choice() -> bool:
    """
    Prompts the user to decide whether to classify another song or quit.

    :return: True if the user wants to classify another song, False otherwise.
    """
    while True:
        choice = input(
            "\nClassify another song? (Y to restart / N to quit): ").lower()
        if choice in ['y', 'yes']:
            return True
        elif choice in ['n', 'no']:
            return False
        else:
            print("Invalid input. Please enter 'Y' or 'N'.")


def filter_predictions(genre_dict, threshold=0.0001):
    """
    Filters out genre predictions below a certain confidence threshold.

    :param genre_dict: Dictionary of genre predictions and their confidences.
    :param threshold: Minimum confidence threshold for inclusion.
    :return: Filtered dictionary of genre predictions.
    """
    return {genre: confidence for genre, confidence in genre_dict.items() if confidence > threshold}


def display_results(prediction, label_encoder):
    """
    Displays the genre classification results.

    :param prediction: Raw prediction output from the model.
    :param label_encoder: Encoder used for genre labels.
    """
    genre_dict = interpret_predictions(prediction, label_encoder)
    filtered_dict = filter_predictions(genre_dict)
    print("\nResults:")
    sorted_genres = sorted(filtered_dict.items(),
                           key=lambda x: x[1], reverse=True)
    for genre, confidence in sorted_genres:
        print(f"{genre}: {confidence:.2%}")


def main():
    """
    Main function to run the music genre classification program.
    """
    model_path = "./nn_training/trained_model.keras"
    label_encoder_path = "./nn_training/label_encoder.pkl"

    label_encoder = joblib.load(label_encoder_path)

    while True:
        print("\nWelcome to Top-n Music Genre Classification Neural Network")

        file_path = get_file_path()

        try:
            prediction = classify_genre(model_path, file_path)
            print("Analysis complete!")
            time.sleep(0.5)
            display_results(prediction, label_encoder)
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")

        if not get_user_choice():
            break

    print("\nThank you for using the Top-n Music Genre Classification Neural Network. Goodbye!")


if __name__ == "__main__":
    main()
