from flask import Flask, request, render_template, jsonify
import os
import threading
import time
import shutil
from nn_genre_guesser.genre_guesser import genre_guesser, interpret_predictions
import joblib

app = Flask(__name__, template_folder='templates')

model_path = "./nn_training/trained_model.keras"
label_encoder_path = "./nn_training/label_encoder.pkl"
label_encoder = joblib.load(label_encoder_path)

if not os.path.exists('uploads'):
    os.makedirs('uploads')


def classify_genre(file_path):
    result = None
    stop_event = threading.Event()

    def process():
        nonlocal result
        result = genre_guesser(model_path, file_path)
        stop_event.set()

    processing_thread = threading.Thread(target=process)
    processing_thread.start()
    processing_thread.join()
    return result


def filter_predictions(genre_dict, threshold=0.0001):
    return {genre: confidence for genre, confidence in genre_dict.items() if confidence > threshold}


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file:
            if not os.path.exists('uploads'):
                os.makedirs('uploads')
            filename = file.filename
            file_path = os.path.join('uploads', filename)
            file.save(file_path)
            prediction = classify_genre(file_path)
            genre_dict = interpret_predictions(prediction, label_encoder)
            filtered_dict = filter_predictions(genre_dict)
            sorted_genres = sorted(filtered_dict.items(),
                                   key=lambda x: x[1], reverse=True)
            sorted_genres = [(genre, float(confidence))
                             for genre, confidence in sorted_genres]

            response = jsonify({'results': sorted_genres})

            # Remove the entire 'uploads' folder after processing
            shutil.rmtree('uploads', ignore_errors=True)

            return response
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
