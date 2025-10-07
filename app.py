from flask import Flask, render_template, request, abort
from werkzeug.utils import secure_filename
import os
import logging
from logging.handlers import RotatingFileHandler
import matplotlib
# Use non-interactive backend to avoid GUI warnings when running inside Flask
matplotlib.use('Agg')
import numpy as np
import librosa
import matplotlib.pyplot as plt

from src.predict import predict_emotion
from src.preprocessing import load_waveform


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB upload limit
ALLOWED_EXTENSIONS = {'wav'}

# Configure logging
os.makedirs('logs', exist_ok=True)
handler = RotatingFileHandler('logs/app.log', maxBytes=1000000, backupCount=3)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.errorhandler(413)
def request_entity_too_large(error):
    return render_template('result.html', error='File too large (max 5 MB).'), 413


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    if 'file' not in request.files:
        abort(400, 'No file part in the request')

    file = request.files['file']
    if file.filename == '':
        abort(400, 'No selected file')

    if not allowed_file(file.filename):
        abort(400, 'Only WAV files are allowed')

    filename = secure_filename(file.filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # Use the src package to predict
        result = predict_emotion(filepath)
        # Try to generate visualizations (waveform/spectrogram) for UI
        signal, sr = load_waveform(filepath)
        import librosa.display

        os.makedirs('static', exist_ok=True)
        waveform_path = os.path.join('static', 'waveform.png')
        plt.figure(figsize=(10, 3))
        librosa.display.waveshow(signal, sr=sr)
        plt.title('Waveform')
        plt.tight_layout()
        plt.savefig(waveform_path)
        plt.close()

        stft = np.abs(librosa.stft(signal))
        # use numpy's max as ref to compute reference value from the magnitude array
        db_stft = librosa.amplitude_to_db(stft, ref=np.max)
        spectrogram_path = os.path.join('static', 'spectrogram.png')
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(db_stft, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        plt.tight_layout()
        plt.savefig(spectrogram_path)
        plt.close()

        return render_template('result.html',
                               emotion=result.get('label'),
                               filename=filename,
                               duration=round(librosa.get_duration(y=signal, sr=sr), 2),
                               sample_rate=sr,
                               mfcc_count=13,
                               audio_url=f"/{app.config['UPLOAD_FOLDER']}/{filename}",
                               waveform_url='/' + waveform_path,
                               spectrogram_url='/' + spectrogram_path,
                               confidence=result.get('confidence'))
    except Exception as e:
        app.logger.exception('Error during prediction')
        return render_template('result.html', error='Internal server error during prediction'), 500


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('static', exist_ok=True)
    app.run(debug=True)
