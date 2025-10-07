from flask import Flask, render_template, request
import librosa
import librosa.display
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
model = tf.keras.models.load_model('models/emotion_model.keras')
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Load audio
    signal, sr = librosa.load(filepath, sr=22050, duration=3, offset=0.5)
    duration = librosa.get_duration(y=signal, sr=sr)
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    mfccs = mfccs.T  # Transpose to (time, features)
    features = np.expand_dims(mfccs, axis=0)  # shape (1, time, 13)

    # Prediction
    prediction = model.predict(features)
    emotion_index = np.argmax(prediction)
    if emotion_index >= len(emotion_labels):
        emotion = "Unknown"
    else:
        emotion = emotion_labels[emotion_index]

    # Generate waveform
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(signal, sr=sr)
    plt.title('Waveform')
    plt.tight_layout()
    waveform_path = os.path.join('static', 'waveform.png')
    plt.savefig(waveform_path)
    plt.close()

    # Generate spectrogram
    stft = np.abs(librosa.stft(signal))
    db_stft = librosa.amplitude_to_db(stft, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(db_stft, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()
    spectrogram_path = os.path.join('static', 'spectrogram.png')
    plt.savefig(spectrogram_path)
    plt.close()

    return render_template('result.html',
                           emotion=emotion,
                           filename=filename,
                           duration=round(duration, 2),
                           sample_rate=sr,
                           mfcc_count=mfccs.shape[0],
                           audio_url=f"/{app.config['UPLOAD_FOLDER']}/{filename}",
                           waveform_url='/' + waveform_path,
                           spectrogram_url='/' + spectrogram_path)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('static', exist_ok=True)
    app.run(debug=True)
