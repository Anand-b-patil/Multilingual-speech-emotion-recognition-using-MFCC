# Multilingual Speech Emotion Recognition 🎙️🧠

## Project Overview

A **Machine Learning** and **Deep Learning** project designed to detect emotions from **speech audio** across multiple languages using advanced **audio signal processing** and **neural networks**.
This project also includes a **Flask web application** for real-time emotion detection and visualization.

---

## Features

✅ Multilingual support — English, Spanish, French, etc.
✅ Emotion detection: **Happy**, **Sad**, **Angry**, **Neutral**, and more
✅ Advanced **feature extraction** using `Librosa` (MFCCs, Chroma, Mel Spectrograms)
✅ Includes both **ML models** (SVM, Random Forest, XGBoost) and **DL models** (CNN, LSTM)
✅ Flask web interface for uploading and predicting emotions from speech
✅ Real-time visualizations — **Waveform** and **Spectrogram**

---

## Dataset

### **Toronto Emotional Speech Set (TESS)**

* Speech recordings from two female actors
* Simulated emotions: anger, happiness, sadness, disgust, fear, surprise, and neutral
* Used for multilingual and emotion classification experiments

---

## Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Anand-b-patil/Multilingual-speech-emotion-recognition-using-MFCC.git
cd Multilingual-speech-emotion-recognition-using-MFCC
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Flask App

```bash
python app.py
```

Then open your browser and go to:

```
http://127.0.0.1:5000/
```

---

## Feature Extraction

Feature extraction is performed using **Librosa**:

* **MFCC (Mel-Frequency Cepstral Coefficients)** — core feature for emotion detection
* **Chroma Frequencies** — captures harmonic content
* **Mel Spectrograms** — visual representation of frequency intensities

---

## Models Used

### 🔹 Machine Learning

* Support Vector Machine (SVM)
* Random Forest Classifier
* XGBoost

### 🔸 Deep Learning

* Convolutional Neural Network (CNN)
* Long Short-Term Memory (LSTM)

Flask app loads a trained **Keras (.keras)** deep learning model for live inference.

---

## Web Application

### 🔸 Home Page

Upload your audio file for emotion detection.

### 🔸 Results Page

Displays:

* Predicted emotion
* Audio duration & sample rate
* MFCC count
* **Waveform** and **Spectrogram** visualization


---

## Results

| Model         | Accuracy | Features Used          |
| ------------- | -------- | ---------------------- |
| SVM           | ~85%     | MFCC                   |
| Random Forest | ~88%     | MFCC + Chroma          |
| XGBoost       | ~90%     | Combined features      |
| CNN-LSTM      | **93%+** | MFCC + Mel Spectrogram |

---

## Tech Stack

| Tool / Library     | Version |
| ------------------ | ------- |
| Python             | 3.8+    |
| NumPy              | 1.24+   |
| Pandas             | 1.5+    |
| Librosa            | 0.10+   |
| scikit-learn       | 1.2+    |
| XGBoost            | 1.7+    |
| TensorFlow / Keras | 2.x     |
| Matplotlib         | 3.x     |
| Flask              | 2.x     |

---

## License

This project is licensed under the **MIT License**.
See the [`LICENSE`](LICENSE) file for more details.

---

## Acknowledgements

* [Toronto Emotional Speech Set (TESS)](https://tspace.library.utoronto.ca/handle/1807/24487)
* [Librosa](https://librosa.org/) — audio feature extraction
* [TensorFlow](https://www.tensorflow.org/) & [scikit-learn](https://scikit-learn.org/) — model training



