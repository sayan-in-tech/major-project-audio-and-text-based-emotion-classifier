# 🎧 NeuroTone: The Vocal Emotions Classifier

## Overview

The **Vocal Emotions Classifier** is a machine learning-powered application that detects emotions from speech audio files. By leveraging advanced audio processing techniques and deep learning, this project identifies emotions such as happiness, sadness, anger, and more from uploaded audio files. The application is built using **Streamlit** for the user interface, **TensorFlow** for the deep learning model, and **Librosa** for audio feature extraction.

## Features

* 🎤  **Audio File Support** : Upload audio files in various formats (e.g., `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.aac`).
* 🔄  **Automatic Conversion** : Non-WAV files are automatically converted to WAV format using [pydub](vscode-file://vscode-app/c:/Users/sayan/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html).
* 🎯  **Emotion Detection** : Predicts emotions such as `happy`, `sad`, `angry`, `neutral`, and more.
* 📈  **Confidence Score** : Displays the confidence level of the predicted emotion.
* 🛠  **Real-Time Feedback** : Progress bars and status updates during audio processing.
* 🎉  **Interactive UI** : Built with Streamlit for a seamless and user-friendly experience.

## How It Works

1. **Audio Upload** : Users upload an audio file via the Streamlit interface.
2. **Preprocessing** :

* If the file is not in WAV format, it is converted using [pydub](vscode-file://vscode-app/c:/Users/sayan/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html).
* Audio features (MFCCs) are extracted using [librosa](vscode-file://vscode-app/c:/Users/sayan/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html).
* Delta and delta² features are computed to enhance the feature set.

1. **Prediction** :

* The pre-trained TensorFlow model processes the features.
* The model outputs the predicted emotion and its confidence score.

1. **Results Display** :

* The detected emotion and confidence score are displayed.
* Users can listen to the uploaded audio directly in the app.

## Installation

### Prerequisites

* Python 3.8 or higher
* Virtual environment (recommended)

### Steps

1. Clone the repository:

   **git** **clone** **https://github.com/sayan-in-tech/vocal-emotions-cl**assifier.git

   **cd** **vocal-emotions-classifier**
2. Create and activate a virtual environment:

   **python** **-m** **venv** **venv**

   **venv\Scripts\activate**  **# On Windows**

   **source** **venv/bin/activate**  **# On macOS/Linux**
3. Install dependencies:

   **pip** **install** **-r** **requirements.txt**
4. Run the application:

   **streamlit** **run** **app.py**
5. Open the application in your browser at `http://localhost:8501`.

## File Structure

**vocal-emotions-classifier/**

**├── app.py                     # Main Streamlit **application

**├── audio_formatter.py         # Audio conversion **utility

**├── models/**

**│   ├── emotion_recognition_model.keras  # **Pre-trained TensorFlow model

**│   ├── scaler_params.npy                # Scaler **parameters for feature normalization

**│   └── label_encoder.pkl                # Label **encoder for emotion mapping

**├── requirements.txt           # Python **dependencies

**├── .gitignore                 # Git ignore file**

**└── README.md                  # Project **documentation

## Key Components

### 1. **`app.py`**

The main application file that:

* Handles file uploads.
* Preprocesses audio files.
* Extracts features and predicts emotions using the pre-trained model.
* Displays results in an interactive UI.

### 2. **[audio_formatter.py](vscode-file://vscode-app/c:/Users/sayan/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html)**

A utility script for converting audio files to WAV format using [pydub](vscode-file://vscode-app/c:/Users/sayan/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html).

### 3. **Models**

* **`emotion_recognition_model.keras`** : A TensorFlow model trained on multiple datasets (e.g., RAVDESS, CREMA-D, TESS, SAVEE).
* **[scaler_params.npy](vscode-file://vscode-app/c:/Users/sayan/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html)** : Contains mean and scale values for feature normalization.
* **`label_encoder.pkl`** : Maps numerical predictions to emotion labels.

## Datasets

The model was trained on the following datasets:

* **RAVDESS** : Ryerson Audio-Visual Database of Emotional Speech and Song.
* **CREMA-D** : Crowd-sourced Emotional Multimodal Actors Dataset.
* **TESS** : Toronto Emotional Speech Set.
* **SAVEE** : Surrey Audio-Visual Expressed Emotion.

## Technologies Used

* **Frontend** : Streamlit
* **Backend** : TensorFlow, Librosa, Pydub
* **Programming Language** : Python
* **Audio Processing** : MFCCs, Delta, Delta² features
* **Machine Learning** : Deep learning with LSTM and CNN layers

## Usage

1. Launch the application.
2. Upload an audio file.
3. Wait for the processing to complete.
4. View the detected emotion and confidence score.

## Example

### Input

* Audio file: `example.wav`

### Output

* Detected Emotion: **Happy**
* Confidence: **92.5%**

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch:

   **git** **checkout** **-b** **feature-name**
3. Commit your changes:

   **git** **commit** **-m** **"Add feature-name"**
4. Push to the branch:

   **git** **push** **origin** **feature-name**
5. Open a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

* **Datasets** : RAVDESS, CREMA-D, TESS, SAVEE
* **Libraries** : TensorFlow, Librosa, Streamlit, Pydub

---

Feel free to reach out for any questions or feedback! 🎉
