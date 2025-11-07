import os

os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

import streamlit as st
import numpy as np
import pandas as pd
import librosa
import pickle
import tensorflow as tf
import joblib
import whisper
import torch
import tempfile
import time
from pathlib import Path
from audio_formatter import convert_to_wav


try:
    torch_classes = getattr(torch, "classes", None)
    if torch_classes is not None:
        if not hasattr(torch_classes, "__path__"):
            class _TorchClassesPath(list):
                def __init__(self):
                    super().__init__()
                    self._path = []

                def __iter__(self):
                    return iter(self._path)

            torch_classes.__path__ = _TorchClassesPath()
        elif not hasattr(torch_classes.__path__, "_path"):
            torch_classes.__path__._path = []
    del torch_classes
except Exception:
    # Safe to continue even if patching torch classes fails.
    pass

BASE_DIR = Path(__file__).resolve().parent
TEXT_MODEL_DIR = BASE_DIR / "Emotion-Detection-in-Text"
TEXT_MODEL_PATH = TEXT_MODEL_DIR / "models" / "emotion_classifier_pipe_lr.pkl"
# Set page config
st.set_page_config(page_title="Emotion Recognition from Speech", page_icon="🎧", layout="centered")

# 🎵 Title Section
st.markdown(
    """
    <h1 style='text-align: center;'>🎧 Emotion Recognition from Speech</h1>
    <p style='text-align: center; font-size: 18px;'>Upload any audio file to detect the emotion in the voice!</p>
    """,
    unsafe_allow_html=True
)

# 🔄 Load model and dependencies
@st.cache_resource
def load_model_and_utils():
    try:
        model = tf.keras.models.load_model("models/emotion_recognition_model.keras")
        scaler_params = np.load("models/scaler_params.npy", allow_pickle=True).item()
        mean = scaler_params['mean']
        scale = scaler_params['scale']
        with open("models/label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
        return model, mean, scale, le
    except Exception as e:
        st.error(f"❌ Error loading model or dependencies: {e}")
        return None, None, None, None

model, mean, scale, le = load_model_and_utils()


@st.cache_resource
def load_text_emotion_model():
    try:
        return joblib.load(TEXT_MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Error loading text emotion model: {e}")
        return None


@st.cache_resource
def load_whisper_model():
    model_name = os.getenv("WHISPER_MODEL_NAME", "small.en")
    try:
        return whisper.load_model(model_name)
    except Exception as e:
        st.error(f"❌ Error loading Whisper model '{model_name}': {e}")
        return None


pipe_lr = load_text_emotion_model()
whisper_model = load_whisper_model()


TEXT_TO_VOICE_EMOTION_MAP = {
    "anger": "angry",
    "joy": "happy",
    "fear": "fear",
    "disgust": "disgust",
    "sadness": "sad",
    "shame": "neutral",
    "surprise": "surprise",
}


def map_text_emotion_to_voice(emotion_label: str) -> str:
    return TEXT_TO_VOICE_EMOTION_MAP.get(emotion_label, emotion_label)


def aggregate_voice_probabilities(prob_matrix, classes):
    voice_probabilities = {}
    for cls, prob in zip(classes, prob_matrix[0]):
        voice_label = map_text_emotion_to_voice(cls)
        voice_probabilities[voice_label] = voice_probabilities.get(voice_label, 0.0) + float(prob)
    return voice_probabilities


def aggregate_sentiment_labels(audio_label: str, audio_conf: float, text_label: str, text_conf: float):
    w_audio = 0.58
    w_text = 0.42
    if audio_label == text_label:
        final_label = audio_label
        final_conf = (w_audio * audio_conf + w_text * text_conf) / (w_audio + w_text)
    else:
        audio_score = w_audio * audio_conf
        text_score = w_text * text_conf
        if audio_score > text_score:
            final_label = audio_label
            final_conf = audio_score / (w_audio + w_text)
        else:
            final_label = text_label
            final_conf = text_score / (w_audio + w_text)
    return final_label, round(final_conf, 3)

# 🎯 Feature Extraction
def extract_features(file_path, target_len=40):
    try:
        audio, sr = librosa.load(file_path, sr=22050)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        if mfcc.shape[1] < target_len:
            pad_width = target_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :target_len]
        return mfcc
    except Exception as e:
        st.error(f"❌ Error extracting features: {e}")
        return None

# 🔁 Delta & Delta²
def enhance_features(X):
    delta = np.diff(X, axis=1)
    delta = np.pad(delta, ((0, 0), (1, 0)), mode='edge')
    delta2 = np.diff(delta, axis=1)
    delta2 = np.pad(delta2, ((0, 0), (1, 0)), mode='edge')
    return np.stack([X, delta, delta2], axis=0)

# 🔍 Prediction
def predict_audio_emotion(file):
    try:
        if model is None:
            return None, None
        features = extract_features(file)
        if features is None:
            return None, None

        enhanced = enhance_features(features)

        mfcc_std = (enhanced[0] - mean[:, np.newaxis]) / scale[:, np.newaxis]
        delta_std = (enhanced[1] - mean[:, np.newaxis]) / scale[:, np.newaxis]
        delta2_std = (enhanced[2] - mean[:, np.newaxis]) / scale[:, np.newaxis]

        enhanced_std = np.stack([mfcc_std, delta_std, delta2_std], axis=0)
        enhanced_std = enhanced_std[:, :, :40]
        input_data = np.expand_dims(enhanced_std[:, :, 0], axis=0)

        preds = model.predict(input_data)
        predicted_index = np.argmax(preds)
        emotion = le.inverse_transform([predicted_index])[0]
        confidence = float(np.max(preds))
        return emotion, confidence
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        return None, None


def transcribe_audio(file_path: str) -> str:
    if whisper_model is None:
        return ""
    try:
        result = whisper_model.transcribe(file_path, language="en")
        return result.get("text", "").strip()
    except Exception as e:
        st.error(f"❌ Transcription error: {e}")
        return ""


def predict_text_emotion(text: str):
    if not text or pipe_lr is None:
        return None, None, {}
    try:
        prediction = pipe_lr.predict([text])[0]
        mapped_prediction = map_text_emotion_to_voice(prediction)
        probability = pipe_lr.predict_proba([text])
        voice_probabilities = aggregate_voice_probabilities(probability, pipe_lr.classes_)
        mapped_confidence = voice_probabilities.get(mapped_prediction, 0.0)
        return mapped_prediction, mapped_confidence, voice_probabilities
    except Exception as e:
        st.error(f"❌ Text emotion prediction error: {e}")
        return None, None, {}

# 📤 File Upload Section
st.markdown("### 📤 Upload your audio file below:")
uploaded_file = st.file_uploader(
    "Upload audio file",
    type=["wav", "mp3", "flac", "ogg", "m4a", "aac"],
    label_visibility="collapsed",
)

if uploaded_file is not None:
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()

    if file_ext != ".wav":
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(percent, text):
            progress_bar.progress(percent)
            status_text.text(text)

        wav_path = convert_to_wav(uploaded_file, progress_callback=update_progress)

        # Clear progress after done
        progress_bar.empty()
        status_text.empty()
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.read())
            wav_path = tmp_file.name

    if wav_path:
        st.audio(wav_path, format="audio/wav")

        with st.spinner("🔍 Analyzing emotion..."):
            time.sleep(1.2)
            audio_label, audio_confidence = predict_audio_emotion(wav_path)
            transcript = transcribe_audio(wav_path)
            text_label, text_confidence, text_probabilities = predict_text_emotion(transcript)

        if audio_label:
            st.markdown("---")
            st.markdown("#### 🔊 Voice Emotion Analysis")
            audio_cols = st.columns([1, 1])

            with audio_cols[0]:
                st.markdown(
                    f"<h3 style='color: #1F77B4;'>Emotion:</h3><h2 style='color: #FF4B4B;'>{audio_label.upper()}</h2>",
                    unsafe_allow_html=True
                )
            with audio_cols[1]:
                st.markdown(
                    f"<h3 style='color: #1F77B4;'>Confidence:</h3><h2 style='color: #2CA02C;'>{audio_confidence * 100:.2f}%</h2>",
                    unsafe_allow_html=True
                )

        if transcript:
            st.markdown("---")
            st.markdown("#### 📝 Transcription")
            st.info(transcript)

        if text_label:
            st.markdown("---")
            st.markdown("#### 📚 Text Emotion Analysis")
            text_cols = st.columns([1, 1])

            with text_cols[0]:
                st.markdown(
                    f"<h3 style='color: #1F77B4;'>Emotion:</h3><h2 style='color: #FF4B4B;'>{text_label.upper()}</h2>",
                    unsafe_allow_html=True
                )
            with text_cols[1]:
                st.markdown(
                    f"<h3 style='color: #1F77B4;'>Confidence:</h3><h2 style='color: #2CA02C;'>{text_confidence * 100:.2f}%</h2>",
                    unsafe_allow_html=True
                )

            if text_probabilities:
                proba_df = (
                    pd.DataFrame.from_dict(text_probabilities, orient="index", columns=["probability"])
                    .sort_values("probability", ascending=False)
                )
                st.bar_chart(proba_df)
        else:
            if pipe_lr is None:
                st.warning("⚠️ Text emotion model unavailable. Check configuration.")
            elif transcript == "":
                if whisper_model is None:
                    st.warning("⚠️ Transcription disabled because Whisper model could not load.")
                else:
                    st.warning("⚠️ Transcription produced empty text. Try another recording.")

        if audio_label and text_label:
            final_label, final_confidence = aggregate_sentiment_labels(
                audio_label, audio_confidence, text_label, text_confidence
            )
            st.markdown("---")
            st.markdown("### 🧠 Final Aggregated Emotion")
            final_cols = st.columns([1, 1])

            with final_cols[0]:
                st.markdown(
                    f"<h3 style='color: #1F77B4;'>Emotion:</h3><h2 style='color: #FF4B4B;'>{final_label.upper()}</h2>",
                    unsafe_allow_html=True
                )
            with final_cols[1]:
                st.markdown(
                    f"<h3 style='color: #1F77B4;'>Confidence:</h3><h2 style='color: #2CA02C;'>{final_confidence * 100:.2f}%</h2>",
                    unsafe_allow_html=True
                )

            st.balloons()

        if not audio_label:
            st.warning("⚠️ Unable to detect voice emotion. Please try another file.")
    else:
        st.error("❌ Conversion failed. Please upload a valid audio file.")
