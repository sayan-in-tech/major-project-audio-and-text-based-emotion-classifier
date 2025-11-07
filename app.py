import streamlit as st
import numpy as np
import librosa
import pickle
import tensorflow as tf
import os
import tempfile
import time
from audio_formatter import convert_to_wav

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
def predict_emotion(file):
    try:
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
        confidence = np.max(preds) * 100
        return emotion, confidence
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        return None, None

# 📤 File Upload Section
st.markdown("### 📤 Upload your audio file below:")
uploaded_file = st.file_uploader("", type=["wav", "mp3", "flac", "ogg", "m4a", "aac"], label_visibility="collapsed")

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
            emotion, confidence = predict_emotion(wav_path)

        if emotion:
            st.markdown("---")
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown(
                    f"<h3 style='color: #1F77B4;'>🔊 Emotion Detected:</h3><h2 style='color: #FF4B4B;'>{emotion.upper()}</h2>",
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(
                    f"<h3 style='color: #1F77B4;'>📈 Confidence:</h3><h2 style='color: #2CA02C;'>{confidence:.2f}%</h2>",
                    unsafe_allow_html=True
                )

            st.balloons()
        else:
            st.warning("⚠️ Unable to detect emotion. Please try another file.")
    else:
        st.error("❌ Conversion failed. Please upload a valid audio file.")
