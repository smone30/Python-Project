import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import joblib
import neattext.functions as nfx
import tempfile
import os
import matplotlib.pyplot as plt
import numpy as np

emotion_model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
whisper_model = whisper.load_model("base")

def predict_emotion(text):
    cleaned = nfx.remove_special_characters(text)
    vect = vectorizer.transform([cleaned])
    prediction = emotion_model.predict(vect)[0]
    
    try:
        probs = emotion_model.predict_proba(vect)[0]
        labels = emotion_model.classes_
    except AttributeError:
        probs = None
        labels = None
    
    return prediction, probs, labels

def plot_emotion_probs(probs, labels):
    fig, ax = plt.subplots()
    ax.bar(labels, probs, color='skyblue')
    ax.set_ylabel("Probability")
    ax.set_title("Emotion Prediction Confidence")
    ax.set_ylim([0, 1])
    plt.xticks(rotation=45)
    st.pyplot(fig)

def record_audio(duration=5, fs=44100):
    st.info(f"Recording for {duration} seconds... Please speak into your microphone.")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    return recording, fs

def save_audio(recording, fs):
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(temp_wav.name, fs, recording)
    return temp_wav.name

def transcribe_audio(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result["text"]

st.title("🎤 Speech-to-Emotion Detector with Visualization")

option = st.radio("Choose input type:", ("Text Input", "Voice Input"))

if option == "Text Input":
    user_text = st.text_area("Enter your text here:")
    if st.button("Detect Emotion"):
        if user_text.strip():
            emotion, probs, labels = predict_emotion(user_text)
            st.success(f"Detected Emotion: {emotion.upper()}")
            if probs is not None:
                plot_emotion_probs(probs, labels)
        else:
            st.warning("Please enter some text.")

elif option == "Voice Input":
    duration = st.slider("Recording duration (seconds):", 1, 10, 5)
    if st.button("Record & Detect Emotion"):
        recording, fs = record_audio(duration)
        audio_path = save_audio(recording, fs)
        with st.spinner("Transcribing audio..."):
            transcribed_text = transcribe_audio(audio_path)
        st.write(f"Transcribed Text: {transcribed_text}")
        emotion, probs, labels = predict_emotion(transcribed_text)
        st.success(f"Detected Emotion: {emotion.upper()}")
        if probs is not None:
            plot_emotion_probs(probs, labels)
        os.remove(audio_path)
