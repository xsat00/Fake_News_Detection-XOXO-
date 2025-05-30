import whisper
import torch
import streamlit as st

@st.cache_resource(show_spinner=False)
def load_whisper_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("base", device=device)
    return model

def transcribe_audio(model, audio_path):
    result = model.transcribe(audio_path)
    return result["text"]
