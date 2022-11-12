import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Resources.Speech.utils import *


def predict_speech(raw_speech):
    speech_model = load_model("Emotion-Recognition-using-Text-with-Emojis-and-Speech/model/speech_model.h5")
    data, sample_rate = librosa.load(raw_speech, duration=3, offset=0)
    res1 = extract_features(data, sample_rate)
    result = np.array(res1)
    pad_sequences(result,2376)
    