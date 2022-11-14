import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from Resources.Speech.utils import *
from Resources.Text.utils import *

labels = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad']


def predict_speech(raw_speech):
    speech_model = load_model(
        "Emotion-Recognition-using-Text-with-Emojis-and-Speech/model/speech_model.h5", compile=False)
    test_audio_arrays = []
    test_x, _ = librosa.load(raw_speech, sr=44100)
    test_audio_arrays.append(test_x)
    test_feature = get_features(np.array(test_x))
    test_final = np.expand_dims(test_feature, axis=2)
    y_pred_test = speech_model.predict(test_final)
    for emo in y_pred_test:
        emotion = labels[np.argmax(emo)]
    return emotion


def predict_text(text_sentence, max_len):
    preprocessed_text = preprocess(text_sentence)
    input_ids, attention_masks = roberta_inference_encode(
        preprocessed_text, maximum_length=max_len)
    roberta_text_model = load_model(
        'Emotion-Recognition-using-Text-with-Emojis-and-Speech/model/roberta.h5', compile=False)
    result = roberta_text_model.predict([input_ids, attention_masks])
    emotion = labels[np.argmax(result)]
    return emotion
