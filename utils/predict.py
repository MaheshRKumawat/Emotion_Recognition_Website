import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from Resources.Speech.utils import *
from Resources.Text.utils import *
from Resources.Text.model import *
import speech_recognition as sr
from pydub import AudioSegment
from tensorflow.keras.models import load_model
from transformers import TFRobertaModel

roberta_model = TFRobertaModel.from_pretrained('roberta-base')

labels = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad']

def roberta_predict(text_sentence):
    max_len = 75
    preprocessed_text = preprocess(text_sentence)
    input_ids, attention_masks = roberta_inference_encode(preprocessed_text, maximum_length = max_len)
    model = roberta_create_model(roberta_model, max_len)
    model.load_weights('Emotion-Recognition-using-Text-with-Emojis-and-Speech/model/roberta.h5')
    result = model.predict([input_ids, attention_masks])
    emotion = labels[np.argmax(result)]
    return emotion

def predict_speech(raw_speech):
    r = sr.Recognizer()
    rs = sr.AudioFile('Data/audio.wav')
    with rs as source:
        audio = r.record(source)
    try:
        text_input = r.recognize_google(audio, language='en-IN', show_all=True)
        print("Google Speech Working")
    except:
        print('Google Speech Recognition Failed')
    text_sentence = text_input['alternative'][0]['transcript']
    preprocessed_text = preprocess(text_sentence)
    input_ids, attention_masks = roberta_inference_encode(
        preprocessed_text, maximum_length=max_len)
    roberta_text_model = load_model(
        'Emotion-Recognition-using-Text-with-Emojis-and-Speech/model/roberta.h5',
        custom_objects={'TFRobertaModel': TFRobertaModel})
    emotion_text = roberta_text_model.predict([input_ids, attention_masks])

    speech_model = load_model(
        "Emotion-Recognition-using-Text-with-Emojis-and-Speech/model/speech.h5", compile=False)
    test_feature = get_features(raw_speech)
    temp = np.resize(test_feature, (4, 2376))
    test_final = np.expand_dims(temp, axis=2)

    emotion_speech = speech_model.predict(test_final)
    result = []
    for e in emotion_speech:
        result.append(e)

    print("\n\n*****************\n")
    print("Emotion Speech: ", emotion_speech)
    print("\n\n*****************\n")
    result.append(emotion_text[0])
    result = np.array(result)
    print("\n\n*****************\n")
    print("Result: ", result)
    print("\n\n*****************\n")
    final_speech_emotion = np.argmax(np.average(result, axis=0), axis=0)
    return labels[final_speech_emotion]

def hybrid_predict_text(text_sentence):
    preprocessed_text = preprocess(text_sentence)
    input_ids, attention_masks = roberta_inference_encode(
        preprocessed_text, maximum_length=max_len)
    roberta_text_model = load_model(
        'Emotion-Recognition-using-Text-with-Emojis-and-Speech/model/roberta-hybrid-model.h5',
        custom_objects={'TFRobertaModel': TFRobertaModel})
    result = roberta_text_model.predict([input_ids, attention_masks])
    emotion = labels[np.argmax(result)]
    return emotion

def bilstm_predict_text(text_sentence):
    return bilstm_predict(text_sentence)

def bert_predict_text(text_sentence):
    return bert_predict(text_sentence)