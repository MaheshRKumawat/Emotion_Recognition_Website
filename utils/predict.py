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

'''
/plain roberta
/plain bert
/plain bi LSTM
/hybrid roberta + lstm + cnn
'''

labels = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad']


def predict_speech(raw_speech):
    '''
    AudioSegment.from_wav(
        raw_speech).export("Data/audio1.mp3", format="mp3")
    sound = AudioSegment.from_mp3(file="Data/audio1.mp3")
    sound.export("Data/audio.wav", format="wav")
    r = sr.Recognizer()
    AUDIO_FILE = "Data/audio.wav"
    text_input = ""

    with sr.AudioFile(AUDIO_FILE) as source:
        audio = r.record(source)  # read the entire audio file

        text_input = r.recognize_google(audio, language='en-IN', show_all=True)
        # print(result)
    '''
    r = sr.Recognizer()
    rs = sr.AudioFile('Data/audio.wav')
    with rs as source:
        audio = r.record(source)
    try:
        text_input = r.recognize_google(audio, language='en-IN', show_all=True)
        print("Google Speech Working")
    except:
        # text_input = r.recognize_sphinx(audio, language='en-IN', show_all=True)
        # print("Sphinx Speech Working")
        print('Google Speech Recognition Failed')
    
    print("\n\n\n\n**************************")
    print('text_input', text_input)
    print("**************************\n\n\n\n") 

    print("\n\n\n\n**************************")
    print("text_input['alternative'][0]['transcript'] \n", text_input['alternative'][0]['transcript'])
    print("**************************\n\n\n\n") 

    # roberta model
    text_sentence = text_input['alternative'][0]['transcript']
    '''
    input_ids, attention_masks = roberta_inference_encode(
        preprocessed_text, maximum_length=max_len)
    # roberta_text_model = text_model()
    roberta_text_model = load_model(
        'Emotion-Recognition-using-Text-with-Emojis-and-Speech/model/roberta.h5',
        custom_objects={'TFRobertaModel': TFRobertaModel})
    result = roberta_text_model.predict([input_ids, attention_masks])
    '''
    preprocessed_text = preprocess(text_sentence)
    input_ids, attention_masks = roberta_inference_encode(
        preprocessed_text, maximum_length=max_len)
    roberta_text_model = load_model(
        'Emotion-Recognition-using-Text-with-Emojis-and-Speech/model/roberta.h5',
        custom_objects={'TFRobertaModel': TFRobertaModel})
    emotion_text = roberta_text_model.predict([input_ids, attention_masks])
    
    print("\nEmotion Text Array: ", emotion_text)

    speech_model = load_model(
        "Emotion-Recognition-using-Text-with-Emojis-and-Speech/model/speech_model.h5", compile=False)
    test_audio_arrays = []
    test_x, _ = librosa.load(raw_speech, sr=44100)
    test_audio_arrays.append(test_x)
    test_feature = get_features(np.array(test_x))
    test_final = np.expand_dims(test_feature, axis=2)
    emotion_speech = speech_model.predict(test_final)
    print("\nEmotion Speech Array: ", emotion_speech)
    result = []
    for e in emotion_speech:
        result.append(e)

    result.append(emotion_text[0])
    print("\n\n\n\n**************************")
    print("result: ", result)
    print("**************************\n\n\n\n")

    final_speech_emotion = np.argmax(np.average(result, axis=0), axis=1)
    print("\n\n\n\n**************************")
    print("final_speech_emotion: ", final_speech_emotion)
    print("**************************\n\n\n\n")

    return labels[final_speech_emotion]

    # result = []
    # for emo in y_pred_test:
    #     speech_emotion = labels[np.argmax(emo)]
    #     result.append(speech_emotion)

    # emotion_speech = y_pred_test

    # append the emotion_text and emotion_speech into one array
    # final_input = np.append(emotion_text, emotion_speech, axis=1)
    # print("\nFinal Input Array: ", final_input)
    # # print('Result: ', result)
    # # print("Emotion Speech Array: ", emotion_speech)
    # # final_input = np.array([emotion_text, emotion_speech])
    # print("\nText Input: ", text_input)
    # final_speech_emotion = np.argmax(np.average(final_input, axis=0), axis=1)
    # print("\nfinal_speech_emotion: ", final_speech_emotion)
    # return final_speech_emotion


def predict_text(text_sentence):
    preprocessed_text = preprocess(text_sentence)
    input_ids, attention_masks = roberta_inference_encode(
        preprocessed_text, maximum_length=max_len)
    roberta_text_model = load_model(
        'Emotion-Recognition-using-Text-with-Emojis-and-Speech/model/roberta.h5',
        custom_objects={'TFRobertaModel': TFRobertaModel})
    result = roberta_text_model.predict([input_ids, attention_masks])
    emotion = labels[np.argmax(result)]
    return emotion
