import numpy as np
from Resources.Speech.utils import *
from Resources.Text.utils import *
from Resources.Text.model import *
import speech_recognition as sr
from tensorflow.keras.models import load_model
from transformers import TFRobertaModel
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

roberta_model = TFRobertaModel.from_pretrained('roberta-base')

#edf = pd.read_csv("EmoTag/data/EmoTag1200-scores.csv")

labels = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad']


def roberta_predict(text_sentence):
    max_len = 75
    preprocessed_text = preprocess(text_sentence)
    input_ids, attention_masks = roberta_inference_encode(
        preprocessed_text, maximum_length=max_len)
    model = roberta_create_model(roberta_model, max_len)
    model.load_weights(
        'Emotion-Recognition-using-Text-with-Emojis-and-Speech/model/roberta.h5')
    result = model.predict([input_ids, attention_masks])
    emotion = labels[np.argmax(result)]
    return emotion


##########################################
##########################################
############## New Speech ################
##########################################
##########################################

# def predict_speech(path):

#     r = sr.Recognizer()
#     rs = sr.AudioFile('Data/audio.wav')
#     with rs as source:
#         audio = r.record(source)
#     try:
#         text_input = r.recognize_google(audio, language='en-IN', show_all=True)
#         print("Google Speech Working")
#     except:
#         print('Google Speech Recognition Failed')
#     text_sentence = text_input['alternative'][0]['transcript']
#     print("Transcript : ", text_sentence)
#     preprocessed_text = preprocess(text_sentence)
#     input_ids, attention_masks = roberta_inference_encode(
#         preprocessed_text, maximum_length=max_len)
#     roberta_text_model = load_model(
#         "Emotion-Recognition-using-Text-with-Emojis-and-Speech\model\\roberta-hybrid-model.h5",
#         custom_objects={'TFRobertaModel': TFRobertaModel})
#     emotion_text = np.array(roberta_text_model.predict([input_ids, attention_masks]))

#     encoder = OneHotEncoder()
#     #fetch the last column of labels and perform one hot encoding on them
#     label = encoder.fit_transform(np.array(labels).reshape(-1,1)).toarray()
    
#     # load the best model
#     res_model = load_model(
#         "Emotion-Recognition-using-Text-with-Emojis-and-Speech/model/speech_cnn.hdf5")

#     # get audio features from the recorded voice
#     feature = get_features_recorded(path)
#     # apply min max scaling
#     scaler = MinMaxScaler()
#     feature = scaler.fit_transform(feature)
#     # get the predicted label
#     label = np.array(res_model.predict(feature))
    
#     # sum1 = np.sum([label, emotion_text], axis=0)
    
#     # get the label information by reversing one hot encoded output
#     label_predicted = encoder.inverse_transform(label)
#     print("\n\n\nThe Emotion Predicted For Recorded Audio Using Microphone is {}".format(
#         label_predicted))

#     print("\n\n\n*********************************\n\n")
#     print("Emotion Text", emotion_text)
#     print("\n\n")
#     print("Label", label)
#     print("\n\n")
#     print("Label Predicted", label_predicted)
#     print("\n\n")
#     print("Label Predicted[0]", label_predicted[0])
#     print("\n\n\n*********************************\n\n")
    
#     return label_predicted[0]


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


def predict_speech(raw_speech):
    r = sr.Recognizer()
    rs = sr.AudioFile(raw_speech)
    with rs as source:
        audio = r.record(source)
    try:
        text_input = r.recognize_google(audio, language='en-IN', show_all=True)
        print("Google Speech Working")
        text_sentence = text_input['alternative'][0]['transcript']
        print("\n\nAll Transcripts : ", text_input)
        print("\nTranscript with highest confidence: ", text_sentence)
        print("Transcript : ", text_sentence)
        preprocessed_text = preprocess(text_sentence)
        input_ids, attention_masks = roberta_inference_encode(
            preprocessed_text, maximum_length=max_len)
        roberta_text_model = load_model(
            "Emotion-Recognition-using-Text-with-Emojis-and-Speech\model\\roberta-hybrid-model.h5",
            custom_objects={'TFRobertaModel': TFRobertaModel})
        emotion_text = roberta_text_model.predict([input_ids, attention_masks])

        speech_model = load_model(
            "Emotion-Recognition-using-Text-with-Emojis-and-Speech/model/new_speech_86.h5", compile=False)
        test_feature = get_features(raw_speech)
        temp = np.resize(test_feature, (4, 2388))
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
        print("emotion_text", emotion_text)
        final_speech_emotion = np.argmax(emotion_text[0], axis=0)
        print("final_speech_emotion", final_speech_emotion)
    except:
        print('Google Speech Recognition Failed')
    return labels[final_speech_emotion]