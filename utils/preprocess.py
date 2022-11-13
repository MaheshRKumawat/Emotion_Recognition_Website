import numpy as np
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, TFBertModel
from tqdm._tqdm_notebook import tqdm_notebook
from .predict import predict_speech

tqdm_notebook.pandas()
load_dotenv()

df = pd.read_csv(
    'Emotion-Recognition-using-Text-with-Emojis-and-Speech/text_dataset/train.csv')

train, test = train_test_split(df, random_state=42, test_size=0.2)
train.shape, test.shape
train, val = train_test_split(train, random_state=42, test_size=0.1)
train.shape, val.shape
max_len = 70
input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
input_mask = Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
bert = TFBertModel.from_pretrained('bert-base-cased')

# (0 is the last hidden states,1 means pooler_output)
embeddings = bert(input_ids, attention_mask=input_mask)[0]
out = tf.keras.layers.GlobalMaxPool1D()(embeddings)
out = Dense(256, activation='relu')(out)
out = tf.keras.layers.Dropout(0.1)(out)
out = Dense(128, activation='relu')(out)
out = tf.keras.layers.Dropout(0.1)(out)
out = Dense(32, activation='relu')(out)
#out = tf.keras.layers.Dropout(0.6)(out)
y = Dense(5, activation='sigmoid')(out)

bert_model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=y)
bert_model.layers[2].trainable = True


bilstm_model = load_model(
    "Emotion-Recognition-using-Text-with-Emojis-and-Speech/model/bi-lstm.h5")
bert_model.load_weights(
    "Emotion-Recognition-using-Text-with-Emojis-and-Speech/model/bert_model.h5")
speech_model = load_model(
    "Emotion-Recognition-using-Text-with-Emojis-and-Speech/model/speech_model.h5")


X = train['text']
bilstm_tokenizer = Tokenizer(15212, lower=True, oov_token='UNK')
bilstm_tokenizer.fit_on_texts(X)

bert_encoded_dict = {0: 'anger', 1: 'fear', 2: 'happy', 3: 'neutral', 4: 'sad'}
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')


def get_key(value):
    dictionary = {'happy': 0, 'angry': 1, 'neutral': 2, 'sad': 3, 'fear': 4}
    for key, val in dictionary.items():
        if (val == np.argmax(value)):
            return key


def bilstm_predict(sentence):
    print("Original text:", sentence)
    sentence_lst = []
    sentence_lst.append(sentence)
    sentence_seq = bilstm_tokenizer.texts_to_sequences(sentence_lst)
    print("preprocessed text : ", sentence_seq)
    sentence_padded = pad_sequences(sentence_seq, maxlen=80, padding='post')
    ans = get_key(bilstm_model.predict(sentence_padded))
    return ans


def bilstm_preprocess(raw_text):
    return bilstm_predict(raw_text)


def bert_preprocess(raw_text):
    x_val = bert_tokenizer(
        text=[raw_text],
        add_special_tokens=True,
        max_length=70,
        truncation=True,
        padding='max_length',
        return_tensors='tf',
        return_token_type_ids=False,
        return_attention_mask=True,
        verbose=True)
    bert_emotion_val = bert_model.predict(
        {'input_ids': x_val['input_ids'], 'attention_mask': x_val['attention_mask']})*100
    print(bert_emotion_val)
    bert_emotion_key = np.argmax(bert_emotion_val)
    return bert_encoded_dict[bert_emotion_key]


def speech_preprocess(raw_speech):
    # The model can handle 2-3 seconds of speech
    # Break the speech into chunks of 2.5 seconds
    # and predict the emotion for each chunk
    # and return the emotion with the highest frequency
    # as the final emotion

    # emotion_list = []
    # for i in range(0, len(raw_speech), 2500):
    #     chunk = raw_speech[i:i+2500]
    #     emotion_list.append(get_key(predict_speech(chunk)))
    # return max(set(emotion_list), key=emotion_list.count)
    pass