import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from transformers import TFRobertaModel
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from transformers import AutoTokenizer, TFBertModel, RobertaTokenizerFast
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPool1D, LSTM, Flatten, LeakyReLU, Input


def hybrid_model():
    max_len = 75
    roberta_model = TFRobertaModel.from_pretrained('roberta-base')
    input_ids = Input(shape=(max_len,), name='input_ids', dtype=tf.int32)
    attention_masks = Input(
        shape=(max_len,),  name='attention_mask', dtype=tf.int32)
    base_model_output = roberta_model([input_ids, attention_masks])

    x = base_model_output[0]

    x1 = Dropout(0.1)(x)
    x1 = Conv1D(128, 5, padding='same')(x1)
    x1 = MaxPool1D(pool_size=1, padding='valid')(x1)
    x1 = Conv1D(256, 5, padding='same')(x1)
    x1 = MaxPool1D(pool_size=1, padding='valid')(x1)
    x1 = Conv1D(512, 5, padding='same')(x1)
    x1 = MaxPool1D(pool_size=1, padding='valid')(x1)
    x1 = Conv1D(1024, 5, padding='same')(x1)
    x1 = MaxPool1D(pool_size=1, padding='valid')(x1)

    x1 = LSTM(1024, return_sequences=True)(x1)
    x1 = LSTM(2048, return_sequences=True)(x1)
    x1 = Dropout(0.3)(x1)

    x1 = Dense(1024, activation='relu')(x1)
    x1 = LeakyReLU()(x1)
    x1 = Dense(512, activation='relu')(x1)
    x1 = LeakyReLU()(x1)
    x1 = Dense(256, activation='relu')(x1)
    x1 = LeakyReLU()(x1)
    x1 = Dense(64, activation='relu')(x1)
    x1 = LeakyReLU()(x1)
    x1 = Flatten()(x1)

    output = Dense(5, activation='softmax')(x1)

    model = tf.keras.models.Model(
        inputs=[input_ids, attention_masks], outputs=output)

    return model


def bert_model():
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

    return bert_model


bilstm_model = load_model(
    "Emotion-Recognition-using-Text-with-Emojis-and-Speech/model/bi-lstm.h5")

bert_model = bert_model()
bert_model.load_weights(
    "Emotion-Recognition-using-Text-with-Emojis-and-Speech/model/bert_model.h5")


def get_key(value):
    dictionary = {'happy': 0, 'angry': 1, 'neutral': 2, 'sad': 3, 'fear': 4}
    for key, val in dictionary.items():
        if (val == np.argmax(value)):
            return key


def bilstm_predict(sentence):
    df = pd.read_csv(
        'Emotion-Recognition-using-Text-with-Emojis-and-Speech/text_dataset/train.csv')
    train, test = train_test_split(df, random_state=42, test_size=0.2)
    train, val = train_test_split(train, random_state=42, test_size=0.1)

    X = train['text']
    bilstm_tokenizer = Tokenizer(15212, lower=True, oov_token='UNK')
    bilstm_tokenizer.fit_on_texts(X)
    print("Original text:", sentence)
    sentence_lst = []
    sentence_lst.append(sentence)
    sentence_seq = bilstm_tokenizer.texts_to_sequences(sentence_lst)
    print("preprocessed text : ", sentence_seq)
    sentence_padded = pad_sequences(sentence_seq, maxlen=80, padding='post')
    ans = get_key(bilstm_model.predict(sentence_padded))
    return ans


def bert_predict(raw_text):
    bert_encoded_dict = {0: 'anger', 1: 'fear',
                         2: 'happy', 3: 'neutral', 4: 'sad'}
    bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
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
