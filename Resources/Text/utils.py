import re
import nltk
import numpy as np
import pandas as pd
import contractions
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.optimizers import Adam
from transformers import RobertaTokenizerFast
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

max_len = 75


def preprocess(sentence):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    sentence = re.sub('[^A-z]', ' ', sentence)
    negative = ['not', 'neither', 'nor', 'but', 'however', 'although', 'nonetheless', 'despite', 'except',
                'even though', 'yet']
    stop_words = [z for z in stop_words if z not in negative]
    preprocessed_tokens = [lemmatizer.lemmatize(contractions.fix(temp.lower(
    ))) for temp in sentence.split() if temp not in stop_words]  # lemmatization
    return ' '.join([x for x in preprocessed_tokens]).strip()


def roberta_encode(data, maximum_length):
    input_ids = []
    attention_masks = []

    for i in range(len(data.text)):
        encoded = tokenizer.encode_plus(
            data.text[i],
            add_special_tokens=True,
            max_length=maximum_length,
            pad_to_max_length=True,
            return_attention_mask=True,
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    return np.array(input_ids), np.array(attention_masks)


def roberta_inference_encode(data, maximum_length):
    input_ids = []
    attention_masks = []

    encoded = tokenizer.encode_plus(
        data,
        add_special_tokens=True,
        max_length=maximum_length,
        pad_to_max_length=True,
        return_attention_mask=True
    )

    input_ids.append(encoded['input_ids'])
    attention_masks.append(encoded['attention_mask'])
    return np.array(input_ids), np.array(attention_masks)

def roberta_create_model(bert_model, max_len):
    input_ids = tf.keras.Input(shape=(max_len,),dtype='int32')
    attention_masks = tf.keras.Input(shape=(max_len,),dtype='int32')

    output = bert_model([input_ids,attention_masks])
    output = output[1]

    output = tf.keras.layers.Dense(5, activation='softmax')(output)
    model = tf.keras.models.Model(inputs = [input_ids,attention_masks],outputs = output)
    model.compile(Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    return model