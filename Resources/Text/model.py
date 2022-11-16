import tensorflow as tf
from transformers import TFRobertaModel
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPool1D, LSTM, Flatten, LeakyReLU

max_len = 75

roberta_model = TFRobertaModel.from_pretrained('roberta-base')


def create_model(bert_model, max_len):
    input_ids = tf.keras.Input(shape=(max_len,), dtype='int32')
    attention_masks = tf.keras.Input(shape=(max_len,), dtype='int32')
    base_model_output = bert_model([input_ids, attention_masks])

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


def text_model():
    model = create_model(roberta_model, max_len)
    return model
