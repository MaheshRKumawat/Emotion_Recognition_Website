import os
import numpy as np
import streamlit as st
from io import BytesIO
import streamlit.components.v1 as components
from utils.predict import *
from utils.demojize import *

st.markdown(
    '''<style>.css-1egvi7u {margin-top: -3rem;}</style>''', unsafe_allow_html=True)
st.markdown(
    '''<style>.stAudio {height: 45px;}</style>''', unsafe_allow_html=True)
st.markdown(
    '''<style>.css-v37k9u a {color: #ff4c4b;}</style>''', unsafe_allow_html=True)
st.markdown(
    '''<style>.css-nlntq9 a {color: #ff4c4b;}</style>''', unsafe_allow_html=True)


def text_app():
    st.title('Text Emotion Recognizer')
    st.write('\n\n')
    st.write('''
    ## Enter the text below to analyze the emotion
    ''')
    text = st.text_area("Enter text here")
    Analyze = st.button("Analyze")
    if Analyze:
        if text != '':
            st.write('\n\n')
            # demojize start
            text = remove_emoji(text)
            print("Demojized text :", text)
            # demojize end
            roberta_emotion = roberta_predict(text)
            hybrid_emotion = hybrid_predict_text(text)
            bilstm_emotion = bilstm_predict_text(text)
            bert_emotion = bert_predict_text(text)
            st.write('#')
            placeholder = st.empty()
            placeholder.info(placeholder_value(hybrid_emotion, 'Hybrid'))
            st.write("##")
            placeholder = st.empty()
            placeholder.info(placeholder_value(roberta_emotion, 'RoBerta'))
            st.write('#')
            placeholder = st.empty()
            placeholder.info(placeholder_value(bilstm_emotion, "Bi-LSTM"))
            st.write('#')
            placeholder = st.empty()
            placeholder.info(placeholder_value(bert_emotion, "BERT"))

            st.write("##")
            st.write(
                "We would like to recommend you songs or videos to cheer you up!")
            st.write("Click on the Song or Video Page at the left to get started!")
            st.write("##")
            final_emotion = hybrid_emotion
            # save the emotion to a file and save it to Data folder
            with open('Data/emotion.txt', 'w') as f:
                f.write(final_emotion)

        else:
            st.write('\n\n')
            st.write('''## Please enter the text to analyze the emotion''')


def placeholder_value(emotion, model_name):
    text = ''
    if emotion == 'Happy':
        text = model_name + \
            " model predicts that you are Happy! Here are some songs to cheer you more!"
    elif emotion == 'Neutral':
        text = model_name + \
            " model predicts that you are Neutral! Here are some songs to cheer you up!"
    elif emotion == 'Angry' or emotion == 'Sad':
        text = model_name + " model predicts that you are " + \
            emotion + "! Here are some songs to make you feel realxed!"
    else:
        text = model_name + " model predicts that you are " + emotion + \
            "! Here are some songs we recommend to relax you a bit!"
    return text


if __name__ == '__main__':
    text_app()
