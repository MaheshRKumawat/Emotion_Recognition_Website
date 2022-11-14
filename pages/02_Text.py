import os
import numpy as np
import streamlit as st
from io import BytesIO
import streamlit.components.v1 as components
from utils.predict import *
from pages.Song import Song_Recommendation

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
    emotion = predict_text(text)

    if st.button('Analyze'):
        st.write("##")
        placeholder = st.empty()
        if emotion == 'Happy':
            placeholder.success(f'Emotion: {emotion}! Here are some songs to cheer you more!')
        elif emotion == 'Neutral':
            placeholder.info(f'Emotion: {emotion}! Here are some songs to cheer you up!')
        elif emotion == 'Angry' or emotion == 'Sad':
            placeholder.error(f'Emotion: {emotion}! Here are some songs to make you feel realxed!')
        else:
            placeholder.warning(f'Emotion: {emotion}! Here are some songs we recommend to relax you a bit!')

        st.write("##")

        st.write("We would like to recommend you songs or videos to cheer you up!")

        st.write("##")

if __name__ == '__main__':
    text_app()
