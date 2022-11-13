import os
import numpy as np
import streamlit as st
from io import BytesIO
import streamlit.components.v1 as components
# from utils.preprocess import *
from utils.predict import *

st.markdown(
    '''<style>.css-1egvi7u {margin-top: -3rem;}</style>''', unsafe_allow_html=True)
st.markdown(
    '''<style>.stAudio {height: 45px;}</style>''', unsafe_allow_html=True)
st.markdown(
    '''<style>.css-v37k9u a {color: #ff4c4b;}</style>''', unsafe_allow_html=True)
st.markdown(
    '''<style>.css-nlntq9 a {color: #ff4c4b;}</style>''', unsafe_allow_html=True)


def audiorec_demo_app():
    # parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join("audio_record/frontend/build")
    st_audiorec = components.declare_component("st_audiorec", path=build_dir)

    st.title('Speech Emotion Recognizer')
    st.write('\n\n')
    val = st_audiorec()
    st.write(
        'Audio data received, analyzing emotion...')

    if isinstance(val, dict):
        with st.spinner('retrieving audio-recording...'):
            ind, val = zip(*val['arr'].items())
            ind = np.array(ind, dtype=int)  # convert to np array
            val = np.array(val)             # convert to np array
            sorted_ints = val[ind]
            stream = BytesIO(
                b"".join([int(v).to_bytes(1, "big") for v in sorted_ints]))
            wav_bytes = stream.read()

        # wav_bytes contains audio data in format to be further processed
        # display audio data as received on the Python side
        st.audio(wav_bytes, format='audio/wav')
        # print("Hello there: ",wav_bytes)

        # save audio data to file in Data folder
        with open('Data/audio.wav', 'wb') as f:
            f.write(wav_bytes)

        # run the audio emotion recognition model
        
        emotion = predict_speech('Data/audio.wav')

        st.write(emotion)

if __name__ == '__main__':
    audiorec_demo_app()
