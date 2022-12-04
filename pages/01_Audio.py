import os
import Home
import gridfs
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from io import BytesIO
from utils.predict import *

db = Home.db
user_id = st.session_state.get('user_id', None)
fs = gridfs.GridFS(db)
filename = "speech.wav"


st.markdown(
    '''<style>.css-1egvi7u {margin-top: -3rem;}</style>''', unsafe_allow_html=True)
st.markdown(
    '''<style>.stAudio {height: 45px;}</style>''', unsafe_allow_html=True)
st.markdown(
    '''<style>.css-v37k9u a {color: #ff4c4b;}</style>''', unsafe_allow_html=True)
st.markdown(
    '''<style>.css-nlntq9 a {color: #ff4c4b;}</style>''', unsafe_allow_html=True)

def audiorec_app():
    build_dir = os.path.join("audio_record/frontend/build")
    st_audiorec = components.declare_component("st_audiorec", path=build_dir)

    st.title('Speech Emotion Recognizer')
    st.write('\n\n')
    val = st_audiorec()

    if isinstance(val, dict):
        st.write('Audio data received, analyzing emotion...')
        with st.spinner('retrieving audio-recording...'):
            ind, val = zip(*val['arr'].items())
            ind = np.array(ind, dtype=int)
            val = np.array(val)
            sorted_ints = val[ind]
            stream = BytesIO(
                b"".join([int(v).to_bytes(1, "big") for v in sorted_ints]))
            wav_bytes = stream.read()

        st.audio(wav_bytes, format='audio/wav')
        
        with open('Data/audio.wav', 'wb') as f:
            f.write(wav_bytes)

        emotion = predict_speech('Data/audio.wav')

        if wav_bytes:
            emotion = "Happy"

            placeholder = st.empty()

            if emotion == 'Happy':
                placeholder.success(
                    f'Emotion: {emotion}! Here are some songs to cheer you more!')
            elif emotion == 'Neutral':
                placeholder.info(
                    f'Emotion: {emotion}! Here are some songs to cheer you up!')
            elif emotion == 'Angry' or emotion == 'Sad':
                placeholder.error(
                    f'Emotion: {emotion}! Here are some songs to make you feel realxed!')
            else:
                placeholder.warning(
                    f'Emotion: {emotion}! Here are some songs we recommend to relax you a bit!')

            st.session_state.emotion = emotion
            st.session_state.speech_id = fs.put(wav_bytes, filename=filename)
            st.session_state.speech = True
            st.session_state.text = False
            st.session_state.feedback = False
            st.write("##")
            st.write("We would like to recommend you songs or videos to cheer you up!")
            st.write("Click on the Song or Video Page at the left to get started!")


if __name__ == '__main__':
    user_id = st.session_state.get('user_id', None)
    if user_id is None:
        st.write("Please login to continue")
    else:
        audiorec_app()
