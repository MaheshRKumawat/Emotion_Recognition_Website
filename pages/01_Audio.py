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

def audiorec_app():
    # parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join("audio_record/frontend/build")
    st_audiorec = components.declare_component("st_audiorec", path=build_dir)

    st.title('Speech Emotion Recognizer')
    st.write('\n\n')
    val = st_audiorec()

    if isinstance(val, dict):
        st.write('Audio data received, analyzing emotion...')
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

        # recommend songs based on emotion
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

        # col1, col2, col3 = st.columns([2, 1, 2])
        # with col1:
        #     if col1.button("Recommend Songs"):
        #         # move to song recommendation page
        #         pass
                
                
        # with col3:
        #     if col3.button("Recommend Videos"):
        #         # st.session_state.runpage = Song_Recommendation(emotion)
        #         st.write("Coming Soon!")

        # st.write("##")


if __name__ == '__main__':
    audiorec_app()
