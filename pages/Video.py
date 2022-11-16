import streamlit as st
import streamlit.components.v1 as components
from Resources.Recommendation.utils import *
import webbrowser


def Video_Recommendation(Emotion):

    st.title("Video Recommendation")
    text = "#### Based on your emotion which is \"" + \
        Emotion + "\" let us recommend you videos from YouTube"
    st.write(text)
    st.write("###")
    
    df = pd.DataFrame()
    df = get_channel(Emotion)
    df = df.reset_index(drop=True)

    # Show buttons in 4 columns
    col1, col2, col3, col4 = st.columns(4)
    for i, (uri, name) in enumerate(zip(df["Link"], df["Channel"])):
        if i % 4 == 0:
            with col1:
                if st.button(name):
                    webbrowser.open(uri)
        elif i % 4 == 1:
            with col2:
                if st.button(name):
                    webbrowser.open(uri)
        elif i % 4 == 2:
            with col3:
                if st.button(name):
                    webbrowser.open(uri)
        else:
            with col4:
                if st.button(name):
                    webbrowser.open(uri)


if __name__ == '__main__':
    # take emotion from Data folder file
    path = 'Data/emotion.txt'
    emotion = "Neutral"
    with open(path, 'r') as f:
        emotion = f.read()
    Video_Recommendation(emotion)
