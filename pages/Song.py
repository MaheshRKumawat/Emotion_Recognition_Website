import streamlit as st
import streamlit.components.v1 as components
from Resources.Recommendation.utils import *


def Song_Recommendation(Emotion):

    st.title("Song Recommendation")
    text = "#### Based on your emotion which is \"" + Emotion + "\" let us recommend you songs from Spotify"
    st.write(text)
    st.write("###")

    genre_names = ['Dance Pop', 'Electronic', 'Electropop', 'Hip Hop',
                   'Jazz', 'K-pop', 'Latin', 'Pop', 'Pop Rap', 'R&B', 'Rock']
    # default genre
    genre = 'Pop'
    # 11 genres so 11 buttons
    st.write("##### Select your genre")
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    col6, col7, col8, col9, col10, col11 = st.columns([1, 1, 1, 1, 1, 1])

    for i in range(len(genre_names)):
        if i == 0:
            if col1.button(genre_names[i]):
                genre = genre_names[i]
        elif i == 1:
            if col2.button(genre_names[i]):
                genre = genre_names[i]
        elif i == 2:
            if col3.button(genre_names[i]):
                genre = genre_names[i]
        elif i == 3:
            if col4.button(genre_names[i]):
                genre = genre_names[i]
        elif i == 4:
            if col5.button(genre_names[i]):
                genre = genre_names[i]
        elif i == 5:
            if col6.button(genre_names[i]):
                genre = genre_names[i]
        elif i == 6:
            if col7.button(genre_names[i]):
                genre = genre_names[i]
        elif i == 7:
            if col8.button(genre_names[i]):
                genre = genre_names[i]
        elif i == 8:
            if col9.button(genre_names[i]):
                genre = genre_names[i]
        elif i == 9:
            if col10.button(genre_names[i]):
                genre = genre_names[i]
        elif i == 10:
            if col11.button(genre_names[i]):
                genre = genre_names[i]

    st.write("##")
    with st.container():
        st.write("##### Select the year range")
        start_year, end_year = st.slider('', 1990, 2019, (2015, 2019))
    st.write("##")

    tracks_per_page = 12

    test_feat = get_test_feat(Emotion)

    uris, audios = n_neighbors_uri_audio(
        genre, start_year, end_year, test_feat)

    tracks = []
    for uri in uris:
        track = """<iframe src="https://open.spotify.com/embed/track/{}" width="260" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>""".format(
            uri)
        tracks.append(track)

    if 'previous_inputs' not in st.session_state:
        st.session_state['previous_inputs'] = [
            genre, start_year, end_year]

    current_inputs = [genre, start_year, end_year] + test_feat
    if current_inputs != st.session_state['previous_inputs']:
        if 'start_track_i' in st.session_state:
            st.session_state['start_track_i'] = 0
        st.session_state['previous_inputs'] = current_inputs

    if 'start_track_i' not in st.session_state:
        st.session_state['start_track_i'] = 0

    with st.container():
        col1, col2, col3 = st.columns([2, 1, 2])
        if st.button("Recommend More Songs"):
            if st.session_state['start_track_i'] < len(tracks):
                st.session_state['start_track_i'] += tracks_per_page

        current_tracks = tracks[st.session_state['start_track_i']
            : st.session_state['start_track_i'] + tracks_per_page]
        current_audios = audios[st.session_state['start_track_i']
            : st.session_state['start_track_i'] + tracks_per_page]
        if st.session_state['start_track_i'] < len(tracks):
            for i, (track, audio) in enumerate(zip(current_tracks, current_audios)):
                if i % 2 == 0:
                    with col1:
                        components.html(
                            track,
                            height=400,
                        )
                else:
                    with col3:
                        components.html(
                            track,
                            height=400,
                        )
        else:
            st.write("No songs left to recommend")


if __name__ == '__main__':
    Song_Recommendation("Neutral")