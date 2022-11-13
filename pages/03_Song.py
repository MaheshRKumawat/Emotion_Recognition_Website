import streamlit as st
import streamlit.components.v1 as components
from Resources.Recommendation.utils import *

st.title("Song Recommendation")
st.write("Based on your mood let us recommend you a song from Spotify")
st.markdown("##")

with st.container():
    col1, col2, col3, col4 = st.columns((2, 0.5, 0.5, 0.5))
    with col3:
        st.markdown("***Choose your genre:***")
        genre = st.radio(
            "",
            genre_names, index=genre_names.index("Pop"))
    with col1:
        st.markdown("***Choose features to customize:***")
        start_year, end_year = st.slider(
            'Select the year range',
            1990, 2019, (2015, 2019)
        )

tracks_per_page = 12
# test_feat = [acousticness, danceability, energy, instrumentalness, valence, tempo]
test_feat = [get_random_number(), get_random_number(), get_random_number(
), get_random_number(), get_random_number(), get_random_number()]
uris, audios = n_neighbors_uri_audio(genre, start_year, end_year, test_feat)

tracks = []
for uri in uris:
    track = """<iframe src="https://open.spotify.com/embed/track/{}" width="260" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>""".format(
        uri)
    tracks.append(track)

if 'previous_inputs' not in st.session_state:
    st.session_state['previous_inputs'] = [
        genre, start_year, end_year] + test_feat

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

    current_tracks = tracks[st.session_state['start_track_i']                            : st.session_state['start_track_i'] + tracks_per_page]
    current_audios = audios[st.session_state['start_track_i']                            : st.session_state['start_track_i'] + tracks_per_page]
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
