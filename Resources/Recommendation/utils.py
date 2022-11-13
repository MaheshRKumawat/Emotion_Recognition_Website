import random
import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors

path = "Emotion-Recognition-using-Text-with-Emojis-and-Speech/Song/filtered_track_df.csv"

genre_names = ['Dance Pop', 'Electronic', 'Electropop', 'Hip Hop',
               'Jazz', 'K-pop', 'Latin', 'Pop', 'Pop Rap', 'R&B', 'Rock']
               
audio_feats = ["acousticness", "danceability",
               "energy", "instrumentalness", "valence", "tempo"]

@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv(path)
    df['genres'] = df.genres.apply(
        lambda x: [i[1:-1] for i in str(x)[1:-1].split(", ")])
    exploded_track_df = df.explode("genres")
    return exploded_track_df

exploded_track_df = load_data()

def n_neighbors_uri_audio(genre, start_year, end_year, test_feat):
    genre = genre.lower()
    genre_data = exploded_track_df[(exploded_track_df["genres"] == genre) & (
        exploded_track_df["release_year"] >= start_year) & (exploded_track_df["release_year"] <= end_year)]
    genre_data = genre_data.sort_values(by='popularity', ascending=False)[:500]

    neigh = NearestNeighbors()
    neigh.fit(genre_data[audio_feats].to_numpy())

    n_neighbors = neigh.kneighbors([test_feat], n_neighbors=len(
        genre_data), return_distance=False)[0]

    uris = genre_data.iloc[n_neighbors]["uri"].tolist()
    audios = genre_data.iloc[n_neighbors][audio_feats].to_numpy()
    return uris, audios

def get_random_number():
    return round(random.uniform(0.01, 0.99), 2)
