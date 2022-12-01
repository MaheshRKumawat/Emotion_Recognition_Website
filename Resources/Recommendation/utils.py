import random
import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors

path = "Emotion-Recognition-using-Text-with-Emojis-and-Speech/Song/filtered_track_df.csv"

genre_names = ['Dance Pop', 'Electronic', 'Electropop', 'Hip Hop',
               'Jazz', 'K-pop', 'Latin', 'Pop', 'Pop Rap', 'R&B', 'Rock']
               
audio_feats = ["acousticness", "danceability",
               "energy", "instrumentalness", "valence", "tempo"]

def get_test_feat(emotion):
    if emotion == "Happy":
        acousticness = 0.2 - round(random.uniform(0.01, 0.09), 2)
        danceability = 0.7 + round(random.uniform(0.01, 0.09), 2)
        energy = 0.7 + round(random.uniform(0.01, 0.09), 2)
        instrumentalness = 0.1 - round(random.uniform(0.01, 0.09), 2)
        valence = 0.7 + round(random.uniform(0.01, 0.09), 2)
        tempo = 120 + random.randint(0, 20)

    # If user emotion is sad then we will recommend happy with high valence and energy
    elif emotion == "Sad":
        acousticness = 0.2 - round(random.uniform(0.01, 0.09), 2)
        danceability = 0.7 - round(random.uniform(0.01, 0.19), 2)
        energy = 0.6 - round(random.uniform(0.01, 0.09), 2)
        instrumentalness = 0.1 - round(random.uniform(0.01, 0.09), 2)
        valence = 0.9 + round(random.uniform(0.01, 0.09), 2)
        tempo = 120 - random.randint(0, 20)

    # If user emotion is angry then we will recommend relaxing songs with high valence and low energy
    elif emotion == "Angry":
        acousticness = 0.2 - round(random.uniform(0.01, 0.09), 2)
        danceability = 0.7 - round(random.uniform(0.01, 0.09), 2)
        energy = 0.4 - round(random.uniform(0.01, 0.09), 2)
        instrumentalness = 0.1 - round(random.uniform(0.01, 0.09), 2)
        valence = 0.9 + round(random.uniform(0.01, 0.09), 2)
        tempo = 120 - random.randint(0, 20)

    # If user emotion is fear then we will recommend songs with high valence and low energy
    elif emotion == "Fear":
        acousticness = 0.2 - round(random.uniform(0.01, 0.09), 2)
        danceability = 0.7 - round(random.uniform(0.01, 0.09), 2)
        energy = 0.4 - round(random.uniform(0.01, 0.09), 2)
        instrumentalness = 0.1 - round(random.uniform(0.01, 0.09), 2)
        valence = 0.9 + round(random.uniform(0.01, 0.09), 2)
        tempo = 120 - random.randint(0, 20)

    # If user emotion is neutral then we will recommend songs with high valence and energy
    else:
        acousticness = 0.2 - round(random.uniform(0.01, 0.09), 2)
        danceability = 0.7 + round(random.uniform(0.01, 0.09), 2)
        energy = 0.7 + round(random.uniform(0.01, 0.09), 2)
        instrumentalness = 0.1 - round(random.uniform(0.01, 0.09), 2)
        valence = 0.7 + round(random.uniform(0.01, 0.09), 2)
        tempo = 120 + random.randint(0, 20)

    test_feat = [acousticness, danceability, energy, instrumentalness, valence, tempo]

    return test_feat

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


comedy = 'Emotion-Recognition-using-Text-with-Emojis-and-Speech/Video/comedy.csv'
education = 'Emotion-Recognition-using-Text-with-Emojis-and-Speech/Video/educational.csv'
entertainment = 'Emotion-Recognition-using-Text-with-Emojis-and-Speech/Video/entertainment.csv'
news = 'Emotion-Recognition-using-Text-with-Emojis-and-Speech/Video/news.csv'

df_comedy = pd.read_csv(comedy)
df_education = pd.read_csv(education)
df_entertainment = pd.read_csv(entertainment)
df_news = pd.read_csv(news)

def get_channel(emotion):
    if emotion == "Happy":
        return pd.concat([df_news, df_education, df_entertainment])
    elif emotion == "Sad" or emotion == "Angry":
        return pd.concat([df_comedy, df_entertainment])
    else:
        return pd.concat([df_comedy, df_education, df_entertainment, df_news])