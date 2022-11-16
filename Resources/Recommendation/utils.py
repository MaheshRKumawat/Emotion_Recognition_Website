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


'''
Youtube Channels to recommend

If Emotion is happy reccomend channels related to news, education, entertainment, etc.

If Emotion is sad/angry recommend channels related to comedy, entertainment, etc.

If Emotion is neutral/fear recommend channels related to all the above emotions


Educational Channels:
Ted-Ed
Kurzgesagt
Minute Physics
Veritasium
Vsauce
SciShow
PBS Space Time
National Geographic
RealLifeLore
MinuteEarth

Entertainment Channels:
MrBeast
PewDiePie
Markiplier
Jacksepticeye
CaseyNeistat
Ryan Trahan

Comedy Channels:
The Tonight Show Starring Jimmy Fallon
The Daily Show with Trevor Noah
The Ellen DeGeneres Show
Jimmy Kimmel Live
Saturday Night Live
The Daily Show with Trevor Noah
Kapil Sharma
Tanmay Bhat
Ashish Chanchlani
Amit Bhadana
BB Ki Vines

News Channels:
BBC News
CNN
Al Jazeera English
DW News
France 24 English
RT
CBS News
Fox News
NDTV
MSNBC
ABC News

'''

# channels = 'Emotion-Recognition-using-Text-with-Emojis-and-Speech/Video/channels.csv'

# # df 0 to 10 are educational channels
# # df 10 to 16 are entertainment channels
# # df 16 to 27 are comedy channels
# # df 27 to 38 are news channels

# df = pd.read_csv(channels)

# def get_channel(emotion):
#     #If Emotion is happy reccomend channels related to news, education, entertainment, etc.
#     if emotion == "Happy":
#         return df[27:38] + df[0:10] + df[10:16]
#     #If Emotion is sad/angry recommend channels related to comedy, entertainment, etc.
#     elif emotion == "Sad" or emotion == "Angry":
#         return df[16:27] + df[10:16]
#     #If Emotion is neutral/fear recommend channels related to all the above emotions
#     else:
#         return df[0:38]


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

# def get_channel(emotion):
#     #If Emotion is happy reccomend channels related to news, education, entertainment, etc.
#     if emotion == "Happy":
#         return pd.concat([pd.read_csv(news), pd.read_csv(education), pd.read_csv(entertainment)])
#     #If Emotion is sad/angry recommend channels related to comedy, entertainment, etc.
#     elif emotion == "Sad" or emotion == "Angry":
#         return pd.concat([pd.read_csv(comedy), pd.read_csv(entertainment)])
#     #If Emotion is neutral/fear recommend channels related to all the above emotions
#     else:
#         return pd.concat([pd.read_csv(comedy), pd.read_csv(education), pd.read_csv(entertainment), pd.read_csv(news)])

# def load_youtube_data():
#     comedy_df = pd.read_csv(comedy)
#     education_df = pd.read_csv(education)
#     entertainment_df = pd.read_csv(entertainment)
#     news_df = pd.read_csv(news)
#     return comedy_df, education_df, entertainment_df, news_df

# comedy_df, education_df, entertainment_df, news_df = load_youtube_data()

# def get_youtube_data(emotion):
#     if emotion == "Happy":
#         # combine df and send
#         # send key as channel name and value as link
#         df = pd.concat([comedy_df, education_df, entertainment_df, news_df])
#         return df
#     elif emotion == "Sad":
#         # combine df and send
#         df = pd.concat([comedy_df, entertainment_df])
#         return df
#     elif emotion == "Angry":
#         # combine df and send
#         df = pd.concat([comedy_df, entertainment_df])
#         return df
#     elif emotion == "Fear":
#         # combine df and send
#         df = pd.concat([comedy_df, entertainment_df, news_df, education_df])
#         return df
#     else:
#         df = pd.concat([entertainment_df, education_df, comedy_df, news_df])
#         return df

# # Key is Channel Name and Value is link to the channel
# youtube_channels = pd.read_csv(youtube_channels_path)
# youtube_channels = youtube_channels.set_index('Channel Name').T.to_dict('list')

# # List of all the channels
# channels = list(youtube_channels.keys())

# def get_channels(channel_type):
#     if channel_type == 'Educational':
#         # Channels 0 to 11
#         return channels[:12]
#     elif channel_type == 'Entertainment':
#         # Channels 12 to 16
#         return channels[12:17]
#     elif channel_type == 'Comedy':
#         # Channels 17 to 26
#         return channels[17:27]
#     elif channel_type == 'News':
#         # Channels 27 to 37
#         return channels[27:38]

# def get_channel_link(channel_name):
#     return youtube_channels[channel_name][1]

# def get_channel(emotion):
#     #If Emotion is happy reccomend channels related to news, education, entertainment, etc.
#     if emotion == "Happy":
#         return get_channels('Educational') + get_channels('Entertainment') + get_channels('News')
#     #If Emotion is sad/angry recommend channels related to comedy, entertainment, etc.
#     elif emotion == "Sad" or emotion == "Angry":
#         return get_channels('Comedy') + get_channels('Entertainment')
#     #If Emotion is neutral/fear recommend channels related to all the above emotions
#     else:
#         return get_channels('Educational') + get_channels('Entertainment') + get_channels('Comedy') + get_channels('News')
