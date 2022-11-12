# authorization.py
import tekore as tk
from dotenv import load_dotenv
import os
import streamlit as st
from streamlit_webrtc import webrtc_streamer

load_dotenv()

# spotify_login_btn = st.button("Login into Spotify to listen songs")
# st.write(spotify_login_btn)


def auth():
    client_id = os.environ['SP_CLIENT_ID']
    secret_key = os.environ['SP_SECRET_KEY']
    token = tk.request_client_token(client_id, secret_key)
    return tk.Spotify(token)


if st.button("Login into Spotify to listen songs"):

    token_recv = auth()

    if token_recv is not None:
        validate = token_recv
        st.write("User logged into spotify !!")
        st.write("Details", {token_recv})

    else:
        st.write("Invalid Credentials !!")
