import os
import time
import streamlit as st
# import streamlit_google_oauth as oauth
# from dotenv import load_dotenv
# from utils.preprocess import *

# load_dotenv()

# username = ""
# login_start_time = time.time()

# login_info = oauth.login(
#     client_id=os.environ['CLIENT_ID'],
#     client_secret=os.environ['CLIENT_SECRET_KEY'],
#     redirect_uri=os.environ['CLIENT_REDIRECT_URI'],
#     login_button_text="Continue with Google",
#     logout_button_text="Logout",
# )

# if login_info:
#     login_start_time = time.time()
#     user_id, user_email = login_info
#     st.write(f"Welcome {user_email}")
#     username = user_email
# else:
#     st.write("Please login to continue")

st.set_page_config(page_title="Emotion Recognition", layout="wide")

st.header("Emotion Recognizer using Text and Speech")

st.write("## Capstone Project")
st.write('##')
st.write("## Speech Recognition uses 75 percent output from speecg model along with 25 percent from the transcribed text input fed to text model")
st.write('##')
st.write("## Here we implemented 3 types of text models: ")
st.write("### 1. Plain RoBerta Model")
st.write("### 2. Plain Bert Model")
st.write("### 3. Plain Bi-LSTM Model")
st.write("### 4. Hybrid Model (RoBerta + LSTM + CNN)")