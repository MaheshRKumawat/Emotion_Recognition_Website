import os
import time
import webbrowser
import numpy as np
import streamlit as st
from io import BytesIO
from dotenv import load_dotenv
from keras.models import load_model
import streamlit_google_oauth as oauth
import streamlit.components.v1 as components

load_dotenv()   