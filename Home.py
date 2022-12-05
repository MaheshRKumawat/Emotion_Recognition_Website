import streamlit as st
import pymongo

client = pymongo.MongoClient(st.secrets["mongo"]["host"])
db = client["erts_db"]
user_db = db["users"]

if 'username' not in st.session_state:
    st.session_state.username = ''

if 'form' not in st.session_state:
    st.session_state.form = 'login_form'

if 'user_id' not in st.session_state:
    st.session_state.user_id = None

if 'emotion' not in st.session_state:
    st.session_state.emotion = None

st.set_page_config(page_title="Emotion Recognition", layout="wide")


def select_signup():
    st.session_state.form = 'signup_form'


def user_update(name):
    st.session_state.username = name


def home_page(username):
    placeholder = st.empty()
    placeholder.success(f"You are logged in as {username}")
    st.header('Emotion Recognition Using Text with emoji and Speech')
    placeholder = st.empty()
    placeholder.info("1. RoBerta Model")
    placeholder = st.empty()
    placeholder.info("2. Bert Model")
    placeholder = st.empty()
    placeholder.info("3. Bi-LSTM Model")
    placeholder = st.empty()
    placeholder.info("4. Hybrid Model (RoBerta + LSTM + CNN)")
    placeholder = st.empty()
    placeholder.info("5. MFCC + CNN Speech Model")

    # ask user to logout
    if st.button("Logout"):
        st.session_state.username = ''
        st.session_state.user_id = None
        st.session_state.form = 'login_form'
        st.experimental_rerun()


with st.container():

    if st.session_state.form == 'signup_form':
        st.header('Sign Up')
        st.subheader('Please enter your details')
        name = st.text_input('Name')
        email = st.text_input('Email')
        password = st.text_input('Password', type='password')
        confirm_password = st.text_input('Confirm Password', type='password')
        if st.button('Submit'):
            if password == confirm_password:
                user = {
                    'username': name,
                    'email': email,
                    'password': password
                }
                user_db.insert_one(user)
                st.session_state.form = ''
                st.session_state.username = name
                st.session_state.user_id = user['_id']
                home_page(st.session_state.username)
                st.experimental_rerun()
            else:
                st.error('Password does not match')
        
        st.subheader('Already have an account?')
        if st.button('Login'):
            st.session_state.form = 'login_form'
            st.experimental_rerun()

    elif st.session_state.form == 'login_form':
        st.header('Login')
        st.subheader('Please enter your details')
        username = st.text_input('Username')
        password = st.text_input('Password', type='password')
        if st.button('Submit'):
            user = user_db.find_one({'username': username, 'password': password})
            if user:
                st.session_state.form = ''
                st.session_state.username = username
                st.session_state.user_id = user['_id']
                # show home page
                home_page(st.session_state.username)
                st.experimental_rerun()
            else:
                st.error('Invalid username or password')
        # ask user to signup
        st.write("#")
        st.write('New user?')
        if st.button("Sign Up"):
            st.session_state.form = 'signup_form'
            st.experimental_rerun()

    elif st.session_state.form == '':
        home_page(st.session_state.username)