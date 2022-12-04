import streamlit as st
import pymongo

client = pymongo.MongoClient(st.secrets["mongo"]["host"])
db = client["erts_db"]
user_db = db["users"]

st.session_state.user_id = None
st.session_state.emotion = None

st.set_page_config(page_title="Emotion Recognition", layout="wide")


def select_signup():
    st.session_state.form = 'signup_form'


def user_update(name):
    st.session_state.username = name


def home_page():
    st.empty()
    st.header('Emotion Recognition Using Text with emoji and Speech')
    placeholder = st.empty()
    placeholder.info("1. RoBerta Model")
    placeholder = st.empty()
    placeholder.info("2. Bert Model")
    placeholder = st.empty()
    placeholder.info("3. Bi-LSTM Model")
    placeholder = st.empty()
    placeholder.info("4. Hybrid Model (RoBerta + LSTM + CNN)")


with st.container():

    # Initialize Session States.
    if 'username' not in st.session_state:
        st.session_state.username = ''

    if 'form' not in st.session_state:
        st.session_state.form = ''

    if st.session_state.username != '':
        st.write(f"You are logged in as {st.session_state.username.upper()}")
        home_page()

    # Initialize Sing In or Sign Up forms
    if st.session_state.form == 'signup_form' and st.session_state.username == '':

        signup_form = st.form(key='signup_form', clear_on_submit=True)
        new_username = signup_form.text_input(label='Enter Username*')
        new_user_email = signup_form.text_input(label='Enter Email Address*')
        new_user_pas = signup_form.text_input(
            label='Enter Password*', type='password')
        user_pas_conf = signup_form.text_input(
            label='Confirm Password*', type='password')
        note = signup_form.markdown('**required fields*')
        signup = signup_form.form_submit_button(label='Sign Up')

        if signup:
            if '' in [new_username, new_user_email, new_user_pas]:
                st.error('Some fields are missing')
            else:
                if user_db.find_one({'username': new_username}):
                    st.error('Username already exists')
                if user_db.find_one({'email': new_user_email}):
                    st.error('Email is already registered')
                else:
                    if new_user_pas != user_pas_conf:
                        st.error('Passwords do not match')
                    else:
                        user_update(new_username)
                        user_db.insert_one(
                            {'username': new_username, 'email': new_user_email, 'password': new_user_pas})
                        home_page()
                        # take user id from database
                        st.session_state.user_id = user_db.find_one(
                            {'username': new_username})['_id']
                        user_id = st.session_state.user_id
                        st.success('You have successfully registered!')
                        st.success(
                            f"You are logged in as {new_username.upper()}")
                        del new_user_pas, user_pas_conf

    elif st.session_state.username == '':
        login_form = st.form(key='signin_form', clear_on_submit=True)
        username = login_form.text_input(label='Enter Username')
        user_pas = login_form.text_input(
            label='Enter Password', type='password')

        if user_db.find_one({'username': username, 'password': user_pas}):
            login = login_form.form_submit_button(
                label='Sign In', on_click=user_update(username))
            if login:
                home_page()
                st.session_state.user_id = user_db.find_one(
                    {'username': username})['_id']
                user_id = st.session_state.user_id
                st.success(f"You are logged in as {username.upper()}")
                del user_pas
        else:
            login = login_form.form_submit_button(label='Sign In')
            if login:
                st.error(
                    "Username or Password is incorrect. Please try again or create an account.")
    else:
        logout = st.button(label='Log Out')
        if logout:
            st.session_state.user_id = None
            user_id = st.session_state.user_id
            st.empty()
            user_update('')
            st.session_state.form = ''

    # 'Create Account' button
    if st.session_state.username == "" and st.session_state.form != 'signup_form':
        signup_request = st.button('Create Account', on_click=select_signup)








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
