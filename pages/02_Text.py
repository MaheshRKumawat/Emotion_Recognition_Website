import Home
import streamlit as st
# from utils.predict import *
# from utils.demojize import *

db = Home.db
user_id = st.session_state.get('user_id', None)
username = st.session_state.get('username', None)
text_db = db["text"]

st.markdown(
    '''<style>.css-1egvi7u {margin-top: -3rem;}</style>''', unsafe_allow_html=True)
st.markdown(
    '''<style>.stAudio {height: 45px;}</style>''', unsafe_allow_html=True)
st.markdown(
    '''<style>.css-v37k9u a {color: #ff4c4b;}</style>''', unsafe_allow_html=True)
st.markdown(
    '''<style>.css-nlntq9 a {color: #ff4c4b;}</style>''', unsafe_allow_html=True)


def text_app():
    st.title('Text Emotion Recognizer')
    st.write('\n\n')
    st.write('''
    ## Enter the text below to analyze the emotion
    ''')
    text = st.text_area("Enter text here")
    Analyze = st.button("Analyze")
    if Analyze:
        if text != '':
            st.write('\n\n')
            # # demojize start
            # text = remove_emoji(text)
            # print("Demojized text :", text)
            # # demojize end
            # roberta_emotion = roberta_predict(text)
            # hybrid_emotion = hybrid_predict_text(text)
            # bilstm_emotion = bilstm_predict_text(text)
            # bert_emotion = bert_predict_text(text)
            roberta_emotion = "Sad"
            hybrid_emotion = "Sad"
            bilstm_emotion = "Sad"
            bert_emotion = "Sad"
            st.write('#')
            placeholder = st.empty()
            placeholder.info(placeholder_value(hybrid_emotion, 'Hybrid'))
            st.write("##")
            placeholder = st.empty()
            placeholder.info(placeholder_value(roberta_emotion, 'RoBerta'))
            st.write('#')
            placeholder = st.empty()
            placeholder.info(placeholder_value(bilstm_emotion, "Bi-LSTM"))
            st.write('#')
            placeholder = st.empty()
            placeholder.info(placeholder_value(bert_emotion, "BERT"))

            st.write("##")
            st.write(
                "We would like to recommend you songs or videos to cheer you up!")
            st.write("Click on the Song or Video Page at the left to get started!")
            st.write("##")
            final_emotion = hybrid_emotion

            st.session_state.emotion = final_emotion
            st.session_state.speech = False
            st.session_state.text = True
            st.session_state.feedback = False
            st.session_state.text_id = text_db.insert_one({"user_id": user_id, "username": username, "text": text, "predicted_emotion": st.session_state.get('emotion') , "actual_emotion": st.session_state.get('emotion'), "feedback": None}).inserted_id
            st.write("##")
            st.write("We would like to recommend you songs or videos to cheer you up!")
            st.write("Click on the Song or Video Page at the left to get started!")

        else:
            st.write('\n\n')
            st.write('''## Please enter the text to analyze the emotion''')


def placeholder_value(emotion, model_name):
    text = ''
    if emotion == 'Happy':
        text = model_name + \
            " model predicts that you are Happy! Here are some songs to cheer you more!"
    elif emotion == 'Neutral':
        text = model_name + \
            " model predicts that you are Neutral! Here are some songs to cheer you up!"
    elif emotion == 'Angry' or emotion == 'Sad':
        text = model_name + " model predicts that you are " + \
            emotion + "! Here are some songs to make you feel realxed!"
    else:
        text = model_name + " model predicts that you are " + emotion + \
            "! Here are some songs we recommend to relax you a bit!"
    return text


if __name__ == '__main__':
    user_id = st.session_state.get('user_id', None)
    if user_id is None:
        st.write("Please login to continue")
    else:
        text_app()