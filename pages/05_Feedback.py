import streamlit as st
import Home

db = Home.db
speech_db = db["speech"]
text_db = db["text"]
user_id = st.session_state.get('user_id', None)
speech_id = st.session_state.get('speech_id', None)
text_id = st.session_state.get('text_id', None)
speech = st.session_state.get('speech', None)
text = st.session_state.get('text', None)
username = st.session_state.get('username', None)
emotion = st.session_state.get('emotion', None)

if "feedback" not in st.session_state:
    st.session_state.feedback = False

if text == True:
    based_on = "Based on **`Text Model`**"
elif speech == True:
    based_on = "Based on **`Speech Model`**"

def update_db(Emotion, feedback):
    if speech_id != None and speech != False:
        speech_db.find_one_and_update({"user_id": user_id, "speech_id": speech_id}, {"$set": {
            "speech_id": speech_id, "user_id": user_id, "username": username, "predicted_emotion": st.session_state.get('emotion'), "actual_emotion": Emotion, "feedback": feedback}}, upsert=True)
    elif text_id != None and text != False:
        text_db.find_one_and_update(
            {"_id": text_id}, {"$set": {"actual_emotion": Emotion, "feedback": feedback}}, upsert=True)
    st.write("Updated!")

def feedback():
    if st.session_state.feedback == False:
        st.write(based_on)
        st.write("Your emotion was **`" + emotion + "`**")
        st.write("How was the reccomendation?")
        feedback = st.text_input("Enter your feedback")
        st.write("Was the emotion correctly predicted?")
        checkbox1 = st.checkbox("Yes")
        checkbox2 = st.checkbox("No")
        if checkbox1 == True:
            update_db(emotion, feedback)
            st.session_state.feedback = True
            st.write("Thank you for your feedback!")
        elif checkbox2 == True:
            st.write("Enter the correct emotion")
            Emotion = st.selectbox("Select your actual emotion", ["Happy", "Sad", "Angry", "Fear", "Neutral"])
            if st.button("Submit"):
                update_db(Emotion, feedback)
                st.session_state.feedback = True
                st.write("Thank you for your feedback!")
    else:
        st.write("You have already given feedback!")


if __name__ == '__main__':
    user_id = st.session_state.get('user_id', None)
    if user_id is None:
        st.write("Please login to continue")
    else:
        feedback()