
import streamlit as st
import joblib

model = joblib.load('spam_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

st.title("SMS Spam Detection System")

st.write("Enter a message to check whether it is Spam or Normal.")

message = st.text_area("Enter your SMS message:")

if st.button("Predict"):

    if message.strip():

        message_vector = tfidf.transform([message])

        prediction = model.predict(message_vector)[0]

        if prediction == 1:
            st.error("This message is SPAM")
        else:
            st.success("This message is NORMAL")

    else:
        st.warning("Please enter a message.")
