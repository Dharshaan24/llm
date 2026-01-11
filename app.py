import streamlit as st

# ✅ THIS MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Ticket Classification AI",
    layout="centered"
)

# -------------------------------------------------
# ALL OTHER IMPORTS (NO st.* ABOVE THIS LINE)
# -------------------------------------------------
import re
import pickle
import google.generativeai as genai

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# -------------------------------------------------
# CONFIG
# -------------------------------------------------
MAX_LEN = 150


# -------------------------------------------------
# LOAD ARTIFACTS (AFTER set_page_config)
# -------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = load_model("ticket_lstm_model.h5")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    return model, tokenizer, label_encoder


model, tokenizer, label_encoder = load_artifacts()


# -------------------------------------------------
# GEMINI CONFIG
# -------------------------------------------------
genai.configure(api_key="YOUR_GEMINI_API_KEY")
gemini_model = genai.GenerativeModel("gemini-1.5-flash")


# -------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text


def predict_queue(ticket_text):
    text = clean_text(ticket_text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    pred = model.predict(padded).argmax()
    return label_encoder.inverse_transform([pred])[0]


def generate_reply(ticket_text, predicted_queue):
    try:
        prompt = f"""
You are a customer support assistant.

Customer Ticket:
{ticket_text}

Department:
{predicted_queue}

Write a polite acknowledgment in 2–3 sentences.
Do not promise resolution.
"""
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception:
        return (
            "Thank you for contacting us. "
            "Your request has been forwarded to our support team."
        )


# -------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------
st.title("🎫 Automatic Ticket Classification")

ticket_text = st.text_area(
    "Enter Customer Ticket",
    height=150,
    placeholder="I was charged twice for my subscription"
)

if st.button("Classify Ticket"):
    if ticket_text.strip() == "":
        st.warning("Please enter a ticket.")
    else:
        with st.spinner("Processing..."):
            queue = predict_queue(ticket_text)
            reply = generate_reply(ticket_text, queue)

        st.success("Done!")
        st.subheader("Predicted Queue")
        st.info(queue)

        st.subheader("Auto Reply")
        st.write(reply)
