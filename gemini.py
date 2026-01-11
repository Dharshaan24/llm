# =========================================================
# AUTOMATIC TICKET CLASSIFICATION + GEMINI AUTO REPLY
# FINAL STABLE VERSION (FREE TIER SAFE)
# =========================================================

# -------------------------
# STEP 0: INSTALL (RUN ONCE)
# -------------------------
# pip install datasets tensorflow scikit-learn pandas google-generativeai


# -------------------------
# STEP 1: IMPORT LIBRARIES
# -------------------------
from datasets import load_dataset
import pandas as pd
import re
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import google.generativeai as genai


# -------------------------
# STEP 2: LOAD DATASET
# -------------------------
dataset = load_dataset("Tobi-Bueck/customer-support-tickets")
df = dataset["train"].to_pandas()
df = df[["body", "queue"]]


# -------------------------
# STEP 3: REDUCE CLASSES (IMPORTANT)
# -------------------------
def simplify_queue(q):
    q = q.lower()
    if "billing" in q or "payment" in q:
        return "Billing"
    elif "technical" in q or "it" in q or "software" in q:
        return "Technical Support"
    elif "return" in q or "exchange" in q:
        return "Returns"
    elif "account" in q:
        return "Account"
    else:
        return "General Support"

df["queue"] = df["queue"].apply(simplify_queue)


# -------------------------
# STEP 4: TEXT CLEANING
# -------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text

df["clean_body"] = df["body"].astype(str).apply(clean_text)


# -------------------------
# STEP 5: TOKENIZATION
# -------------------------
tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
tokenizer.fit_on_texts(df["clean_body"])
sequences = tokenizer.texts_to_sequences(df["clean_body"])


# -------------------------
# STEP 6: PADDING
# -------------------------
MAX_LEN = 150

X = pad_sequences(
    sequences,
    maxlen=MAX_LEN,
    padding="post",
    truncating="post"
)


# -------------------------
# STEP 7: LABEL ENCODING
# -------------------------
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["queue"])

print("Classes:", label_encoder.classes_)


# -------------------------
# STEP 8: TRAIN / VALIDATION SPLIT
# -------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -------------------------
# STEP 9: BUILD LSTM MODEL
# -------------------------
model = Sequential([
    Embedding(input_dim=20000, output_dim=128),
    LSTM(128),
    Dropout(0.3),
    Dense(len(label_encoder.classes_), activation="softmax")
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()


# -------------------------
# STEP 10: TRAIN MODEL
# -------------------------
model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=6,
    batch_size=64
)


# -------------------------
# STEP 11: EVALUATE MODEL
# -------------------------
y_pred = model.predict(X_val).argmax(axis=1)

print(
    classification_report(
        y_val,
        y_pred,
        target_names=label_encoder.classes_
    )
)


# -------------------------
# STEP 12: QUEUE PREDICTION FUNCTION
# -------------------------
def predict_queue(ticket_text):
    text = clean_text(ticket_text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    pred = model.predict(padded).argmax()
    return label_encoder.inverse_transform([pred])[0]


# -------------------------
# STEP 13: GEMINI API CONFIG (STABLE)
# -------------------------
# 🔴 REPLACE WITH YOUR API KEY
genai.configure(api_key="AIzaSyAQiv09Blo3tYpsAoko9HXU9TS5ymwOWRk")

gemini_model = genai.GenerativeModel("gemini-1.5-flash")


# -------------------------
# STEP 14: PROMPT CREATION
# -------------------------
def create_prompt(ticket_text, predicted_queue):
    return f"""
You are a customer support assistant.

Customer Ticket:
{ticket_text}

Department:
{predicted_queue}

Write a polite and professional acknowledgment.
Do NOT promise resolution.
Keep it short (2–3 sentences).
"""


# -------------------------
# STEP 15: GENERATE REPLY (WITH FALLBACK)
# -------------------------
def generate_reply(ticket_text, predicted_queue):
    try:
        prompt = create_prompt(ticket_text, predicted_queue)
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception:
        return (
            "Thank you for contacting us. "
            "Your request has been forwarded to the appropriate support team."
        )


# -------------------------
# STEP 16: END-TO-END TEST
# -------------------------
ticket = "I was charged twice for my subscription last month"

predicted_queue = predict_queue(ticket)
reply = generate_reply(ticket, predicted_queue)

print("\n==============================")
print("TICKET:", ticket)
print("PREDICTED QUEUE:", predicted_queue)
print("AUTO REPLY:\n", reply)
print("==============================")



# ===============================
# SAVE MODEL & OBJECTS (RUN ONCE)
# ===============================

model.save("ticket_lstm_model.h5")

import pickle

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("Model and files saved successfully!")