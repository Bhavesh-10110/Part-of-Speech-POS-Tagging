# RNN Practical (Many-to-Many)
# Part-of-Speech (POS) Tagging using Simple RNN (Many-to-Many)
# Dataset :- ner.csv

# Import Libraries

import os
import ast
import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, TimeDistributed
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

MODEL = "pos_rnn.keras"
WORD_TOKENIZER = "word_tokenizer.pkl"
TAG_TOKENIZER = "tag_tokenizer.pkl"

MAX_WORDS = 20000
MAX_LEN = 50
DATASET_PATH = "ner.csv"

# train model

def train_model():
    print("Training model...")

    df = pd.read_csv(DATASET_PATH, encoding="latin1")

    # Clean Dataset
    df["Sentence"] = df["Sentence"].apply(lambda x: x.split() if isinstance(x, str) else [])
    df["POS"] = df["POS"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

    # Filter out length mismatches to ensure perfect token-tag alignment
    df = df[df["Sentence"].apply(len) == df["POS"].apply(len)]

    sentences = df["Sentence"].tolist()
    pos_tags = df["POS"].tolist()

    # Tokenizer
    word_tokenizer = Tokenizer(
        num_words=MAX_WORDS,
        lower=True,
        oov_token="<OOV>",
        filters=""
    )
    word_tokenizer.fit_on_texts(sentences)
    X = word_tokenizer.texts_to_sequences(sentences)

    # Save tokenizer
    with open(WORD_TOKENIZER, "wb") as f:
        pickle.dump(word_tokenizer, f)

    # Tag encoding
    unique_tags = sorted(list(set(tag for sentence in pos_tags for tag in sentence)))
    tag2idx = {tag: idx + 1 for idx, tag in enumerate(unique_tags)}
    idx2tag = {idx: tag for tag, idx in tag2idx.items()}

    y = []
    for sentence in pos_tags:
        y.append([tag2idx[tag] for tag in sentence])

    # Save tag tokenizer metadata
    with open(TAG_TOKENIZER, "wb") as f:
        pickle.dump({"tag2idx": tag2idx, "idx2tag": idx2tag}, f)

    # Pad sequences
    X = pad_sequences(X, maxlen=MAX_LEN, padding="post", truncating="post")
    y = pad_sequences(y, maxlen=MAX_LEN, padding="post", truncating="post")

    # Categorical targets
    y = to_categorical(y, num_classes=len(unique_tags) + 1)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # Build RNN Model
    model = Sequential()

    # Embedding layer
    model.add(Embedding(
        input_dim=MAX_WORDS,
        output_dim=128,
        input_length=MAX_LEN
    ))

    # Simple RNN Layer (return_sequences=True for many-to-many sequence tagging)
    model.add(SimpleRNN(
        128,
        return_sequences=True
    ))

    # TimeDistributed Output layer
    model.add(TimeDistributed(
        Dense(len(unique_tags) + 1, activation="softmax")
    ))

    model.summary()

    # Compile
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Train
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=10,
        batch_size=32
    )

    # Save the model
    model.save(MODEL)

    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test)
    print("\nAccuracy: ", accuracy)

    # Prediction metrics evaluation
    predictions = model.predict(X_test)
    pred_classes = np.argmax(predictions, axis=-1)
    true_classes = np.argmax(y_test, axis=-1)

    # Flatten sequences for evaluation (skipping padding/zero-tag values)
    flat_pred = []
    flat_true = []
    for p_seq, t_seq in zip(pred_classes, true_classes):
        for p_val, t_val in zip(p_seq, t_seq):
            if t_val != 0:
                flat_pred.append(p_val)
                flat_true.append(t_val)

    unique_val_tags = sorted(list(set(flat_true)))
    target_names = [idx2tag[i] for i in unique_val_tags]

    print("\nClassification Report:\n")
    print(classification_report(flat_true, flat_pred, labels=unique_val_tags, target_names=target_names))

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(flat_true, flat_pred))


# Predictions

def predict_pos(sentence):
    model = load_model(MODEL)

    with open(WORD_TOKENIZER, "rb") as f:
        word_tokenizer = pickle.load(f)

    with open(TAG_TOKENIZER, "rb") as f:
        tag_data = pickle.load(f)
        idx2tag = tag_data["idx2tag"]

    # Tokenize input text preserving punctuation
    words = re.findall(r"\w+|[^\w\s]", sentence)
    words = words[:MAX_LEN]

    sequence = word_tokenizer.texts_to_sequences([words])
    sequence = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post"
    )

    prediction = model.predict(sequence, verbose=0)[0]
    predicted_ids = np.argmax(prediction, axis=-1)

    results = []
    for i, word in enumerate(words):
        tag_id = predicted_ids[i]
        tag = idx2tag.get(tag_id, "O")
        results.append((word, tag))

    return results


def get_tag_style(tag):
    if tag.startswith("NN"):     # Nouns: Blue
        return "#eff6ff", "#bfdbfe", "#1e40af"
    elif tag.startswith("VB"):   # Verbs: Green
        return "#f0fdf4", "#bbf7d0", "#166534"
    elif tag.startswith("JJ"):   # Adjectives: Purple
        return "#faf5ff", "#e9d5ff", "#6b21a8"
    elif tag.startswith("RB") or tag == "WRB":  # Adverbs: Pink
        return "#fdf2f8", "#fbcfe8", "#9d174d"
    elif tag in ["PRP", "PRP$", "WP", "WP$"]:  # Pronouns: Teal
        return "#f0fdfa", "#99f6e4", "#0f766e"
    elif tag == "CD":            # Numbers: Orange
        return "#fff7ed", "#ffedd5", "#c2410c"
    elif tag in [".", ",", ":", "``", "''", "$", ";", "!", "?", "(", ")", "-", "--"]:  # Punctuation: Gray
        return "#f9fafb", "#e5e7eb", "#374151"
    else:                        # Others: Slate/Gray
        return "#f8fafc", "#e2e8f0", "#475569"


if not os.path.exists(MODEL):
    train_model()


# Streamlit App

st.title("Part-of-Speech (POS) Tagging using Simple RNN")
st.write("Many to Many RNN Example")

sentence = st.text_input("Enter a sentence", placeholder="Barack Obama was born in Hawaii")

if st.button("Predict"):
    if not sentence.strip():
        st.error("Please enter a sentence first.")
    else:
        results = predict_pos(sentence)

        # Display results as chips
        chips = []
        for w, t in results:
            bg_col, border_col, text_col = get_tag_style(t)
            chips.append(
                f"<span style='background:{bg_col};border:1px solid {border_col};"
                f"border-radius:8px;padding:4px 8px;margin:2px;display:inline-block;"
                f"font-size:14px;'><b style='color:#1e293b;'>{w}</b> "
                f"<span style='color:{text_col};font-weight:bold;'>[{t}]</span></span>"
            )
        chip_html = " ".join(chips)
        st.markdown(chip_html, unsafe_allow_html=True)