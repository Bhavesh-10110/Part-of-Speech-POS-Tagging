# ===========================================================
# RNN Practical (Many-to-Many)
# Part-of-Speech (POS) Tagging using Simple RNN
# Streamlit Frontend
# ===========================================================

import os
import ast
import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Embedding,
    SimpleRNN,
    Dense,
    TimeDistributed
)

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical


# ===========================================================
# Constants
# ===========================================================

MODEL = "pos_rnn.keras"

WORD_TOKENIZER = "word_tokenizer.pkl"
TAG_TOKENIZER = "tag_tokenizer.pkl"

MAX_WORDS = 20000
MAX_LEN = 50

DATASET_PATH = "ner.csv"


# ===========================================================
# Train Model
# (Logic unchanged from the original script, only prints are
#  mirrored into the Streamlit UI via st.write / st.pyplot)
# ===========================================================

def train_model(status=None, log=None):
    """
    Trains the Simple RNN POS tagger.

    status : a st.status(...) context object used to show progress
    log    : a st.empty() placeholder used to stream text logs
    """

    def emit(msg):
        print(msg)
        if log is not None:
            log.markdown(msg)

    emit("Loading Dataset...")

    # -------------------------------------------------------
    # Load Dataset
    # -------------------------------------------------------

    df = pd.read_csv(DATASET_PATH, encoding="latin1")

    emit("Dataset Loaded Successfully!")

    # -------------------------------------------------------
    # Data Exploration
    # -------------------------------------------------------

    if status is not None:
        status.write("Exploring dataset...")

    with st.expander("Dataset Preview & Info", expanded=False):
        st.write("**First 5 Rows**")
        st.dataframe(df.head())

        st.write("**Dataset Shape:**", df.shape)
        st.write("**Columns:**", list(df.columns))
        st.write("**Missing Values**")
        st.write(df.isnull().sum())

    # -------------------------------------------------------
    # Data Cleaning
    # -------------------------------------------------------

    if status is not None:
        status.write("Cleaning dataset...")

    emit("Cleaning Dataset...")

    df["Sentence"] = df["Sentence"].apply(lambda x: x.split() if isinstance(x, str) else [])
    df["POS"] = df["POS"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

    # Filter out length mismatches to ensure perfect token-tag alignment
    df = df[df["Sentence"].apply(len) == df["POS"].apply(len)]

    emit("Cleaning Completed!")

    # -------------------------------------------------------
    # Sentence Grouping
    # -------------------------------------------------------

    sentences = df["Sentence"].tolist()
    pos_tags = df["POS"].tolist()

    emit(f"Number of Sentences : {len(sentences)}")

    with st.expander("Sample Sentence & Tags", expanded=False):
        st.write("**First Sentence:**", sentences[0])
        st.write("**First POS Tags:**", pos_tags[0])

    # -------------------------------------------------------
    # Word Encoding
    # -------------------------------------------------------

    if status is not None:
        status.write("Encoding words...")

    emit("Encoding Words...")

    word_tokenizer = Tokenizer(
        num_words=MAX_WORDS,
        lower=True,
        oov_token="<OOV>",
        filters=""
    )

    word_tokenizer.fit_on_texts(sentences)

    X = word_tokenizer.texts_to_sequences(sentences)

    emit(f"Vocabulary Size : {len(word_tokenizer.word_index)}")

    with open(WORD_TOKENIZER, "wb") as f:
        pickle.dump(word_tokenizer, f)

    emit("Word Tokenizer Saved Successfully!")

    # -------------------------------------------------------
    # Tag Encoding
    # -------------------------------------------------------

    if status is not None:
        status.write("Encoding POS tags...")

    emit("Encoding POS Tags...")

    unique_tags = sorted(
        list(set(tag for sentence in pos_tags for tag in sentence))
    )

    tag2idx = {
        tag: idx + 1
        for idx, tag in enumerate(unique_tags)
    }

    idx2tag = {
        idx: tag
        for tag, idx in tag2idx.items()
    }

    y = []

    for sentence in pos_tags:
        encoded_sentence = []
        for tag in sentence:
            encoded_sentence.append(tag2idx[tag])
        y.append(encoded_sentence)

    emit(f"Number of POS Tags : {len(unique_tags)}")

    with open(TAG_TOKENIZER, "wb") as f:
        pickle.dump(
            {
                "tag2idx": tag2idx,
                "idx2tag": idx2tag
            },
            f
        )

    emit("POS Tag Dictionary Saved Successfully!")

    # -------------------------------------------------------
    # Padding
    # -------------------------------------------------------

    if status is not None:
        status.write("Padding sequences...")

    emit("Padding Sequences...")

    X = pad_sequences(
        X,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    y = pad_sequences(
        y,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    emit(f"Shape of X : {X.shape} &nbsp;&nbsp; Shape of y : {y.shape}")

    y = to_categorical(
        y,
        num_classes=len(unique_tags) + 1
    )

    emit(f"Shape after One-Hot Encoding : {y.shape}")

    # -------------------------------------------------------
    # Train-Test Split
    # -------------------------------------------------------

    if status is not None:
        status.write("Splitting dataset...")

    emit("Splitting Dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    emit(f"Training Samples : {X_train.shape[0]} &nbsp;&nbsp; Testing Samples : {X_test.shape[0]}")

    # -------------------------------------------------------
    # Build Simple RNN
    # -------------------------------------------------------

    if status is not None:
        status.write("Building model...")

    emit("Building Model...")

    model = Sequential()

    model.add(
        Embedding(
            input_dim=MAX_WORDS,
            output_dim=128,
            input_length=MAX_LEN
        )
    )

    model.add(
        SimpleRNN(
            128,
            return_sequences=True
        )
    )

    model.add(
        TimeDistributed(
            Dense(
                len(unique_tags) + 1,
                activation="softmax"
            )
        )
    )

    summary_lines = []
    model.summary(print_fn=lambda line: summary_lines.append(line))

    with st.expander("Model Summary", expanded=False):
        st.code("\n".join(summary_lines))

    # -------------------------------------------------------
    # Compile Model
    # -------------------------------------------------------

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # -------------------------------------------------------
    # Train Model
    # -------------------------------------------------------

    if status is not None:
        status.write("Training model... this may take a while")

    emit("Training Model...")

    epochs = st.session_state.get("epochs", 10)
    batch_size = st.session_state.get("batch_size", 32)

    progress_bar = st.progress(0, text="Starting training...")

    class StreamlitProgress(__import__("tensorflow").keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            frac = (epoch + 1) / epochs
            progress_bar.progress(
                min(frac, 1.0),
                text=(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"loss: {logs.get('loss', 0):.4f} - "
                    f"accuracy: {logs.get('accuracy', 0):.4f} - "
                    f"val_loss: {logs.get('val_loss', 0):.4f} - "
                    f"val_accuracy: {logs.get('val_accuracy', 0):.4f}"
                )
            )

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[StreamlitProgress()],
        verbose=0
    )

    # -------------------------------------------------------
    # Save Model
    # -------------------------------------------------------

    model.save(MODEL)

    emit("Model Saved Successfully!")

    # -------------------------------------------------------
    # Evaluate Model
    # -------------------------------------------------------

    if status is not None:
        status.write("Evaluating model...")

    emit("Evaluating Model...")

    loss, accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    st.metric("Test Loss", f"{loss:.4f}")
    st.metric("Test Accuracy", f"{accuracy * 100:.2f}%")

    # -------------------------------------------------------
    # Plot Accuracy & Loss
    # -------------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history.history["accuracy"])
    axes[0].plot(history.history["val_accuracy"])
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(["Train", "Validation"])

    axes[1].plot(history.history["loss"])
    axes[1].plot(history.history["val_loss"])
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(["Train", "Validation"])

    plt.tight_layout()

    st.pyplot(fig)

    return model, loss, accuracy


# ===========================================================
# Inference Helpers
# ===========================================================

@st.cache_resource(show_spinner=False)
def load_artifacts():
    """
    Loads the trained model + tokenizers from disk.
    Cached so repeated predictions don't reload from disk each time.
    """

    model = load_model(MODEL)

    with open(WORD_TOKENIZER, "rb") as f:
        word_tokenizer = pickle.load(f)

    with open(TAG_TOKENIZER, "rb") as f:
        tag_data = pickle.load(f)

    return model, word_tokenizer, tag_data["idx2tag"]


def predict_pos(sentence, model, word_tokenizer, idx2tag):
    """
    Takes a raw sentence string, tokenizes + pads it,
    runs it through the model, and returns a list of
    (word, predicted_tag) pairs for the real (non-padded) tokens.
    """

    # Use re.findall to split punctuation from words during inference
    words = re.findall(r"\w+|[^\w\s]", sentence)
    words = words[:MAX_LEN]  # Limit words to MAX_LEN to prevent IndexError

    sequence = word_tokenizer.texts_to_sequences([words])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    prediction = model.predict(padded, verbose=0)[0]

    predicted_ids = np.argmax(prediction, axis=-1)

    results = []

    for i, word in enumerate(words):
        tag_id = predicted_ids[i]
        tag = idx2tag.get(tag_id, "O")
        results.append((word, tag))

    return results


def get_tag_style(tag):
    """
    Returns a tuple of (background_color, border_color, text_color)
    for styling POS tag chips in the UI.
    """
    if tag.startswith("NN"):  # Nouns: Blue
        return "#eff6ff", "#bfdbfe", "#1e40af"
    elif tag.startswith("VB"):  # Verbs: Green
        return "#f0fdf4", "#bbf7d0", "#166534"
    elif tag.startswith("JJ"):  # Adjectives: Purple
        return "#faf5ff", "#e9d5ff", "#6b21a8"
    elif tag.startswith("RB") or tag == "WRB":  # Adverbs: Pink
        return "#fdf2f8", "#fbcfe8", "#9d174d"
    elif tag in ["PRP", "PRP$", "WP", "WP$"]:  # Pronouns: Teal
        return "#f0fdfa", "#99f6e4", "#0f766e"
    elif tag == "CD":  # Numbers: Orange
        return "#fff7ed", "#ffedd5", "#c2410c"
    elif tag in [".", ",", ":", "``", "''", "$", ";", "!", "?", "(", ")", "-", "--"]:  # Punctuation: Gray
        return "#f9fafb", "#e5e7eb", "#374151"
    else:  # Others: Slate/Gray
        return "#f8fafc", "#e2e8f0", "#475569"


def artifacts_available():
    return (
        os.path.exists(MODEL)
        and os.path.exists(WORD_TOKENIZER)
        and os.path.exists(TAG_TOKENIZER)
    )


# ===========================================================
# Streamlit Frontend
# ===========================================================

def main():

    st.set_page_config(
        page_title="POS Tagger - Simple RNN",
        page_icon="🏷️",
        layout="centered"
    )

    st.title("🏷️ Part-of-Speech Tagging")
    st.caption("Many-to-Many Simple RNN, built with Keras + Streamlit")

    tab_predict, tab_train = st.tabs(["🔍 Tag a Sentence", "⚙️ Train Model"])

    # -----------------------------------------------------
    # Tab: Predict
    # -----------------------------------------------------

    with tab_predict:

        st.subheader("Try the tagger")

        if not artifacts_available():
            st.warning(
                "No trained model found yet. Go to the "
                "**Train Model** tab first to train and save the model."
            )
        else:
            sentence = st.text_input(
                "Enter a sentence",
                placeholder="e.g. Barack Obama was born in Hawaii"
            )

            if st.button("Predict POS Tags", type="primary", use_container_width=True):
                if not sentence.strip():
                    st.error("Please enter a sentence first.")
                else:
                    with st.spinner("Loading model & predicting..."):
                        model, word_tokenizer, idx2tag = load_artifacts()
                        results = predict_pos(sentence, model, word_tokenizer, idx2tag)

                    st.success("Done!")

                    # Table view
                    result_df = pd.DataFrame(results, columns=["Word", "POS Tag"])
                    st.dataframe(result_df, use_container_width=True, hide_index=True)

                    # Inline chip / badge view
                    st.write("")
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

    # -----------------------------------------------------
    # Tab: Train
    # -----------------------------------------------------

    with tab_train:

        st.subheader("Train / Retrain the Model")

        st.write(
            f"Dataset expected at: `{DATASET_PATH}` "
            "(must contain `Sentence` and `POS` columns as "
            "string-encoded Python lists)."
        )

        if not os.path.exists(DATASET_PATH):
            st.error(f"Dataset file '{DATASET_PATH}' not found in the working directory.")
        else:
            st.success(f"Dataset file '{DATASET_PATH}' found.")

        col1, col2 = st.columns(2)

        with col1:
            epochs = st.number_input("Epochs", min_value=1, max_value=100, value=10, step=1)

        with col2:
            batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)

        st.session_state["epochs"] = epochs
        st.session_state["batch_size"] = batch_size

        if artifacts_available():
            st.info("A trained model already exists. Training again will overwrite it.")

        if st.button(
            "Start Training",
            type="primary",
            use_container_width=True,
            disabled=not os.path.exists(DATASET_PATH)
        ):
            log_placeholder = st.empty()

            with st.status("Training in progress...", expanded=True) as status:
                try:
                    model, loss, accuracy = train_model(status=status, log=log_placeholder)
                    status.update(label="Training complete!", state="complete")

                    # Clear cached artifacts so the Predict tab picks up the new model
                    load_artifacts.clear()

                except Exception as e:
                    status.update(label="Training failed", state="error")
                    st.exception(e)


if __name__ == "__main__":
    main()