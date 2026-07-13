# Part-of-Speech (POS) Tagging using Simple RNN

An interactive Part-of-Speech (POS) Tagger application powered by a Many-to-Many Simple Recurrent Neural Network (RNN) built with Keras and TensorFlow. The app is wrapped in a premium Streamlit web frontend for real-time inference and programmatic training visualization.

---

## 🏷️ Features

- **Real-Time Prediction**: Enter any sentence and get instant word-by-word POS tags.
- **Premium Color-Coded Badges**: Predicted tags are color-coded (e.g. blue for nouns, green for verbs, purple for adjectives, gray for punctuation) to provide immediate, high-fidelity visualization.
- **Live Training Sandbox**: Train or retrain the model directly from the UI, with real-time accuracy/loss charts, progress bars, and test dataset evaluations.
- **Robust Alignment**: Tokenizer settings configured to ensure a strict 1-to-1 alignment between dataset words and POS tags.
- **Robust Inference**: Automated punctuation splitting and out-of-bounds length constraints (up to 50 words) to prevent runtime crashes.

---

## 📁 Repository Structure

```
├── app.py                  # Streamlit frontend & RNN training/inference logic
├── .gitignore              # Git ignore rules for datasets, cached files, and models
├── requirements.txt        # Python dependency packages list
└── README.md               # Project documentation
```

> [!NOTE]
> The model (`pos_rnn.keras`), tokenizers (`word_tokenizer.pkl`, `tag_tokenizer.pkl`), and datasets (`ner.csv`) are excluded from Git repository tracking to prevent large binary clutter.

---

## ⚙️ Installation and Setup

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd Many-Many-RNN
```

### 2. Install Dependencies
Make sure you have Python 3.8+ installed, then run:
```bash
pip install -r requirements.txt
```

### 3. Add the Dataset
Place the dataset `ner.csv` inside the project root directory. The dataset is expected to contain `Sentence` and `POS` columns.

---

## 🚀 Running the Tagger

Start the Streamlit application:
```bash
streamlit run app.py
```

Open the local address printed by Streamlit (usually `http://localhost:8501`) in your browser to interact with the application.
- **Inference**: Enter a sentence in the **Tag a Sentence** tab to visualize the tags.
- **Training**: Go to the **Train Model** tab, set your training hyperparameters (epochs, batch size), and click **Start Training**.
