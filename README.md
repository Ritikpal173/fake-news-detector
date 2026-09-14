# 📰 Fake News Detection using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end Natural Language Processing (NLP) and Machine Learning project designed to classify news articles as **REAL** or **FAKE**. Built using TF-IDF vectorization and a Passive-Aggressive Classifier to achieve high-accuracy detection on text datasets.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)
- [Project Architecture](#-project-architecture)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Model Evaluation](#-model-evaluation)
- [Project Structure](#-project-structure)
- [Future Enhancements](#-future-enhancements)
- [License](#-license)

---

## 🔍 Overview

In an era of rapid digital information exchange, misinformation and fake news propagate swiftly. This project provides an automated pipeline that:
1. Ingests raw news article text and headlines.
2. Cleans and transforms text data into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**.
3. Classifies claims using a **Passive-Aggressive Classifier**, an online learning algorithm optimized for large-scale and high-dimensional text data.
4. Delivers an accuracy exceeding **92%** on test benchmarks.

---

## 📊 Dataset

The model utilizes the standard **Fake and Real News Dataset** (`news.csv`), containing approximately 6,335 articles labeled as either `REAL` or `FAKE`.

| Column | Description |
| :--- | :--- |
| `Unnamed: 0` | Unique article identifier |
| `title` | News article headline |
| `text` | Full body text of the article |
| `label` | Target label (`REAL` or `FAKE`) |

> **Note:** Due to size considerations, raw datasets should be downloaded and placed in the project root or configured via paths in the script.

---

## 🛠 Tech Stack

- **Language:** Python 3.8+
- **Data Manipulation:** `pandas`, `numpy`
- **Machine Learning & NLP:** `scikit-learn` (`TfidfVectorizer`, `PassiveAggressiveClassifier`, metrics)
- **Visualization:** `matplotlib`, `seaborn`

---

## ⚙️ Project Architecture

```text
Raw Text Input
      │
      ▼
Text Preprocessing & Tokenization
      │
      ▼
TF-IDF Vectorization (Stop words removal, max_df thresholding)
      │
      ▼
Passive-Aggressive Classification
      │
      ▼
Prediction: [ REAL | FAKE ]
```

### Why Passive-Aggressive Classifier?
- **Passive:** If the prediction is correct, keep the model weights unchanged.
- **Aggressive:** If the prediction is wrong, make aggressive updates to the weights to correct the mistake.
- Highly effective for high-dimensional sparse representations typical of TF-IDF matrices.

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/fake-news-detection.git
cd fake-news-detection
```

### 2. Set Up a Virtual Environment (Optional but Recommended)
```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

Run the training and evaluation script:

```bash
python fake_news_detection.py
```

### Inference Example
```python
# Sample inference snippet
new_article = ["Breaking news: Scientists discover water on new exoplanet..."]
transformed_input = tfidf_vectorizer.transform(new_article)
prediction = pac.predict(transformed_input)
print(f"Prediction: {prediction[0]}")
```

---

## 📈 Model Evaluation

On an 80/20 train-test split (1,267 test instances), the model yields:

- **Accuracy:** ~93%
- **Confusion Matrix:**

```text
               Predicted FAKE    Predicted REAL
Actual FAKE         588                49
Actual REAL          42               588
```

- **Precision & Recall:** Balanced across both `REAL` and `FAKE` classes with an F1-score of ~0.93.

---

## 📁 Project Structure

```text
fake-news-detection/
├── data/
│   └── news.csv              # Dataset (or add to .gitignore)
├── notebooks/
│   └── exploration.ipynb     # Exploratory analysis & experiments
├── fake_news_detection.py      # Main pipeline & training script
├── requirements.txt            # Project dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # Project documentation
```

---

## 🔮 Future Enhancements

- [ ] Deploy as an interactive web app using **Streamlit** or **Flask**.
- [ ] Incorporate transformer-based models (BERT / RoBERTa) for contextual embeddings.
- [ ] Build a browser extension for real-time news article verification.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
