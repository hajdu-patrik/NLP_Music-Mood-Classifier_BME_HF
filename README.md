# Music Mood Recommender

This project is a Python-based NLP system that analyzes song lyrics to determine their emotional profile and recommends similar songs. It was built for the "Natural Language and Semantic Technologies" course at BME-VIK.

The system first trains a machine learning model on a large dataset of emotionally-labeled texts. It then uses this model to classify songs from the Spotify Million Song Dataset into six emotional categories (sadness, joy, love, anger, fear, surprise). Finally, it provides recommendations based on both matching emotions and textual similarity (TF-IDF & Cosine Similarity).

## Features

* **Emotion Classification:** Classifies song lyrics into one of six core emotions.
* **Similarity-Based Recommendation:** Recommends songs that share the same emotional profile and have similar lyrical content.
* **Interactive CLI:** A simple command-line interface to input an artist and song title.
* **Robust Input Handling:** Cleans and normalizes user input (handles whitespace and case-insensitivity).

## Tech Stack

* **Python 3.10+**
* **Scikit-learn:** For machine learning (Pipeline, TF-IDF, Logistic Regression, Cosine Similarity).
* **Pandas:** For data loading and manipulation.
* **NLTK:** For text preprocessing (tokenization, stopwords).

## Datasets Used

1.  **Emotion Classification Model:** [Emotions in Text Dataset](https://www.kaggle.com/datasets/nelgiriyewithana/emotions/data) (approx. 417k labeled texts).
2.  **Song Database:** [Spotify Million Song Dataset](https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset) (approx. 57k songs with lyrics).

---

## Setup & Usage

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
```

### 2. Create and Activate Virtual Environment

**Windows (Git Bash):**
```bash
python -m venv .venv
source .venv/Scripts/activate
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Datasets

1.  Download the [Spotify Dataset](https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset) and place the `spotify_millsongdata.csv` in the root folder.
2.  Download the [Emotions Dataset](https://www.kaggle.com/datasets/nelgiriyewithana/emotions/data) and place the `emotions.csv` (you may need to rename it from `text.csv`) in the root folder.

### 5. Run the Application

Once the environment is active and the CSVs are in place, run the script:

```bash
python main.py
```

The script will first train the emotion model (which may take a few minutes) and then launch the interactive prompt.

## Example Interaction

```bash
--- Music Mood Recommender System Started ---
To exit, type: 'exit'

Enter artist name: abba
Enter song title: cassandra

--- Analysis: abba - cassandra ---
Determined emotion: sadness

Recommendations (based on similar emotion and text):
  1. Conway Twitty - Don't Tell Me You're Sorry (Similarity: 0.28)
  2. The Temptations - Sorry Is A Sorry Word (Similarity: 0.26)
  3. Gordon Lightfoot - Remember Me (Similarity: 0.18)
  4. Hanson - Being Me (Similarity: 0.17)
  5. Religious Music - Angels Among Us (Similarity: 0.15)

Enter artist name: exit
```

## Model Performance

The emotion classification model (Logistic Regression on TF-IDF features) achieved a **90% weighted average F1-score** on the validation set (83,362 samples).
