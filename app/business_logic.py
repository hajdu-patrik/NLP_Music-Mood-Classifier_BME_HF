import numpy as np
import time
import re
import os
import joblib
import json
import gzip
import argparse
from scipy.sparse import save_npz, load_npz
from sklearn.metrics.pairwise import cosine_similarity

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model')

# New data paths
LYRICS_METADATA_PATH = os.path.join(MODEL_DIR, 'lyrics_metadata.json.gz')
LYRICS_MATRIX_PATH = os.path.join(MODEL_DIR, 'lyrics_tfidf_matrix.npz')
EMOTION_MODEL_PATH = os.path.join(MODEL_DIR, 'emotion_pipeline.joblib')
LYRICS_VECTORIZER_PATH = os.path.join(MODEL_DIR, 'lyrics_vectorizer.joblib')

def preprocess_text(text):
    # Simple regex-based tokenization without NLTK
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Only letters
    tokens = re.findall(r'\b[a-z]+\b', text)
    return " ".join(tokens)

def get_recommendations(artist, song, lyrics_data, tfidf_matrix, top_n=5):
    try:
        # 1. Search in the list (search without Pandas)
        target_index = -1
        song_entry = None
        
        artist_lower = artist.lower().strip()
        song_lower = song.lower().strip()

        for idx, item in enumerate(lyrics_data):
            if item['artist'].lower() == artist_lower and item['song'].lower() == song_lower:
                target_index = idx
                song_entry = item
                break
        
        if target_index == -1:
            return {"error": f"Sorry, the song '{artist} - {song}' was not found in the database."}

        # 2. Extract data
        target_emotion_label = song_entry['emotion_label']
        target_emotion_name = song_entry['emotion_name']
        
        # 3. Calculate similarity
        # Extract the corresponding row from the matrix
        song_vector = tfidf_matrix[target_index]
        # Calculate similarity with ALL songs
        sim_scores = cosine_similarity(song_vector, tfidf_matrix)[0]
        
        # 4. Manual filtering and sorting
        # Build a list: (index, score), but only for those with matching emotions
        candidates = []
        for i, score in enumerate(sim_scores):
            if i == target_index: continue # We skip ourselves
            
            # We only add it if the emotion matches
            if lyrics_data[i]['emotion_label'] == target_emotion_label:
                candidates.append((i, score))
        
        # Sort by score (descending)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Select Top N
        top_candidates = candidates[:top_n]
        
        # Format results
        recommendations = []
        for idx, score in top_candidates:
            rec_item = lyrics_data[idx].copy()
            rec_item['similarity'] = score
            recommendations.append (rec_item)

        return {
            "error": None,
            "input_song": {
                "artist": song_entry['artist'],
                "song": song_entry['song'],
                "emotion": target_emotion_name
            },
            "recommendations": recommendations
        }
                
    except Exception as e:
        print(f"An error occurred during recommendation: {e}")
        return {"error": f"An internal error occurred: {e}"}


def load_artifacts():
    print("\nLoading pre-trained models and processed data...")
    start_time = time.time()
    
    try:
        # 1. Load metadata from JSON
        with gzip.open(LYRICS_METADATA_PATH, "rt", encoding="utf-8") as f:
            lyrics_data = json.load(f)

        # 2. Load matrix
        tfidf_matrix = load_npz(LYRICS_MATRIX_PATH)
        try:
            emotion_model = joblib.load(EMOTION_MODEL_PATH)
            tfidf_vectorizer = joblib.load(LYRICS_VECTORIZER_PATH)
        except:
            print("Warning: Could not load emotion models. Search will work, but new text classification won't.")
            emotion_model = None
            tfidf_vectorizer = None

        print(f"Artifacts loaded successfully. Duration: {time.time() - start_time:.2f} sec.")
        
        return {
            "emotion_model": emotion_model,
            "lyrics_df": lyrics_data,
            "tfidf_matrix": tfidf_matrix,
            "tfidf_vectorizer": tfidf_vectorizer
        }
    except Exception as e:
        print(f"Error loading files: {e}.")
        return None

def initialize_app(force_regenerate=False):
    if force_regenerate:
        print("Warning: Regeneration is not supported in this optimization mode.")
    
    return load_artifacts()

# The CLI part can remain for testing
def main_cli():
    artifacts = load_artifacts()
    if artifacts:
        print("System loaded. Ready for queries.")
    else:
        print("Failed to load system.")

if __name__ == "__main__":
    main_cli()