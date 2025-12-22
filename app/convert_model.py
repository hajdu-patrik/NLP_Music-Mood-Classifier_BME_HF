import pandas as pd
import json
import gzip
import os

# Paths
input_path = "model/analyzed_lyrics.pkl.gz"
output_path = "model/lyrics_metadata.json.gz"

print(f"Loading: {input_path}...")
try:
    df = pd.read_pickle(input_path)
except:
    df = pd.read_pickle("model/analyzed_lyrics.pkl")

print(f"Original size (rows): {len(df)}")

# Convert to list, and DROP the 'text' column to save space.
# The order (index) must be preserved to match the TF-IDF matrix!
data_list = []
for index, row in df.iterrows():
    data_list.append({
        "artist": row['artist'],
        "song": row['song'],
        "emotion_label": int(row['emotion_label']),
        "emotion_name": row['emotion_name']
    })

print(f"Saving to: {output_path}...")
with gzip.open(output_path, "wt", encoding="utf-8") as f:
    json.dump(data_list, f)

print("Done! You can now delete the old .pkl and .pkl.gz files from the model folder.")