from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import json

def extract_text_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    text_parts = []
    for key, value in data.items():
        if isinstance(value, str):
            text_parts.append(value)
        elif isinstance(value, list):
            text_parts.extend([str(v) for v in value])
        elif isinstance(value, dict):
            text_parts.extend([str(v) for v in value.values()])
    return " ".join(text_parts)

def rank_cvs_in_sector(sector_dir):
    json_files = [f for f in os.listdir(sector_dir) if f.endswith(".json")]
    if not json_files:
        return {}

    # Extract CV text
    texts = [extract_text_from_json(os.path.join(sector_dir, f)) for f in json_files]

    # TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Similarity
    sim_matrix = cosine_similarity(tfidf_matrix)

    # Average similarity
    avg_scores = sim_matrix.mean(axis=1)

    # Normalize to 0–100
    scaler = MinMaxScaler((0, 100))
    normalized_scores = scaler.fit_transform(avg_scores.reshape(-1, 1)).flatten()

    # Save scores into JSON files
    for i, filename in enumerate(json_files):
        score = round(float(normalized_scores[i]), 2)
        json_path = os.path.join(sector_dir, filename)

        # Load JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Add score
        data["Score"] = score

        # Overwrite JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    # Also return a dictionary of scores
    return {
        json_files[i]: round(float(normalized_scores[i]), 2)
        for i in range(len(json_files))
    }