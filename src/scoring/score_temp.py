import os
import json
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def compute_raw_score(data):
    score = 0

    # Education
    score += min(len(data.get("Education", [])) * 5, 15)

    # Experience
    score += min(len(data.get("Experience", [])) * 8, 25)

    # Skills
    score += min(len(data.get("Skills", [])) * 4, 25)

    # Projects
    score += min(len(data.get("Projects", [])) * 5, 20)

    # Completeness (bonus for having multiple sections)
    sections_present = sum(1 for key in ["Education", "Experience", "Skills", "Projects"] if data.get(key))
    score += (sections_present / 4) * 10

    # Extras (Certifications, Achievements)
    extras = data.get("Certifications", []) or data.get("Achievements", [])
    if extras:
        score += 5

    return min(score, 100)


def rank_cvs_in_sector(sector_dir):
    print(sector_dir)
    files = [f for f in os.listdir(sector_dir) if f.endswith(".json")]
    print(files)
    scores = []
    data_map = {}

    for f in files:
        path = os.path.join(sector_dir, f)
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        raw_score = compute_raw_score(data)
        scores.append(raw_score)
        data_map[f] = raw_score

    # Normalize scores between 0–100
    print(scores)
    scaler = MinMaxScaler((0, 100))
    normalized = scaler.fit_transform(np.array(scores).reshape(-1, 1)).flatten()
    print(normalized)

    # Map back to file names
    ranked = {files[i]: round(normalized[i], 2) for i in range(len(files))}
    return ranked
