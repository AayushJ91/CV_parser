# src/nlp/skills.py
from pathlib import Path
import csv
import re
from typing import Dict, List, Set

# This file extracts the skills of a person from a paragraph
# suppose a CV has a paragraph about its skills
# then this file will parse that para and give us the skills
# that are listed in skills.csv

def load_skills(csv_path: str = "data/skills.csv") -> Dict[str, List[str]]:
    """
    Returns: {canonical: [synonym1, synonym2, ...]}
    Synonyms are expected to be | separated in CSV and will be normalized to lowercase.
    """
    skills = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            canon = row["canonical"].strip()
            synonyms = [s.strip().lower() for s in row["synonyms"].split("|") if s.strip()]
            # Always include the canonical itself as a synonym match
            synonyms = list(sorted(set(synonyms + [canon.lower()])))
            skills[canon] = synonyms
    return skills

def compile_patterns(skills_dict: Dict[str, List[str]]) -> Dict[str, List[re.Pattern]]:
    """
    Compile word-boundary regex for each synonym to avoid partial hits (e.g., 'c' in 'action').
    Handles spaces and dots in tech names.
    """
    compiled = {}
    for canon, syns in skills_dict.items():
        pats = []
        for s in syns:
            # escape special chars, allow optional dots/spaces in things like "node.js"
            escaped = re.escape(s).replace(r"\ ", r"\s+").replace(r"\.", r"\.?" )
            pat = re.compile(rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])", re.IGNORECASE)
            pats.append(pat)
        compiled[canon] = pats
    return compiled

def detect_skills(text: str, patterns: Dict[str, List[re.Pattern]]) -> Set[str]:
    """
    Returns a set of canonical skills present in the text.
    """
    if not text:
        return set()
    text_lc = text.lower()
    found = set()
    for canon, pats in patterns.items():
        if any(p.search(text_lc) for p in pats):
            found.add(canon)
    return found
