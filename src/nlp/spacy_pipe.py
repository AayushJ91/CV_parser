# src/nlp/spacy_pipe.py
from functools import lru_cache
from typing import Iterable, List, Tuple, Dict
import spacy

# used NLP (Natural Language Processing)
# spaCy is an NLP library that lets us process and analyze natural language text. 
# Named Entity Recoginition is just a part of spaCy that deals with entitites
# spaCy have more than just entity - tokens, Dependencies(related to grammar), POS(part of speech)
# Specifically, spaCy takes raw text (string) and turns it into a Doc object, which contains tokens, sentences, and linguistic annotations (POS tags, dependencies, entities, etc.).
# breaks it into different entities by itself
# gives different entity lable like ['NAME', 'LOC','ORG']


# Load once, reuse everywhere (faster)
@lru_cache(maxsize=1)
def get_nlp(disable: Iterable[str] = ()):
    """
    Load and cache a spaCy pipeline. 
    Tip: disable components you don't need to speed up, e.g. disable=("tagger","parser","lemmatizer")
    """
    return spacy.load('en_core_web_trf', disable=list(disable))

def extract_entities(text: str, labels: Iterable[str] = None) -> List[Tuple[str, str]]:
    """
    Return [(entity_text, label), ...]. If labels is given, only keep those labels.
    """
    if not text:
        return []
    nlp = get_nlp()
    doc = nlp(text)
    ents = [(ent.text.strip(), ent.label_) for ent in doc.ents]
    if labels:
        labels = set(labels)
        ents = [e for e in ents if e[1] in labels]
    return ents

def extract_person_names(text: str) -> List[str]:
    """
    Return unique PERSON entities (likely candidate names).
    """
    if not text:
        return []
    nlp = get_nlp()
    doc = nlp(text)
    return sorted({ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON"})

def quick_ner_summary(text: str) -> Dict[str, int]:
    """
    Count entities by label for a quick glance (e.g., how many PERSON/ORG/DATE were found).
    """
    if not text:
        return {}
    from collections import Counter
    nlp = get_nlp()
    doc = nlp(text)
    return dict(Counter(ent.label_ for ent in doc.ents))

# ---- Optional CLI so you can run this file directly ----
if __name__ == "__main__":
    import argparse, json, pathlib
    from src.parsing.resume_parser import parse_resume       # uses your existing orchestrator
    from src.parsing.utils import clean_text                 # your Day 7 cleaner (optional but recommended)

    parser = argparse.ArgumentParser(description="Run spaCy NER on a resume (any format).")
    parser.add_argument("path", help="Path to a resume file (.pdf, .docx, .png, .jpg, or .txt)")
    parser.add_argument("--model", default="en_core_web_sm", help="spaCy model name")
    parser.add_argument("--limit", type=int, default=30, help="How many entities to print")
    args = parser.parse_args()

    p = pathlib.Path(args.path)
    text = ""

    if p.suffix.lower() == ".txt":
        text = p.read_text(encoding="utf-8", errors="ignore")
    else:
        # Reuse your resume parser to get text from PDF/DOCX/Images (with OCR)
        text = parse_resume(str(p))

    # optional cleaning
    try:
        text = clean_text(text)
    except Exception:
        pass

    # load the requested model (cached)
    get_nlp.cache_clear()
    get_nlp(args.model)

    ents = extract_entities(text)
    out = {
        "file": str(p),
        "sample_text": text[:300],
        "entity_counts": quick_ner_summary(text),
        "entities_preview": ents[: args.limit]
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
