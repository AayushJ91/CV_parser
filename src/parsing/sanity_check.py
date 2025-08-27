import re
import spacy
import pdfplumber
from pathlib import Path
import docx2txt   # for DOCX
# this Checks whether the CVs in resumes_raw is missing something important or anything else

nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(path):
    with pdfplumber.open(path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

def extract_text_from_docx(path):
    return docx2txt.process(path) or ""

def extract_text(path):
    """Auto-detect file type and extract text accordingly."""
    if path.suffix.lower() == ".pdf":
        return extract_text_from_pdf(path)
    elif path.suffix.lower() == ".docx":
        return extract_text_from_docx(path)
    else:
        return ""

def has_email(text):
    return bool(re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-z]{2,}", text))

def has_name(text):
    doc = nlp(text)
    return any(ent.label_ == "PERSON" for ent in doc.ents)

def is_long_enough(text, min_chars=800):
    return len(text) > min_chars

if __name__ == "__main__":
    folder = Path("data/resumes_raw")
    for file in folder.glob("*"):
        text = extract_text(file)
        print(file.name, {
            "Email": has_email(text),
            "Name": has_name(text),
            "Long Enough": is_long_enough(text)
        })
