# this returns us a clean text which does not consist of bullets, unnecessary spacings and all
import re

def clean_text(text: str) -> str:
    # normalize whitespace
    text = re.sub(r"\s+", " ", text)
    # normalize bullets (•, -, *, etc.)
    text = re.sub(r"[•\-*▪◦]", "-", text)
    # normalize unicode
    text = text.encode("utf-8", "ignore").decode()
    return text.strip()