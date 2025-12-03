import os
import json
import streamlit as st
import fitz  # PyMuPDF
import spacy
import re
from docx import Document
from parsing.pdf_text import extract_text_from_pdf
from parsing.docx_text import extract_text_from_docx
from parsing.resume_parser import extract_name, extract_email, extract_sections, identify_sector
from pathlib import Path
from transformers import pipeline
from scoring.score import rank_cvs_in_sector

# -----------------------------
# 📁 PATH SETUP
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(BASE_DIR, "data", "resumes_raw")
TEXT_DIR = os.path.join(BASE_DIR, "data", "resumes_text")
JSON_DIR = os.path.join(BASE_DIR, "data", "cv_jsons")

for d in [RAW_DIR, TEXT_DIR, JSON_DIR]:
    os.makedirs(d, exist_ok=True)

# -----------------------------
# 🧠 LOAD NLP MODEL
# -----------------------------
nlp = spacy.load("en_core_web_sm")

# -----------------------------
# ⚙️ EXTRACTION FUNCTIONS
# -----------------------------

@st.cache_resource
def load_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

classifier = load_classifier()
def identify_sector_ai(cv_text: str) -> str:
    """
    Identify the job sector of a CV using a zero-shot classification model.
    """

    # Define your sector labels
    candidate_labels = [
        "Software Developer",
        "Data Scientist",
        "Product Manager",
        "UI/UX Designer",
        "DevOps Engineer",
        "Business Analyst",
        "Marketing Specialist",
        "Prompt Engineer",
        "Cybersecurity Engineer",
        "Human Resources Manager",
        "Finance Analyst"
    ]

    # Run the AI model
    result = classifier(
        sequences=cv_text,
        candidate_labels=candidate_labels,
        multi_label=False  # choose the single best label
    )

    # Get the top label
    best_sector = result["labels"][0]
    confidence = result["scores"][0]

    print(f"Predicted Sector: {best_sector} (Confidence: {confidence:.2f})")
    return best_sector


def extract_all_info(text):
    sections = extract_sections(text)
    final_data = {
        "Name": extract_name(text),
        "Email": extract_email(text),
        "Sector": identify_sector_ai(text),
        **sections
    }
    return final_data

# -----------------------------
# 🖥️ STREAMLIT UI
# -----------------------------
st.title("📄 CV Analyzer – Upload & Extract")

uploaded_file = st.file_uploader("Upload your CV (PDF or DOCX)", type=["pdf", "docx"])

if uploaded_file:
    # Save uploaded file
    file_path = os.path.join(RAW_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"✅ File saved to {RAW_DIR}")

    # Extract text
    if uploaded_file.name.lower().endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
        text_file = os.path.join(TEXT_DIR, uploaded_file.name.replace(".pdf", ".txt"))
    elif uploaded_file.name.lower().endswith(".docx"):
        text = extract_text_from_docx(file_path)
        text_file = os.path.join(TEXT_DIR, uploaded_file.name.replace(".docx", ".txt"))
    else:
        st.error("❌ Unsupported file type. Please upload a PDF or DOCX file.")
        st.stop()

    # Save extracted text
    with open(text_file, "w", encoding="utf-8") as f:
        f.write(text)
    st.info(f"📝 Text extracted and saved at {TEXT_DIR}")

    # 🧠 Identify Sector (AI)
    st.info("🤖 Identifying sector using AI model...")
    sector = identify_sector_ai(text)
    main_sector = sector.split("(")[0].strip()
    st.success(f"🏷️ Detected Sector: {sector}")

    # Create sector folder
    sector_dir = os.path.join(JSON_DIR, main_sector.replace(" ", "_"))
    os.makedirs(sector_dir, exist_ok=True)

    # Extract structured info
    data = extract_all_info(text)
    data["Sector"] = main_sector

    # Save JSON inside the sector folder
    json_file = os.path.join(sector_dir, uploaded_file.name.rsplit(".", 1)[0] + ".json")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    st.success(f"🎯 Data saved as JSON: {json_file}")

    # Display output
    st.subheader("🔍 Extracted Information:")
    st.json(data)


    # sector_dir = os.path.join(JSON_DIR, main_sector)

# Save user's JSON first
    # Get the base filename (without extension)
    base_name = os.path.splitext(uploaded_file.name)[0]

# Create JSON path dynamically (works for both PDF and DOCX)
    user_json_path = os.path.join(sector_dir, base_name + ".json")

# Now score all CVs in the same sector
    sector_scores = rank_cvs_in_sector(sector_dir)

    user_score = sector_scores.get(os.path.basename(user_json_path), "N/A")

    st.metric("📊 Your CV Score", f"{user_score}/100")
