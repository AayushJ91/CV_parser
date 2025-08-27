# Resume Parsing Pipeline

This project extracts, cleans, and saves text from resumes in multiple formats 
(`.pdf`, `.docx`, `.png`, `.jpg`).

## 📂 Project Structure
data/
    resumes_raw/ # Input resumes (PDF, DOCX, Images)
    resumes_text/ # Output cleaned text files
src/
    resume_parser.py # Parser script
    files
    for
    extracting
    text
    sanity_check.py # Checks whether the resumes in resumes_raw is missing something
    utils.py # utiliy fuction
main.ipynb # Notebook for experiments