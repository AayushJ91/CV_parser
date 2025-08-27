# for pdf files
import pdfplumber, pathlib

def extract_text_from_pdf(pdf_path, output_path=None):
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    if output_path != None:  # only write if provided
        pathlib.Path(output_path).write_text(text, encoding="utf-8")
    return text