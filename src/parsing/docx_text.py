import docx2txt
from pathlib import Path

# extracting text from a docx file
def extract_text_from_docx(docx_path: str, output_path: str = None) -> str:
    """
    Extract text from a .docx file and optionally save to a .txt file.
    """
    text = docx2txt.process(docx_path)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
    
    return text