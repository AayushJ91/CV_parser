from PIL import Image
import pytesseract # for OCR (optical character recoginition) to read and extract text from images
import pathlib

# extracting text from a single image
def extract_text_from_image(image_path, output_path=None, lang="eng"):
    """
    Extract text from an image using OCR.
    image_path: str or Path to the image file (.png, .jpg, etc.)
    output_path: optional path to save extracted text
    lang: language for OCR (default English)
    """
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image, lang=lang)

    if output_path:
        pathlib.Path(output_path).write_text(text, encoding="utf-8")

    return text

# extracting text from a scanned pdf (a pdf that has been scanned and converted to an image)
from pdf2image import convert_from_path

def extract_text_from_scanned_pdf(pdf_path, output_folder="data/resumes_text", lang="eng"):
    """
    Convert each page of a scanned PDF into an image, then OCR it.
    """
    pages = convert_from_path(pdf_path)  # converts all pages into images
    pdf_path = pathlib.Path(pdf_path)
    output_folder = pathlib.Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    all_text = []
    for i, page in enumerate(pages, start=1):
        text = pytesseract.image_to_string(page, lang=lang)
        all_text.append(text)

        # save each page separately
        output_file = output_folder / f"{pdf_path.stem}_page{i}.txt"
        output_file.write_text(text, encoding="utf-8")

    return "\n".join(all_text)