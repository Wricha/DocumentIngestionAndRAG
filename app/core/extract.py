from typing import Tuple
from io import BytesIO
from PyPDF2 import PdfReader

def extract_text_from_pdf(upload_file: BytesIO) -> str:
    upload_file.seek(0)  # make sure we're at the start
    pdf = PdfReader(upload_file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

def extract_text_from_txt(data: bytes, encoding: str = "utf-8") -> str:
    return data.decode(encoding, errors="ignore")