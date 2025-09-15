from typing import Tuple
from io import BytesIO
from PyPDF2 import PdfReader

def extract_text_from_pdf(data: bytes) -> str:
    reader = PdfReader(BytesIO(data))
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            continue
    
    return "\n\n".join(texts)

def extract_text_from_txt(data: bytes, encoding: str = "utf-8") -> str:
    return data.decode(encoding, errors="ignore")