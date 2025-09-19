from typing import List
import re
import nltk
from nltk.tokenize import sent_tokenize

try:
    nltk.data.find("tokenizers/punkt")
except Exception:
    nltk.download("punkt")

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Sliding window chunking
def chunk_sliding(text: str, chunk_size:int=500, overlap: int = 50) -> List[str]:
    text = clean_text(text)
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end= min(start + chunk_size, length)
        chunks.append(text[start:end])
        if end == length:
            break
        start = max(0, end - overlap)

    return chunks

# Sentence-based chunking
def chunk_sentences(text: str, max_chunk_size: int = 500) -> List[str]:
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)
        if current_len + sentence_len > max_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_len = sentence_len
        else:
            current_chunk.append(sentence)
            current_len += sentence_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))

    return [clean_text(chunk) for chunk in chunks]