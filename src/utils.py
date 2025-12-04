import os
from typing import List

def load_text_files_from_folder(folder_path: str) -> List[str]:
    docs = []
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        if os.path.isfile(fpath) and fname.lower().endswith((".txt", ".md")):
            with open(fpath, "r", encoding="utf-8") as f:
                docs.append(f.read())
    return docs


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """
    Simple sliding window chunking.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

