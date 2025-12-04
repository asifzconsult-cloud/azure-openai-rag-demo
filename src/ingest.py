import os
import faiss
import numpy as np
from typing import List
from openai import OpenAI

from utils import load_text_files_from_folder, chunk_text
import config  # your local config.py


def get_client() -> OpenAI:
    return OpenAI(
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_API_BASE
    )


def embed_texts(client: OpenAI, texts: List[str]) -> np.ndarray:
    """
    Returns a 2D numpy array of embeddings.
    """
    response = client.embeddings.create(
        model=config.EMBEDDING_MODEL,
        input=texts
    )
    vectors = [d.embedding for d in response.data]
    return np.array(vectors).astype("float32")


def build_vector_store():
    client = get_client()
    print("[*] Loading documents...")
    docs = load_text_files_from_folder(config.DOCS_PATH)

    print(f"[*] Loaded {len(docs)} documents.")
    all_chunks = []
    chunk_sources = []

    for idx, doc in enumerate(docs):
        chunks = chunk_text(doc, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        all_chunks.extend(chunks)
        chunk_sources.extend([f"doc_{idx}"] * len(chunks))

    print(f"[*] Total chunks: {len(all_chunks)}")
    print("[*] Embedding chunks...")
    embeddings = embed_texts(client, all_chunks)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs(os.path.dirname(config.VECTOR_STORE_PATH), exist_ok=True)
    faiss.write_index(index, config.VECTOR_STORE_PATH)

    # save chunks and sources for later lookup
    np.save("data/chunks.npy", np.array(all_chunks, dtype=object))
    np.save("data/chunk_sources.npy", np.array(chunk_sources, dtype=object))

    print("[✓] Vector store built and saved.")


if __name__ == "__main__":
    build_vector_store()
