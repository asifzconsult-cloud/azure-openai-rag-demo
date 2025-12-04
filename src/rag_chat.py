import faiss
import numpy as np
from openai import OpenAI
from rich.console import Console

import config  # local config.py

console = Console()


def get_client() -> OpenAI:
    return OpenAI(
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_API_BASE
    )


def load_vector_store():
    index = faiss.read_index(config.VECTOR_STORE_PATH)
    chunks = np.load("data/chunks.npy", allow_pickle=True)
    return index, chunks


def embed_query(client: OpenAI, query: str) -> np.ndarray:
    resp = client.embeddings.create(
        model=config.EMBEDDING_MODEL,
        input=[query]
    )
    return np.array(resp.data[0].embedding, dtype="float32").reshape(1, -1)


def retrieve_relevant_chunks(index, chunks, query_embedding, k: int = 4):
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]


def build_prompt(query: str, context_chunks) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    prompt = f"""You are an assistant helping with Azure/IT operations.

Use the context below to answer the question. If the answer is not in the context, say you don't know.

Context:
{context}

Question: {query}

Answer:"""
    return prompt


def chat():
    client = get_client()
    index, chunks = load_vector_store()

    console.print("[bold blue]RAG Chat – Azure OpenAI + FAISS Demo[/bold blue]")
    console.print("Type 'exit' to quit.\n")

    while True:
        query = console.input("[bold green]You:[/bold green] ")
        if query.lower().strip() in {"exit", "quit"}:
            break

        q_emb = embed_query(client, query)
        top_chunks = retrieve_relevant_chunks(index, chunks, q_emb, k=4)
        prompt = build_prompt(query, top_chunks)

        completion = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        answer = completion.choices[0].message.content
        console.print(f"[bold cyan]Assistant:[/bold cyan] {answer}\n")


if __name__ == "__main__":
    chat()
