# Azure OpenAI RAG Demo

Small end-to-end RAG (Retrieval-Augmented Generation) example using:

- OpenAI / Azure OpenAI (embeddings + chat)
- Local FAISS vector store
- Simple text files as knowledge base

This repo demonstrates the core pattern I use when building internal knowledge assistants and copilots for IT / cloud operations teams.

---

## Features

- Ingests `.txt` / `.md` documents from `data/sample_docs`
- Splits them into overlapping chunks
- Generates embeddings using `text-embedding-3-small` (or Azure equivalent)
- Stores embeddings in a FAISS index
- Runs a small RAG chat loop:
  - Embed query
  - Retrieve top-k chunks
  - Build prompt with context
  - Call chat model to answer

This can be adapted to:
- Azure AI Search instead of FAISS
- Enterprise document stores
- IT SOPs, runbooks, troubleshooting guides, etc.

---

## Setup

```bash
git clone https://github.com/<your-username>/azure-openai-rag-demo.git
cd azure-openai-rag-demo

python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
