# Rename this file to config.py and fill in your real values.
# NOTE: Do NOT commit config.py to GitHub.

OPENAI_API_KEY = "your-openai-api-key"
OPENAI_API_BASE = "https://api.openai.com/v1"  # or Azure OpenAI endpoint
OPENAI_MODEL = "gpt-4.1-mini"  # or your Azure deployment name

EMBEDDING_MODEL = "text-embedding-3-small"  # or ada equivalent
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
VECTOR_STORE_PATH = "data/vector_store.index"
DOCS_PATH = "data/sample_docs"
