# Shared configurations for the quarterly project

COLLECTION_NAME = "quarterly_docs_nomic"
PERSIST_DIR = "./chroma_db"
EMBED_MODEL_NAME = "nomic-embed-text:latest"
LLM_MODEL_NAME = "qwen2.5:1.5b"
BASE_URL = "http://localhost:11434"

SYSTEM_PROMPT = """You are an expert financial and technical analyst.
Use the provided context to accurately answer the user's question.
If the context does not contain the answer, say 
'I cannot answer this based on the provided context.'
Always be precise and objective."""
