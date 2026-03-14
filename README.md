# Quarterly

Quarterly is a local-first, interactive tool designed for intelligent document analysis and research. It leverages **LlamaIndex**, **FastAPI**, and **ChromaDB** to provide a fast, privacy-focused RAG (Retrieval Augmented Generation) experience using locally running LLMs via **Ollama**.

## Features

- **Local RAG**: Process and query your documents without sending data to external APIs.
- **FastAPI Backend**: A robust server that manages resource initialization, health checks, and streaming responses.
- **Interactive REPL**: A polished CLI for seamless document ingestion and querying.
- **Streaming Responses**: Get real-time answers from your LLM token by token.
- **Support for Directories**: Ingest entire folders of documents with a single command.

## Prerequisites

- **Python**: 3.13+ recommended.
- **Ollama**: Must be running locally.
  - Recommended LLM: `qwen2.5:1.5b`
  - Recommended Embedding: `nomic-embed-text:latest`
- **uv**: Python package and project manager.

## Setup & Installation

1. **Install uv**: Follow the [uv installation guide](https://github.com/astral-sh/uv).
2. **Install Dependencies**:
   ```bash
   make install
   ```
   *This will sync all project dependencies and set up the virtual environment.*

## Running the Project

Quarterly is split into two main components: the **Server** (API) and the **REPL** (CLI).

### 1. Start the Server
In your first terminal, start the FastAPI backend:
```bash
make server
```
The server will initialize your vector store and verify connectivity with Ollama. It runs by default on `http://localhost:8000`.

### 2. Start the REPL
In a second terminal, start the interactive interface:
```bash
make repl
```

## CLI Commands

Once inside the REPL, you can use the following commands:

- `/ask <your question>`: Query your ingested documents.
- `/ingest <path>`: Ingest a specific file or an entire directory recursively.
- `/model`: Interactively select and set the active LLM model.
- `/host <url>`: View or update the target server URL (persisted in `~/.quarterly.json`).
- `/help`: Show available commands and usage hints.
- `/exit` or `/quit`: Close the REPL.

## Development

- **Linting & Formatting**:
  ```bash
  make lint
  make format
  ```
- **Testing**:
  ```bash
  make test
  ```
