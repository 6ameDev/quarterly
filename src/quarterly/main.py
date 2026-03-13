import sys
import traceback
from contextlib import asynccontextmanager

import chromadb
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from quarterly import config, ollama
from quarterly.analyst import Analyst
from quarterly.ingestor import Ingestor
from quarterly.schemas import IngestRequest, QuestionRequest


class AppState:
    def __init__(self):
        self.ingestor: Ingestor | None = None
        self.analyst: Analyst | None = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing centralized resources...")

    if not await ollama.is_healthy(config.BASE_URL):
        print(f"CRITICAL ERROR: Ollama is not reachable at {config.BASE_URL}")
        sys.exit(1)

    try:
        Settings.embed_model = OllamaEmbedding(
            model_name=config.EMBED_MODEL_NAME, base_url=config.BASE_URL
        )
        Settings.llm = Ollama(
            model=config.LLM_MODEL_NAME,
            system_prompt=config.SYSTEM_PROMPT,
            request_timeout=120.0,
            temperature=0.1,
        )

        db = chromadb.PersistentClient(path=config.PERSIST_DIR)
        chroma_collection = db.get_or_create_collection(config.COLLECTION_NAME)

        state.ingestor = Ingestor(collection=chroma_collection)
        state.analyst = Analyst(collection=chroma_collection)

        print("All resources initialized successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR during initialization: {e}")
        traceback.print_exc()
        sys.exit(1)

    yield
    print("Shutting down...")


app = FastAPI(title="Quarterly API", lifespan=lifespan)


@app.post("/ingest")
async def ingest_document(request: IngestRequest):
    if not state.ingestor:
        raise HTTPException(status_code=503, detail="Ingestor not initialized")

    try:
        state.ingestor.ingest_text(request.text, metadata=request.metadata)
        filename = request.metadata.get("filename", "unnamed") if request.metadata else "unnamed"
        return {
            "status": "success",
            "message": f"Document ingested: {filename}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    if not state.analyst:
        raise HTTPException(status_code=503, detail="Analyst not initialized")

    try:
        response = state.analyst.ask(request.question, streaming=True)

        def response_generator():
            yield from response.response_gen

        return StreamingResponse(response_generator(), media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/health")
async def health_check():
    return {"status": "ok"}


def main():
    uvicorn.run("quarterly.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
