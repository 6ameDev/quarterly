import chromadb
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

# llms: llama3.2:1b, qwen2.5:1.5b
# embed models: nomic-embed-text:latest

SYSTEM_PROMPT = """You are a technical data extractor.
Output ONLY the requested information.
No conversational filler.
No "Sure, here is the summary".
If the answer is not in the context, say 'N/A'."""


class Ingestor:
    def __init__(
        self,
        collection_name: str = "quarterly_docs_nomic",
        persist_dir: str = "./chroma_db",
        embed_model_name: str = "nomic-embed-text:latest",
        llm_model_name: str = "qwen2.5:1.5b",
        base_url: str = "http://localhost:11434",
    ):
        # Configure the embedding model
        self.embed_model = OllamaEmbedding(model_name=embed_model_name, base_url=base_url)
        Settings.embed_model = self.embed_model

        # Configure the LLM
        self.llm = Ollama(
            model=llm_model_name,
            system_prompt=SYSTEM_PROMPT,
            request_timeout=120.0,
            temperature=0.1,
        )
        Settings.llm = self.llm

        # Initialize ChromaDB client
        self.db = chromadb.PersistentClient(path=persist_dir)
        self.chroma_collection = self.db.get_or_create_collection(collection_name)

        # Initialize the vector store
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

    def ingest_text(self, text: str, metadata: dict = None) -> None:
        """
        Ingest a single piece of text into the vector store.
        """
        doc = Document(text=text, metadata=metadata or {})
        index = VectorStoreIndex.from_documents([doc], storage_context=self.storage_context)

        filename = metadata.get("filename", "unnamed") if metadata else "unnamed"
        print(f"Successfully ingested document: {filename}, Index: {index}")

    def ingest_documents(self, documents: list[Document]) -> None:
        """
        Ingest multiple LlamaIndex Document objects into the vector store.
        """
        index = VectorStoreIndex.from_documents(documents, storage_context=self.storage_context)
        print(f"Successfully ingested {len(documents)} documents. Index: {index}")

    def read_documents(self, query: str) -> list[Document]:
        """
        Read documents from the vector store based on a query.
        """
        index = VectorStoreIndex.from_vector_store(
            self.vector_store, storage_context=self.storage_context
        )

        query_engine = index.as_query_engine(streaming=True)
        return query_engine.query(query)


if __name__ == "__main__":
    # Quick test ingestion
    ingestor = Ingestor()

    # ingestor.ingest_text(
    #     "LlamaIndex is a data framework for LLM applications to index, retrieve, and query data.",
    #     metadata={"filename": "test.txt"},
    # )

    response = ingestor.read_documents("Summarize the main points of these documents.")
    response.print_response_stream()
