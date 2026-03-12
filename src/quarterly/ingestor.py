import chromadb
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore


class Ingestor:
    def __init__(
        self,
        collection_name: str = "quarterly_docs_nomic",
        persist_dir: str = "./chroma_db",
        embed_model_name: str = "nomic-embed-text:latest",
        base_url: str = "http://localhost:11434",
    ):
        # Configure the embedding model
        self.embed_model = OllamaEmbedding(model_name=embed_model_name, base_url=base_url)
        Settings.embed_model = self.embed_model

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


if __name__ == "__main__":
    ingestor = Ingestor()

    ingestor.ingest_text(
        "LlamaIndex is a data framework for LLM applications to index, retrieve, and query data.",
        metadata={"filename": "test.txt"},
    )
