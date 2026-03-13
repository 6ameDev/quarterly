import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.vector_stores.chroma import ChromaVectorStore


class Ingestor:
    def __init__(self, collection: chromadb.api.models.Collection.Collection):
        self.vector_store = ChromaVectorStore(chroma_collection=collection)
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
