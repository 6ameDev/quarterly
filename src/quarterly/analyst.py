import chromadb
import chromadb.api
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore


class Analyst:
    def __init__(self, collection: chromadb.api.models.Collection.Collection):
        self.vector_store = ChromaVectorStore(chroma_collection=collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index = VectorStoreIndex.from_vector_store(
            self.vector_store, storage_context=self.storage_context
        )

    def ask(self, question: str, streaming: bool = True):
        query_engine = self.index.as_query_engine(
            streaming=streaming,
            similarity_top_k=5,
            response_mode="compact",
        )
        response = query_engine.query(question)
        return response
