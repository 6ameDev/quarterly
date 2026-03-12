import chromadb
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

# llms: qwen2.5:1.5b, llama3.2:1b
# embed models: nomic-embed-text:latest

SYSTEM_PROMPT = """You are an expert financial and technical analyst.
Use the provided context to accurately answer the user's question.
If the context does not contain the answer, say 'I cannot answer this based on the provided context.'
Always be precise and objective."""

class Analyst:
    def __init__(
        self,
        collection_name: str = "quarterly_docs_nomic",
        persist_dir: str = "./chroma_db",
        embed_model_name: str = "nomic-embed-text:latest",
        llm_model_name: str = "qwen2.5:1.5b",
        base_url: str = "http://localhost:11434",
    ):
        self.embed_model = OllamaEmbedding(model_name=embed_model_name, base_url=base_url)
        Settings.embed_model = self.embed_model

        self.llm = Ollama(
            model=llm_model_name,
            system_prompt=SYSTEM_PROMPT,
            request_timeout=120.0,
            temperature=0.1,
        )
        Settings.llm = self.llm

        self.db = chromadb.PersistentClient(path=persist_dir)
        self.chroma_collection = self.db.get_or_create_collection(collection_name)

        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
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

if __name__ == "__main__":
    analyst = Analyst()
    question = "What is LlamaIndex and for what real world use-case can I use it?"

    print(f"Asking: \n{question}\n")
    response = analyst.ask(question, streaming=True)
    response.print_response_stream()
