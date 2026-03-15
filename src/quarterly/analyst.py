import re
import json
import chromadb
import chromadb.api
from llama_index.core import Settings
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter


class Analyst:
    def __init__(self, collection: chromadb.api.models.Collection.Collection):
        self.vector_store = ChromaVectorStore(chroma_collection=collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index = VectorStoreIndex.from_vector_store(
            self.vector_store, storage_context=self.storage_context
        )

    async def ask(self, question: str, streaming: bool = True):
        intent = await self._extract_query_intent(question)

        filters_list = []
        if intent.get("company"):
            filters_list.append(ExactMatchFilter(key="company", value=intent["company"]))
        if intent.get("fiscal_period"):
            filters_list.append(ExactMatchFilter(key="fiscal_period", value=intent["fiscal_period"]))
        if intent.get("year"):
            filters_list.append(ExactMatchFilter(key="year", value=str(intent["year"])))

        print(f"DEBUG: Filters: {filters_list}")

        filters = MetadataFilters(filters=filters_list) if filters_list else None

        query_engine = self.index.as_query_engine(
            streaming=streaming,
            similarity_top_k=5,
            filters=filters,
            response_mode="compact"
        )

        response = await query_engine.aquery(question)

        if filters and (not response.source_nodes or len(response.source_nodes) == 0):
            print(f"DEBUG: No source nodes found with filters. Retrying with global search...")
            fallback_engine = self.index.as_query_engine(
                filters=None,
                similarity_top_k=5,
                streaming=streaming
            )
            return await fallback_engine.aquery(question)

        return response

    async def _extract_query_intent(self, question: str) -> dict:
        prompt = (
            f"Question: {question}\n"
            "Identify the following for metadata filtering: company, fiscal_period, year.\n"
            "CRITICAL RULES:\n"
            "1. If a year is not EXPLICITLY mentioned, set 'year' to null.\n"
            "2. If the user is comparing multiple periods (e.g., 'Q3 vs Q4'), set 'fiscal_period' to null.\n"
            "3. Only extract the primary subject company.\n"
            "Return valid JSON only. Use null for missing values."
        )

        try:
            response = await Settings.llm.acomplete(prompt)
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            print(f"Query intent extraction failed: {e}")

        return {}
