import re
import json
import chromadb
from collections import Counter
from llama_index.core import Settings
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.extractors import SummaryExtractor, KeywordExtractor
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.ingestion import IngestionPipeline


class Ingestor:
    def __init__(self, collection: chromadb.api.models.Collection.Collection):
        self.vector_store = ChromaVectorStore(chroma_collection=collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

    async def ingest_text(self, text: str, metadata: dict = None) -> None:
        doc = Document(text=text, metadata=metadata or {})

        pipeline = IngestionPipeline(
            transformations=[
                TokenTextSplitter(chunk_size=300, chunk_overlap=50),
                SummaryExtractor(
                    summaries=["self"],
                    prompt_template=(
                        "Context: {context_str}\n"
                        "Identify: company_name, fiscal_period, year, and 5 key topics.\n"
                        "Return valid JSON only with keys: "
                        "'company_name', 'fiscal_period', 'year', 'topics'.\n"
                        "Example: {'company_name': 'Apple', 'fiscal_period': 'Q1', 'year': '2025', 'topics': ['iPhone', 'AI', 'Services']}\n"
                        "Do not write any prose."
                    )
                ),
                # KeywordExtractor(keywords=5),
                Settings.embed_model,
            ]
        )

        nodes = await pipeline.arun(documents=[doc], num_workers=1)
        sanitised_nodes = self.sanitize_metadata(nodes)
        await self.vector_store.async_add(sanitised_nodes)

        print(f"Successfully ingested document: {metadata.get('filename')}")

    def sanitize_metadata(self, nodes: list) -> list:
        company_votes = []
        period_votes = []
        year_votes = []

        for node in nodes:
            summary = node.metadata.get("section_summary", "")
            if not summary:
                continue

            try:
                json_match = re.search(r'\{.*\}', summary, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    c_name = data.get("company_name")
                    f_period = data.get("fiscal_period")
                    f_year = data.get("year")

                    if c_name and str(c_name).lower() != "null":
                        company_votes.append(str(c_name))
                    if f_period and str(f_period).lower() != "null":
                        period_votes.append(str(f_period))
                    if f_year and str(f_year).lower() != "null":
                        year_votes.append(str(f_year))

                    topics = data.get("topics", [])
                    if isinstance(topics, list):
                        node.metadata["keywords"] = ", ".join([str(t) for t in topics])
                    else:
                        node.metadata["keywords"] = str(topics)
            except Exception as e:
                print(f"DEBUG: Failed to parse metadata for a node: {e}")

        winner_company = Counter(company_votes).most_common(1)[0][0] if company_votes else "Unknown"
        winner_period = Counter(period_votes).most_common(1)[0][0] if period_votes else "Unknown"
        winner_year = Counter(year_votes).most_common(1)[0][0] if year_votes else "Unknown"

        for node in nodes:
            node.metadata["company"] = winner_company
            node.metadata["fiscal_period"] = winner_period
            node.metadata["year"] = winner_year

            if "section_summary" in node.metadata:
                del node.metadata["section_summary"]

        # --- Final Metadata Audit Log ---
        print(f"\n{'='*60}")
        print(f"DOCUMENT CONSENSUS:")
        print(f"  - Company: {winner_company}")
        print(f"  - Period:  {winner_period}")
        print(f"  - Year:    {winner_year}")
        print(f"{'-'*60}")
        print(f"{'NODE':<6} | {'KEYWORDS/TOPICS':<50}")
        print(f"{'-'*60}")

        for i, node in enumerate(nodes):
            kw_str = node.metadata.get("keywords", "")

            display_kw = (kw_str[:75] + '...') if len(kw_str) > 75 else kw_str
            print(f"{i:<6} | {display_kw}")

        print(f"{'='*60}\n")
        # --- Final Metadata Audit Log End ---

        return nodes

    def ingest_documents(self, documents: list[Document]) -> None:
        VectorStoreIndex.from_documents(documents, storage_context=self.storage_context)
        print(f"Successfully ingested {len(documents)} documents.")
