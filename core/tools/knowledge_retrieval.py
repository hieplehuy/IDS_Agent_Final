import logging
import chromadb
from sentence_transformers import SentenceTransformer
import requests
import json
import urllib.parse
import feedparser  # Cho ArXiv API
import os
logger = logging.getLogger(__name__)

# Cấu hình chung
KNOWLEDGE_COLLECTION_NAME = "external_documents_collection"
SENTENCE_TRANSFORMER_MODEL_NAME = "all-MiniLM-L6-v2"

class KnowledgeRetrievalTool:
    def __init__(self,
                 collection_name: str = KNOWLEDGE_COLLECTION_NAME,
                 embedding_model_name: str = SENTENCE_TRANSFORMER_MODEL_NAME,
                 serpapi_key: str = None):
        self.collection_name = collection_name
        self.encoder = None
        self.collection = None
        self.serpapi_key = serpapi_key or os.getenv("SERP_API_KEY")  # Lấy từ .env
        try:
            self.encoder = SentenceTransformer(embedding_model_name)
            self.client = chromadb.Client()
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"Connected to collection '{self.collection_name}'.")
            except Exception:
                logger.warning(f"Collection '{self.collection_name}' not found. Creating it.")
                self.collection = self.client.get_or_create_collection(name=self.collection_name)
                logger.info(f"Created/accessed collection '{self.collection_name}'.")
        except Exception as e:
            logger.error(f"Failed to initialize SentenceTransformer or ChromaDB: {e}")

    def _query_local_kb(self, query_text: str, top_n: int = 2) -> list:
        # Truy vấn ChromaDB
        if not self.collection or not self.encoder:
            logger.error("ChromaDB or encoder not available.")
            return []
        try:
            query_embedding = self.encoder.encode(query_text).tolist()
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_n,
                include=["documents", "distances"]
            )
            retrieved_docs = results['documents'][0] if results['documents'] else []
            logger.debug(f"Local KB query for '{query_text}' returned: {retrieved_docs}")
            return retrieved_docs
        except Exception as e:
            logger.error(f"Error querying local KB: {e}")
            return []

    def _query_wikipedia_api(self, query_text: str, top_n: int = 1) -> list:
        # Truy vấn Wikipedia API (
        session = requests.Session()
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query_text,
            "srlimit": top_n,
            "utf8": 1
        }
        retrieved_summaries = []
        try:
            response = session.get(url=url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            search_results = data.get("query", {}).get("search", [])
            for item in search_results:
                page_title = item.get("title")
                extract_params = {
                    "action": "query",
                    "format": "json",
                    "titles": page_title,
                    "prop": "extracts",
                    "exintro": True,
                    "explaintext": True,
                    "utf8": 1
                }
                extract_response = session.get(url=url, params=extract_params, timeout=10)
                extract_response.raise_for_status()
                extract_data = extract_response.json()
                page_id = list(extract_data.get("query", {}).get("pages", {}).keys())[0]
                if page_id != "-1":
                    summary = extract_data["query"]["pages"][page_id].get("extract", "No summary available.")
                    retrieved_summaries.append(f"Wikipedia ({page_title}): {summary[:500]}...")
            logger.debug(f"Wikipedia query for '{query_text}' returned: {retrieved_summaries}")
            return retrieved_summaries
        except requests.RequestException as e:
            logger.error(f"Error querying Wikipedia API: {e}")
            return []
    # SerpAPI
    def _query_serpapi(self, query_text: str, top_n: int = 3) -> list:
        """Truy vấn Google Search qua SerpAPI."""
        if not self.serpapi_key:
            logger.error("SerpAPI key not provided.")
            return []
        url = "https://serpapi.com/search"
        params = {
            "q": query_text,
            "api_key": self.serpapi_key,
            "num": top_n,
            "engine": "google"
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            organic_results = data.get("organic_results", [])
            retrieved_results = []
            for result in organic_results[:top_n]:
                title = result.get("title", "No title")
                snippet = result.get("snippet", "No snippet available.")
                retrieved_results.append(f"Google ({title}): {snippet[:500]}...")
            logger.debug(f"SerpAPI query for '{query_text}' returned: {retrieved_results}")
            return retrieved_results
        except requests.RequestException as e:
            logger.error(f"Error querying SerpAPI: {e}")
            return []

    def _query_arxiv(self, query_text: str, top_n: int = 2) -> list:
        """Truy vấn ArXiv API."""
        base_url = "http://export.arxiv.org/api/query"
        query = urllib.parse.quote(f"{query_text} intrusion detection")
        params = {
            "search_query": query,
            "max_results": top_n,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            feed = feedparser.parse(response.text)
            retrieved_papers = []
            for entry in feed.entries[:top_n]:
                title = entry.get("title", "No title").replace("\n", " ")
                summary = entry.get("summary", "No summary available.").replace("\n", " ")
                retrieved_papers.append(f"ArXiv ({title}): {summary[:500]}...")
            logger.debug(f"ArXiv query for '{query_text}' returned: {retrieved_papers}")
            return retrieved_papers
        except requests.RequestException as e:
            logger.error(f"Error querying ArXiv API: {e}")
            return []

    async def retrieve_knowledge(self, query: str) -> dict:
        if not self.encoder or not self.collection:
            logger.error("Encoder or collection not initialized.")
            return {"error": "Knowledge Retrieval Tool not initialized."}
        
        logger.info(f"Retrieving knowledge for query: '{query}'")
        
        # Truy vấn
        local_kb_results = self._query_local_kb(query)
        wikipedia_results = self._query_wikipedia_api(query)
        serpapi_results = self._query_serpapi(query)
        arxiv_results = self._query_arxiv(query)
        
        # Kết hợp kết quả
        combined_knowledge_parts = []
        if local_kb_results:
            combined_knowledge_parts.extend(f"Local KB: {res}" for res in local_kb_results)
        if wikipedia_results:
            combined_knowledge_parts.extend(wikipedia_results)
        if serpapi_results:
            combined_knowledge_parts.extend(serpapi_results)
        if arxiv_results:
            combined_knowledge_parts.extend(arxiv_results)
        
        final_knowledge_output = "\n".join(combined_knowledge_parts) if combined_knowledge_parts else "No information found."
        logger.info(f"Retrieved {len(final_knowledge_output)} chars for query: '{query}'")
        return {"retrieved_knowledge": final_knowledge_output}

_knowledge_retriever_instance = None

def get_knowledge_retriever_instance():
    global _knowledge_retriever_instance
    if _knowledge_retriever_instance is None:
        _knowledge_retriever_instance = KnowledgeRetrievalTool()
    return _knowledge_retriever_instance

async def knowledge_retrieval_tool_function(query: str) -> dict:
    retriever = get_knowledge_retriever_instance()
    if not retriever or not retriever.encoder or not retriever.collection:
        logger.error("Knowledge Retrieval Tool not initialized.")
        return {"error": "Knowledge Retrieval Tool not initialized."}
    return await retriever.retrieve_knowledge(query)