# core/tools/long_term_memory_retrieval.py
import chromadb
import logging
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
import json
import os # << THÊM IMPORT OS

logger = logging.getLogger(__name__)

# --- Đường dẫn lưu trữ cho LTM ChromaDB ---
LTM_DB_PATH = "D:/UIT 2025-2026/Hocmay/Project/IDS_Agent/ChromaDB/" # THƯ MỤC LƯU LTM

def convert_numpy_to_python_native(data):
    if isinstance(data, list):
        return [convert_numpy_to_python_native(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_numpy_to_python_native(value) for key, value in data.items()}
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.bool_):
        return bool(data)
    return data

class LTMRetriever:
    def __init__(self, collection_name="ltm_iot", model_name="all-MiniLM-L6-v2", db_path=LTM_DB_PATH): # Thêm db_path
        self.db_path = db_path
        try:
            os.makedirs(self.db_path, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.db_path)
            self.collection = self.client.get_or_create_collection(name=collection_name)
            logger.info(f"LTMRetriever: ChromaDB collection '{collection_name}' accessed/created at persistent path '{self.db_path}'.")
        except Exception as e:
            logger.error(f"LTMRetriever: Failed to initialize ChromaDB PersistentClient/collection at '{self.db_path}': {e}", exc_info=True)
            self.client = None
            self.collection = None

        try:
            self.encoder = SentenceTransformer(model_name)
            logger.info(f"LTMRetriever: SentenceTransformer model '{model_name}' loaded.")
        except Exception as e:
            logger.error(f"LTMRetriever: Failed to load SentenceTransformer model '{model_name}': {e}", exc_info=True)
            self.encoder = None

        self.lambda1 = 0.5
        self.lambda2 = 0.5

    def add_ltm_entry(self, entry: dict):
        if not self.collection or not self.encoder:
            logger.error("LTMRetriever not properly initialized (collection or encoder missing). Cannot add entry.")
            return
        try:
            entry_serializable = convert_numpy_to_python_native(entry)
            observations_str = json.dumps(entry_serializable["O"])
            embedding = self.encoder.encode(observations_str).tolist()
            metadata_for_chroma = {
                "timestamp": entry_serializable["t"],
                "features": json.dumps(entry_serializable["x"]),
                "reasoning_trace": json.dumps(entry_serializable["R"]),
                "actions": json.dumps(entry_serializable["A"]),
                "final_label": entry_serializable["ŷ"]
            }
            self.collection.add(
                documents=[observations_str],
                embeddings=[embedding],
                metadatas=[metadata_for_chroma],
                ids=[f"ltm_{entry_serializable['t']}"]
            )
            logger.info(f"Added LTM entry with timestamp {entry_serializable['t']} to persistent storage.")
        except Exception as e:
            entry_summary_for_log = {k: str(v)[:200] + '...' if len(str(v)) > 200 else v for k, v in entry.items()}
            logger.error(f"Failed to add LTM entry to persistent storage. Error: {str(e)}. Entry (summary): {json.dumps(entry_summary_for_log, default=str)}", exc_info=True)

    async def retrieve_memory(self, classifier_names: list, classification_results: dict) -> dict:
        if not self.collection or not self.encoder:
            logger.error("LTMRetriever not properly initialized. Cannot retrieve memory.")
            return {"error": "LTMRetriever not properly initialized."}
        try:
            classification_results_serializable = convert_numpy_to_python_native(classification_results)
            observations_str_for_query = json.dumps(classification_results_serializable)
            query_embedding = self.encoder.encode(observations_str_for_query).tolist()
            
            current_time_iso = datetime.now().isoformat()
            current_timestamp_float = datetime.fromisoformat(current_time_iso).timestamp()

            all_entries = self.collection.get(include=["metadatas", "embeddings"])
            
            if not all_entries["ids"]:
                return {"previous_results": []}

            scores = []
            entry_timestamps = [datetime.fromisoformat(m["timestamp"]).timestamp() for m in all_entries["metadatas"]]
            if not entry_timestamps:
                 return {"previous_results": []}

            time_differences = [abs(current_timestamp_float - et) for et in entry_timestamps]
            max_time_diff_value = max(time_differences) if time_differences else 0

            for i, metadata_item in enumerate(all_entries["metadatas"]):
                entry_ts_float = entry_timestamps[i]
                time_diff_value = time_differences[i]
                recency_score = 1.0 - (time_diff_value / max_time_diff_value) if max_time_diff_value > 0 else 1.0
                entry_embedding_vector = all_entries["embeddings"][i]
                norm_query_emb = np.linalg.norm(query_embedding)
                norm_entry_emb = np.linalg.norm(entry_embedding_vector)
                if norm_query_emb == 0 or norm_entry_emb == 0:
                    cosine_similarity_score = 0.0
                else:
                    cosine_similarity_score = np.dot(query_embedding, entry_embedding_vector) / (norm_query_emb * norm_entry_emb)
                combined_score = self.lambda1 * recency_score + self.lambda2 * cosine_similarity_score
                scores.append((combined_score, metadata_item))

            scores.sort(key=lambda x: x[0], reverse=True)
            top_k_retrieved = scores[:5]
            
            previous_results_output = []
            for _, retrieved_entry_metadata in top_k_retrieved:
                try:
                    retrieved_features = json.loads(retrieved_entry_metadata["features"])
                    previous_results_output.append({
                        "timestamp": retrieved_entry_metadata["timestamp"],
                        "features": retrieved_features,
                        "final_label": retrieved_entry_metadata["final_label"]
                    })
                except json.JSONDecodeError as je:
                    logger.error(f"Failed to parse 'features' from retrieved LTM metadata: {je}. Metadata: {retrieved_entry_metadata}")
                except Exception as e_parse:
                    logger.error(f"Error processing retrieved LTM entry: {e_parse}. Metadata: {retrieved_entry_metadata}")
            return {"previous_results": previous_results_output}
        except Exception as e:
            logger.error(f"Memory retrieval failed: {str(e)}", exc_info=True)
            return {"error": f"Memory retrieval failed: {str(e)}"}
        
_ltm_retriever_instance = None

def get_ltm_retriever_instance():
    """Gets a singleton instance of LTMRetriever."""
    global _ltm_retriever_instance
    if _ltm_retriever_instance is None:
        _ltm_retriever_instance = LTMRetriever()
        # Kiểm tra sau khi khởi tạo
        if not _ltm_retriever_instance.collection or not _ltm_retriever_instance.encoder:
            logger.error("LTMRetriever instance could not be properly initialized. Collection or encoder is missing.")
            _ltm_retriever_instance = None # Reset nếu lỗi
    return _ltm_retriever_instance

async def long_term_memory_retrieval_tool_function(classifier_names: list, classification_results: dict) -> dict:
    """
    Tool function for retrieving long-term memory.
    This is the function to be imported and called by the main agent.
    """
    logger.debug(f"LTM tool function called with classifier_names: {classifier_names}, classification_results: {bool(classification_results)}")
    retriever = get_ltm_retriever_instance()
    if not retriever: # Kiểm tra nếu get_ltm_retriever_instance trả về None do lỗi khởi tạo
        logger.error("Failed to get or initialize LTMRetriever instance in tool function.")
        return {"error": "LTM Retriever not properly initialized or failed to initialize."}
    return await retriever.retrieve_memory(classifier_names, classification_results)

