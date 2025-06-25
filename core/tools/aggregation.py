import chromadb
import logging
import json
from google.generativeai import GenerativeModel
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import numpy as np
from typing import Dict, List, Union, Any
from collections import Counter

# Import các tool 
from core.tools.long_term_memory_retrieval import long_term_memory_retrieval_tool_function
from core.tools.knowledge_retrieval import knowledge_retrieval_tool_function

logger = logging.getLogger(__name__)
load_dotenv()


UNKNOWN_CONFIDENCE_THRESHOLD = 0.65

LABEL_MAP_AGGREGATION = {
    "ARP Spoofing": 0, "Benign": 1, "DNS Flood": 2, "Dictionary Attack": 3,
    "ICMP Flood": 4, "OS Scan": 5, "Ping Sweep": 6, "Port Scan": 7, "SYN Flood": 8,
    "Slowloris": 9, "UDP Flood": 10, "Vulnerability Scan": 11,
}
EXPECTED_NUM_CLASSES = len(LABEL_MAP_AGGREGATION)

CLASSIFIER_WEIGHTS = {
    "random_forest_ACI": 1.0, "decision_tree_ACI": 0.9,
    "multi_layer_perceptrons_ACI": 0.85, "svc_ACI": 0.8,
    "k_nearest_neighbors_ACI": 0.7, "logistic_regression_ACI": 0.4
}

LTM_SUPPORT_BONUS = 0.15 # Điểm thưởng cho nhãn được LTM ủng hộ

class AggregationTool:
    def __init__(self,
                 collection_name_local_kb="iot_attacks_knowledge_base",
                 embedding_model_name="all-MiniLM-L6-v2",
                 db_path_local_kb="D:/UIT 2025-2026/Hocmay/Project/IDS_Agent/ChromaDB_Knowledge/",
                 llm_model_name="gemini-1.5-flash-latest"):
        
        self.encoder = None
        self.llm_summarizer = None
        self.local_kb_collection = None
        
        try:
            self.encoder = SentenceTransformer(embedding_model_name)
            logger.info(f"AggregationTool: Initialized SentenceTransformer '{embedding_model_name}'.")
        except Exception as e:
            logger.error(f"AggregationTool: Failed to initialize SentenceTransformer: {e}", exc_info=True)

        try:
            self.llm_summarizer = GenerativeModel(llm_model_name)
            logger.info(f"AggregationTool: Initialized Gemini LLM Summarizer '{llm_model_name}'.")
        except Exception as e:
            logger.error(f"AggregationTool: Failed to initialize Gemini LLM Summarizer: {e}", exc_info=True)
            
        try:
            if db_path_local_kb:
                os.makedirs(db_path_local_kb, exist_ok=True)
                client_local_kb = chromadb.PersistentClient(path=db_path_local_kb)
                self.local_kb_collection = client_local_kb.get_or_create_collection(name=collection_name_local_kb)
                logger.info(f"AggregationTool: Connected to local KB ChromaDB collection '{collection_name_local_kb}' at '{db_path_local_kb}'.")
        except Exception as e:
            logger.error(f"AggregationTool: Failed to initialize local KB ChromaDB: {e}", exc_info=True)

    def summarize_documents_with_llm(self, documents: List[str], query_context: str = "") -> str:
        if not self.llm_summarizer:
            logger.error("LLM Summarizer not initialized for AggregationTool.")
            return "Summary unavailable (LLM not ready)."
        if not documents:
            return "No documents provided for summary."
        try:
            joined_documents = "\n---\n".join(documents)
            prompt = f"Given the query context '{query_context}', concisely summarize the key information relevant to intrusion detection from the following documents in 50 words or less:\n\n{joined_documents}"
            
            safety_settings_summarizer = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            response = self.llm_summarizer.generate_content(prompt, safety_settings=safety_settings_summarizer)
            return response.text.strip() if response.text else "LLM summary was empty."
        except Exception as e:
            logger.error(f"Failed to summarize documents with LLM: {e}", exc_info=True)
            return f"Summary failed: {str(e)}"

    async def aggregate_results(self,
                                classification_results: Dict[str, Dict],
                                memory_retrieval_result: Dict,
                                line_number: int,
                                reasoning_trace: Union[List[str], str]
                               ) -> Dict[str, Any]:
        logger.info(f"--- Starting Aggregation for Line {line_number} ---")
        try:
            if not classification_results:
                logger.error("No classification results provided for aggregation.")
                return {"error": "No classification results provided."}

            # KIỂM TRA ĐIỀU KIỆN "UNKNOWN ATTACK"
            low_confidence_count = 0
            low_confidence_details = []
            num_valid_classifiers = 0

            for classifier_name, result in classification_results.items():
                if "error" in result or not result.get("probabilities"):
                    continue
                
                num_valid_classifiers += 1
                max_confidence = max(result.get("probabilities", [0.0]))
                
                if max_confidence < UNKNOWN_CONFIDENCE_THRESHOLD:
                    low_confidence_count += 1
                    top1_label = result.get("predicted_label_top_1", "N/A")
                    low_confidence_details.append(f"{classifier_name} (predicted '{top1_label}' with confidence {max_confidence:.2f})")

            # Kiểm tra nếu số lượng low-confidence VƯỢT QUÁ 50% số classifier hợp lệ
            if num_valid_classifiers > 0 and (low_confidence_count / num_valid_classifiers) > 0.5:
                logger.warning(f"UNKNOWN ATTACK condition met for line {line_number}. "
                               f"{low_confidence_count}/{num_valid_classifiers} classifiers ({low_confidence_count / num_valid_classifiers:.2%}) "
                               f"had confidence < {UNKNOWN_CONFIDENCE_THRESHOLD}.")
                analysis = (
                    f"Decision: Classified as 'Unknown'. Reason: {low_confidence_count} out of {num_valid_classifiers} "
                    f"classifiers showed low confidence (below {UNKNOWN_CONFIDENCE_THRESHOLD}), which exceeds the 50% threshold. "
                    f"This indicates a potential zero-day attack or an anomalous traffic pattern. "
                    f"Low-confidence classifiers: {'; '.join(low_confidence_details)}."
                )
                return {
                    "line_number": line_number,
                    "analysis": analysis,
                    "predicted_label_top_1": "Unknown",
                    "predicted_label_top_2": "N/A",
                    "predicted_label_top_3": "N/A" 
                }

            # TÍNH ĐIỂM CÓ TRỌNG SỐ BAN ĐẦU
            all_top_labels_from_classifiers = [res["predicted_label_top_1"] for res in classification_results.values() if "error" not in res and res.get("predicted_label_top_1")]
            unique_labels_for_scoring = set(all_top_labels_from_classifiers)
            if not unique_labels_for_scoring:
                 return {"error": "No labels available for scoring."}

            weighted_scores = {label: 0.0 for label in unique_labels_for_scoring}
            logger.info(f"Calculating initial weighted_scores for: {unique_labels_for_scoring}")

            for name, res in classification_results.items():
                if "error" in res: continue
                weight = CLASSIFIER_WEIGHTS.get(name, 1.0)
                probs = res.get("probabilities", [])
                if not probs or len(probs) != EXPECTED_NUM_CLASSES: continue
                for label, idx in LABEL_MAP_AGGREGATION.items():
                    if label in weighted_scores:
                        weighted_scores[label] += float(probs[idx]) * weight
            
            logger.info(f"Initial weighted_scores: {weighted_scores}")

            # SỬ DỤNG LTM ĐỂ CỘNG ĐIỂM THƯỞNG
            ltm_supported_label = None
            if memory_retrieval_result and memory_retrieval_result.get("previous_results"):
                ltm_labels = [entry["final_label"] for entry in memory_retrieval_result["previous_results"]]
                if ltm_labels:
                    ltm_vote_counts = Counter(ltm_labels)
                    most_common_in_ltm = ltm_vote_counts.most_common(1)
                    if most_common_in_ltm:
                        ltm_supported_label = most_common_in_ltm[0][0]
                        if ltm_supported_label in weighted_scores:
                            logger.info(f"LTM strongly supports '{ltm_supported_label}'. Applying LTM_SUPPORT_BONUS of {LTM_SUPPORT_BONUS}.")
                            weighted_scores[ltm_supported_label] += LTM_SUPPORT_BONUS
                        else:
                            logger.info(f"LTM supports '{ltm_supported_label}', but it's not in current predictions. No bonus applied.")
            
            # TRUY XUẤT KIẾN THỨC BỔ SUNG
            knowledge_query_main = " ".join(unique_labels_for_scoring) + " attack characteristics"
            local_kb_summary = "No local KB queried or no relevant info."
            if self.local_kb_collection and self.encoder:
                try:
                    query_embedding = self.encoder.encode(knowledge_query_main).tolist()
                    local_docs_results = self.local_kb_collection.query(query_embeddings=[query_embedding], n_results=2)
                    local_docs = local_docs_results["documents"][0] if local_docs_results and local_docs_results["documents"] else []
                    if local_docs:
                        local_kb_summary = self.summarize_documents_with_llm(local_docs, knowledge_query_main)
                except Exception as e:
                    logger.error(f"Error querying local KB for aggregation: {e}", exc_info=True)
            
            external_knowledge_main_result = await knowledge_retrieval_tool_function(knowledge_query_main)
            external_summary_main = external_knowledge_main_result.get("retrieved_knowledge", "No external knowledge retrieved.")

            #  SẮP XẾP VÀ TRẢ VỀ KẾT QUẢ CUỐI CÙNG
            logger.info(f"Final weighted_scores after LTM bonus: {weighted_scores}")
            sorted_labels = sorted(weighted_scores.keys(), key=lambda label: weighted_scores[label], reverse=True)

            final_reasoning_trace = reasoning_trace if isinstance(reasoning_trace, str) else " -> ".join(reasoning_trace)
            
            analysis_parts = [
                f"Analyzed line {line_number} using classifiers: {', '.join(classification_results.keys())}.",
                f"LTM support detected for: {ltm_supported_label or 'N/A'}.",
                "Sensitivity profile: Balanced.",
                f"LTM (Retrieved {len(memory_retrieval_result.get('previous_results', []))} entries): {json.dumps(memory_retrieval_result, ensure_ascii=False, indent=None)}.",
                f"Raw Classification Results: {json.dumps(classification_results, ensure_ascii=False, indent=None)}.",
                f"Final Weighted Scores (with bonuses): {json.dumps({k: round(v, 4) for k, v in weighted_scores.items()}, ensure_ascii=False, indent=None)}.",
                f"Local KB Summary (Query: '{knowledge_query_main}'): {local_kb_summary}.",
                f"External Knowledge Summary (Query: '{knowledge_query_main}'): {external_summary_main[:500]}..." if external_summary_main else "No external knowledge.",
                f"Agent Reasoning Trace Summary: {final_reasoning_trace}."
            ]
            analysis = " ".join(analysis_parts)
            
            logger.info(f"--- Aggregation Complete for Line {line_number} ---")
            return {
                "line_number": line_number,
                "analysis": analysis,
                "predicted_label_top_1": sorted_labels[0] if len(sorted_labels) > 0 else "Unknown",
                "predicted_label_top_2": sorted_labels[1] if len(sorted_labels) > 1 else "Unknown",
                "predicted_label_top_3": sorted_labels[2] if len(sorted_labels) > 2 else "Unknown"
            }
        
        except Exception as e:
            logger.error(f"Aggregation failed catastrophically: {e}", exc_info=True)
            return {"error": f"Critical aggregation failure: {str(e)}"}

# --- HÀM WRAPPER CHO TOOL (Singleton Pattern) ---
_aggregation_tool_instance = None

def get_aggregation_tool_instance():
    global _aggregation_tool_instance
    if _aggregation_tool_instance is None:
        logger.info("Creating new AggregationTool instance.")
        _aggregation_tool_instance = AggregationTool()
        if not _aggregation_tool_instance.llm_summarizer or not _aggregation_tool_instance.encoder:
            logger.error("AggregationTool instance failed to initialize properly (LLM or encoder missing).")
            _aggregation_tool_instance = None 
    return _aggregation_tool_instance

async def aggregate_results_tool_function(
    classification_results: Dict[str, Dict],
    memory: Dict,
    line_number: int,
    reasoning_trace: Union[List[str], str]
) -> Dict[str, Any]:
    logger.debug(f"aggregate_results_tool_function called for line_number: {line_number}")
    aggregator = get_aggregation_tool_instance()
    if not aggregator:
        return {"error": "Aggregation Tool could not be initialized."}
    return await aggregator.aggregate_results(classification_results, memory, line_number, reasoning_trace)
