import chromadb
import logging
import json
from google.generativeai import GenerativeModel # Giả sử bạn dùng Gemini trực tiếp ở đây
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import numpy as np
from typing import Dict, List, Union, Any # Thêm Any

# Import các tool 
from core.tools.long_term_memory_retrieval import long_term_memory_retrieval_tool_function
from core.tools.knowledge_retrieval import knowledge_retrieval_tool_function

logger = logging.getLogger(__name__)

# --- ĐỊNH NGHĨA CHO AGGREGATION ---
LABEL_MAP_AGGREGATION = {
    "ARP Spoofing": 0,
    "Benign": 1,
    "DNS Flood": 2,
    "Dictionary Attack": 3,
    "ICMP Flood": 4,
    "OS Scan": 5,
    "Ping Sweep": 6,
    "Port Scan": 7,
    "SYN Flood": 8,
    "Slowloris": 9,
    "UDP Flood": 10,
    "Vulnerability Scan": 11,
    # "Unknown": 12 # nếu có lớp Unknown
}
# Số lượng lớp mong đợi từ các model classify
EXPECTED_NUM_CLASSES = len(LABEL_MAP_AGGREGATION)

# Định nghĩa trọng số cho các bộ phân loại tùy chỉnh theo độ chính xác của mô hình phân loại khi train
CLASSIFIER_WEIGHTS = {
    "random_forest_ACI": 1.0,
    "decision_tree_ACI": 0.9,
    "multi_layer_perceptrons_ACI": 0.85,
    "svc_ACI": 0.8, 
    "k_nearest_neighbors_ACI": 0.7,
    "logistic_regression_ACI": 0.4
}
# Điểm thưởng cho nhãn được chọn từ LTM/Knowledge tie-breaking
LTM_KNOWLEDGE_TIE_BREAKER_BONUS = 0.5

class AggregationTool:
    def __init__(self,
                 collection_name_local_kb="iot_attacks_knowledge_base", # Collection cho kiến thức cục bộ 
                 embedding_model_name="all-MiniLM-L6-v2",
                 db_path_local_kb="D:/UIT 2025-2026/Hocmay/Project/IDS_Agent/ChromaDB_Knowledge/", # Đường dẫn riêng cho KB
                 llm_model_name="gemini-1.5-flash-latest"): # Sử dụng tên model mới hơn
        
        self.encoder = None
        self.llm_summarizer = None # LLM dùng để tóm tắt
        self.local_kb_collection = None

        try:
            self.encoder = SentenceTransformer(embedding_model_name)
            logger.info(f"AggregationTool: Initialized SentenceTransformer '{embedding_model_name}'.")
        except Exception as e:
            logger.error(f"AggregationTool: Failed to initialize SentenceTransformer: {e}", exc_info=True)

        try:
            # LLM này chỉ dùng để tóm tắt
            self.llm_summarizer = GenerativeModel(llm_model_name)
            logger.info(f"AggregationTool: Initialized Gemini LLM Summarizer '{llm_model_name}'.")
        except Exception as e:
            logger.error(f"AggregationTool: Failed to initialize Gemini LLM Summarizer: {e}", exc_info=True)
            
        try:
            if db_path_local_kb: # Chỉ khởi tạo nếu có đường dẫn
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
            # Prompt tóm tắt có thể cần ngữ cảnh của query
            prompt = f"Given the query context '{query_context}', concisely summarize the key information relevant to intrusion detection from the following documents in 50 words or less:\n\n{joined_documents}"
            
            # Cấu hình an toàn cho LLM tóm tắt
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
                                memory_retrieval_result: Dict, # Đổi tên cho rõ ràng
                                line_number: int,
                                reasoning_trace: Union[List[str], str] # Có thể là list hoặc tóm tắt string
                               ) -> Dict[str, Any]:
        logger.info(f"--- Starting Aggregation for Line {line_number} ---")
        try:
            if not classification_results:
                logger.error("No classification results provided for aggregation.")
                return {"error": "No classification results provided."}

            # Bước 1: Thu thập nhãn và chuẩn bị tính điểm
            all_top_labels_from_classifiers = []
            for classifier, result in classification_results.items():
                if "error" not in result and result.get("predicted_label_top_1"):
                    all_top_labels_from_classifiers.append(result["predicted_label_top_1"])
                    if result.get("predicted_label_top_2") and "Unknown" not in result.get("predicted_label_top_2"):
                         all_top_labels_from_classifiers.append(result.get("predicted_label_top_2"))
                    if result.get("predicted_label_top_3") and "Unknown" not in result.get("predicted_label_top_3"):
                         all_top_labels_from_classifiers.append(result.get("predicted_label_top_3"))


            if not all_top_labels_from_classifiers:
                logger.warning("No valid top labels found from classifiers. Aggregation might be unreliable.")
                # Nếu không có dự đoán nào, có thể dựa vào LTM hoặc trả về Unknown/Lỗi
                
            unique_labels_for_scoring = set(all_top_labels_from_classifiers)
            if not unique_labels_for_scoring: # Xử lý trường hợp không có nhãn nào cả
                 logger.error("No unique labels to score. Cannot proceed with weighted scoring.")
                 return {"error": "No labels available for scoring."}

            weighted_scores = {label: 0.0 for label in unique_labels_for_scoring}
            logger.info(f"Unique labels identified for scoring: {unique_labels_for_scoring}")

            # Bước 2: Phá Tie (nếu cần) và xác định nhãn ưu tiên từ LTM/Knowledge
            top_label_from_context = None # Nhãn được ưu tiên từ LTM hoặc Knowledge
            label_counts_for_tie = {label: all_top_labels_from_classifiers.count(label) for label in unique_labels_for_scoring}
            
            max_count = 0
            if label_counts_for_tie: max_count = max(label_counts_for_tie.values())
            tied_labels = [label for label, count in label_counts_for_tie.items() if count == max_count]

            if len(tied_labels) > 1 and tied_labels: # Chỉ phá tie nếu có nhiều hơn 1 và có nhãn
                logger.info(f"Tied labels based on frequency: {tied_labels}. Using LTM and external knowledge.")
                
                # Sử dụng LTM (đã được truyền vào qua memory_retrieval_result)
                ltm_previous_labels = [entry["final_label"] for entry in memory_retrieval_result.get("previous_results", [])]
                ltm_counts_for_tied = {label: ltm_previous_labels.count(label) for label in tied_labels if label in ltm_previous_labels}

                if ltm_counts_for_tied and max(ltm_counts_for_tied.values()) > 0:
                    top_label_from_context = max(ltm_counts_for_tied, key=ltm_counts_for_tied.get)
                    logger.info(f"LTM suggests: '{top_label_from_context}' from counts: {ltm_counts_for_tied}")
                else:
                    logger.info("LTM did not resolve tie. Querying external knowledge.")
                    knowledge_query_for_tie = " ".join(tied_labels) + " attack characteristics"
                    # Gọi hàm knowledge_retrieval_tool_function (đã được import)
                    knowledge_for_tie = await knowledge_retrieval_tool_function(knowledge_query_for_tie)
                    external_summary_for_tie = knowledge_for_tie.get("retrieved_knowledge", "").lower()
                    for label in tied_labels:
                        if label.lower() in external_summary_for_tie: # So khớp đơn giản
                            top_label_from_context = label
                            logger.info(f"External knowledge suggests: '{top_label_from_context}' for tie.")
                            break
                    if not top_label_from_context:
                        top_label_from_context = min(tied_labels) # Fallback
                        logger.warning(f"No LTM/Knowledge tie-breaker. Fallback to alphabetical: '{top_label_from_context}'")
            elif tied_labels: # Không có tie, chỉ có một nhãn chiếm đa số
                top_label_from_context = tied_labels[0]
                logger.info(f"Primary candidate from majority/single top: '{top_label_from_context}'")
            else:
                logger.warning("No labels to determine a top candidate from context.")


            # Bước 3: Truy xuất Kiến thức Bổ sung (có thể dựa trên top_label_from_context hoặc all_unique_labels)
            knowledge_query_main = " ".join(unique_labels_for_scoring) + " attack characteristics"
            
            # Truy vấn KB cục bộ 
            local_kb_summary = "No local KB queried or no relevant info."
            if self.local_kb_collection and self.encoder:
                try:
                    query_embedding = self.encoder.encode(knowledge_query_main).tolist()
                    local_docs_results = self.local_kb_collection.query(query_embeddings=[query_embedding], n_results=2)
                    local_docs = local_docs_results["documents"][0] if local_docs_results and local_docs_results["documents"] else []
                    if local_docs:
                        local_kb_summary = self.summarize_documents_with_llm(local_docs, knowledge_query_main)
                        logger.info(f"Local KB summary: {local_kb_summary}")
                except Exception as e:
                    logger.error(f"Error querying local KB for aggregation: {e}", exc_info=True)
            
            # Truy vấn Knowledge Retrieval tool (web, arxiv, etc.)
            external_knowledge_main_result = await knowledge_retrieval_tool_function(knowledge_query_main)
            external_summary_main = external_knowledge_main_result.get("retrieved_knowledge", "No external general knowledge retrieved.")
            logger.info(f"External general knowledge summary (first 200 chars): {external_summary_main[:200]}...")


            # Bước 4: Tính Điểm Có Trọng số
            logger.info(f"Calculating weighted_scores. Initializing for: {unique_labels_for_scoring}")
            for classifier_name, class_result in classification_results.items():
                if "error" in class_result:
                    continue
                
                current_classifier_weight = CLASSIFIER_WEIGHTS.get(classifier_name, 1.0)
                probabilities_all_classes = class_result.get("probabilities", [])

                if not probabilities_all_classes or len(probabilities_all_classes) != EXPECTED_NUM_CLASSES:
                    logger.warning(f"Classifier {classifier_name} provided incomplete/invalid probabilities (expected {EXPECTED_NUM_CLASSES}, got {len(probabilities_all_classes)}). Its probabilities will not be used for detailed scoring.")
                    top1_label = class_result.get("predicted_label_top_1")
                    if top1_label and top1_label in weighted_scores:
                        fallback_score_increase = 0.3 * current_classifier_weight 
                        weighted_scores[top1_label] += fallback_score_increase
                        logger.info(f"  Classifier {classifier_name} (fallback scoring): Label: {top1_label}, Fallback Score Inc: {fallback_score_increase:.4f}, New Total: {weighted_scores[top1_label]:.4f}")
                    continue

                for label_text, numeric_index in LABEL_MAP_AGGREGATION.items():
                    if label_text in weighted_scores: # Chỉ tính điểm cho các nhãn đã được dự đoán bởi ít nhất 1 classifier
                        prob_value = probabilities_all_classes[numeric_index]
                        prob = 0.0
                        if isinstance(prob_value, (int, float, np.number)):
                            prob = float(prob_value)
                        
                        score_increase = prob * current_classifier_weight
                        weighted_scores[label_text] += score_increase
                        logger.info(f"  Classifier {classifier_name}: Label: {label_text}, Prob: {prob:.4f}, Weight: {current_classifier_weight:.2f}, Score Inc: {score_increase:.4f}, New Total: {weighted_scores[label_text]:.4f}")
            
            # Bước 4.5: "Boost" điểm cho top_label_from_context (chỉ được thực thi khi thỏa mãn điều kiện)
            if top_label_from_context and top_label_from_context in weighted_scores:
                logger.info(f"Applying bonus score of {LTM_KNOWLEDGE_TIE_BREAKER_BONUS} to context-preferred label: '{top_label_from_context}'")
                weighted_scores[top_label_from_context] += LTM_KNOWLEDGE_TIE_BREAKER_BONUS
                logger.info(f"  New score for '{top_label_from_context}' after bonus: {weighted_scores[top_label_from_context]:.4f}")

            logger.info(f"Final weighted_scores before sorting: {weighted_scores}")
            if not weighted_scores: # Kiểm tra lại nếu rỗng
                logger.error("Weighted scores dictionary became empty. Cannot determine final labels.")
                return {"error": "No labels to score after processing probabilities."}
                
            sorted_labels = sorted(weighted_scores.keys(), key=lambda label: weighted_scores[label], reverse=True)

            # Bước 5 & 6: Tạo Phân tích và Trả về JSON
            final_reasoning_trace = reasoning_trace
            if isinstance(reasoning_trace, list): # Nếu là list, join lại
                final_reasoning_trace = " -> ".join(reasoning_trace)


            analysis_parts = [
                f"Analyzed line {line_number} using classifiers: {', '.join(classification_results.keys())}.",
                f"Initial top candidate from context (LTM/Knowledge tie-breaking): {top_label_from_context or 'N/A'}.",
                "Sensitivity profile: Balanced.",
                f"LTM (Retrieved {len(memory_retrieval_result.get('previous_results', []))} entries): {json.dumps(memory_retrieval_result, ensure_ascii=False, indent=None)}.",
                f"Raw Classification Results: {json.dumps(classification_results, ensure_ascii=False, indent=None)}.",
                f"Weighted Scores: {json.dumps({k: round(v, 4) for k, v in weighted_scores.items()}, ensure_ascii=False, indent=None)}.",
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