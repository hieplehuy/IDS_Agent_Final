# --- TOOL -----
# Cho việc tạo KB trong initialize_memory không có bước truy xuất bên ngoài
# core/tools/aggregation.py
import chromadb
import logging
import json
from sentence_transformers import SentenceTransformer
import os
import numpy as np

logger = logging.getLogger(__name__)

# --- Đường dẫn lưu trữ cho DB kiến thức của AggregationTool ---
AGGREGATION_DB_PATH = "D:/UIT 2025-2026/Hocmay/Project/IDS_Agent/ChromaDB/" # << THƯ MỤC LƯU DB KIẾN THỨC

label_map_aggregation_init = {
    # map
    0: "ARP Spoofing", 1: "Benign", 2: "DNS Flood", 3: "Dictionary Attack",
    4: "ICMP Flood", 5: "OS Scan", 6: "Ping Sweep", 7: "Port Scan",
    8: "SYN Flood", 9: "Slowloris", 10: "UDP Flood", 11: "Vulnerability Scan"
}
class Aggregation_ChromaDB:
    def __init__(self, collection_name="iot_attacks", model_name="all-MiniLM-L6-v2", db_path=AGGREGATION_DB_PATH): # Thêm db_path
        self.db_path = db_path
        try:
            os.makedirs(self.db_path, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.db_path)
            self.collection = self.client.get_or_create_collection(name=collection_name)
            logger.info(f"Aggregation_ChromaDB (Init Mode): ChromaDB collection '{collection_name}' accessed/created at persistent path '{self.db_path}'.")
        except Exception as e_chroma:
            logger.error(f"Aggregation_ChromaDB (Init Mode): Failed to initialize ChromaDB PersistentClient/collection at '{self.db_path}': {e_chroma}", exc_info=True)
            self.client = None
            self.collection = None
        
        try:
            self.encoder = SentenceTransformer(model_name)
            logger.info(f"Aggregation_ChromaDB (Init Mode): SentenceTransformer '{model_name}' loaded.")
        except Exception as e_encoder:
            logger.error(f"Aggregation_ChromaDB (Init Mode): Failed to load SentenceTransformer: {e_encoder}", exc_info=True)
            self.encoder = None
        
    def _get_text_label_from_value(self, label_value):
        if isinstance(label_value, str):
            return label_value
        if isinstance(label_value, (int, float, np.integer, np.floating)):
            return label_map_aggregation_init.get(int(label_value), f"UnknownLabel({label_value})")
        return str(label_value) if label_value is not None else None

    def summarize_documents_placeholder(self, documents: list) -> str:
        if not documents:
            return "No relevant documents found in local KB for summarization placeholder."
        summary_text = "Retrieved knowledge from local KB: " + " | ".join(doc[:100] + "..." for doc in documents)
        return summary_text
    # hàm chính 
    async def aggregate_results(self, classification_results: dict, memory: dict, line_number: int, reasoning_trace: list) -> dict:
        if not self.encoder or not self.collection:
            return {"error": "Aggregation_ChromaDB (Init Mode) encoder or collection not initialized."}
        try:
            # 1. Thu thập và làm sạch kết quả từ các classifier
            text_labels_from_classifiers = []
            valid_classifier_outputs = {}
            for classifier_name, result_dict in classification_results.items():
                if result_dict and "error" not in result_dict:
                    valid_classifier_outputs[classifier_name] = result_dict
                    top1 = self._get_text_label_from_value(result_dict.get("predicted_label_top_1"))
                    top2 = self._get_text_label_from_value(result_dict.get("predicted_label_top_2"))
                    top3 = self._get_text_label_from_value(result_dict.get("predicted_label_top_3"))
                    if top1 and not top1.startswith("UnknownLabel"): text_labels_from_classifiers.append(top1)
                    if top2 and not top2.startswith("UnknownLabel"): text_labels_from_classifiers.append(top2)
                    if top3 and not top3.startswith("UnknownLabel"): text_labels_from_classifiers.append(top3)
                else:
                    logger.warning(f"Classifier '{classifier_name}' had an error/no result in aggregation: {result_dict.get('error', 'N/A')}")
            # 2. Tạo câu truy vấn và tìm kiếm trong cơ sở tri thức (Knowledge Base)
            unique_text_labels = list(set(filter(None, text_labels_from_classifiers)))
            external_knowledge_summary = "No external knowledge processed (Init Mode)."
            if unique_text_labels:
                query_for_kb = " ".join(unique_text_labels) + " attack characteristics and detection methods"
                query_embedding = self.encoder.encode(query_for_kb).tolist()
                kb_results = self.collection.query(query_embeddings=[query_embedding], n_results=3, include=['documents'])
                retrieved_documents = kb_results["documents"][0] if kb_results.get("documents") and kb_results["documents"][0] else []
                if retrieved_documents:
                    external_knowledge_summary = self.summarize_documents_placeholder(retrieved_documents) 
                else:
                    external_knowledge_summary = "No relevant documents found in local KB (Init Mode)."
            else:
                external_knowledge_summary = "No labels from classifiers to form knowledge query (Init Mode)."
            # 3. Tổng hợp kết quả cuối cùng bằng phương pháp bỏ phiếu 
            final_predicted_labels = ["Unknown"] * 3
            if unique_text_labels:
                label_counts = {label: text_labels_from_classifiers.count(label) for label in unique_text_labels}
                sorted_unique_labels_by_count = sorted(label_counts, key=label_counts.get, reverse=True)
                for i in range(min(3, len(sorted_unique_labels_by_count))):
                    final_predicted_labels[i] = sorted_unique_labels_by_count[i]
            # 4. Tạo chuỗi phân tích tổng hợp
            analysis_text = (
                f"Analyzed line {line_number} (Init Mode). Classifiers: {', '.join(valid_classifier_outputs.keys())}. "
                f"LTM (mock): {json.dumps(memory, ensure_ascii=False)}. "
                f"Classifier Outputs: {json.dumps(valid_classifier_outputs, ensure_ascii=False)}. "
                f"External Knowledge (placeholder): {external_knowledge_summary}. "
            )
            max_analysis_length = 500 
            if len(analysis_text) > max_analysis_length:
                analysis_text = analysis_text[:max_analysis_length] + "..."
            # 5. Trả về kết quả cuối cùng
            return {
                "line_number": line_number, "analysis": analysis_text,
                "predicted_label_top_1": final_predicted_labels[0],
                "predicted_label_top_2": final_predicted_labels[1],
                "predicted_label_top_3": final_predicted_labels[2],
                "retrieved_knowledge_summary": external_knowledge_summary,
            }
        except Exception as e:
            logger.error(f"Aggregation failed (Init Mode): {str(e)}", exc_info=True)
            return {"error": f"Unexpected aggregation error (Init Mode): {str(e)}"}