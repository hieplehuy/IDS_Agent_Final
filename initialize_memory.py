# Chạy tool với định nghĩa chuỗi pine line
# Lấy 5 % dataset để tạo ra cơ sử dữ liệu knowledge nếu kết quả cuối cùng đúng thì thêm vào chromaDB

import pandas as pd
import json
import logging
import asyncio
from datetime import datetime
import os

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"Lỗi import chromadb hoặc sentence_transformers: {e}. Hãy cài đặt chúng bằng pip.")
    exit()

# --- Import tool  ---
try:
    from core.tools.long_term_memory_retrieval import LTMRetriever
    from core.tools.aggregation_ChromaDB import Aggregation_ChromaDB # <<< ĐỔI TÊN IMPORT Ở ĐÂY
    from core.tools.data_preprocessing import data_preprocessing
    from core.tools.classification import classify # File classification.py đã được sửa ở trên
except ImportError as e:
    print(f"Lỗi import: {e}. Hãy đảm bảo các file tool (LTMRetriever, AggregationTool, data_preprocessing, classify) \
có trong thư mục core/tools/ và Python có thể tìm thấy chúng.")
    exit()

# --- Cấu hình Logging ---
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# --- Label Map ---
label_map = {
    "ARP Spoofing": 0, "Benign": 1, "DNS Flood": 2, "Dictionary Attack": 3,
    "ICMP Flood": 4, "OS Scan": 5, "Ping Sweep": 6, "Port Scan": 7,
    "SYN Flood": 8, "Slowloris": 9, "UDP Flood": 10, "Vulnerability Scan": 11
}

# --- Cấu hình cho Knowledge Base của AggregationTool ---
AGGREGATION_DB_STORAGE_PATH = "./ChromaDB/Aggregation_Knowledge_storage" # << THƯ MỤC LƯU DB KIẾN THỨC
AGGREGATION_KNOWLEDGE_COLLECTION_NAME = "iot_attacks" # Giữ nguyên, khớp với Aggregation_ChromaDB
AGGREGATION_ENCODER_MODEL_NAME = "all-MiniLM-L6-v2" # Giữ nguyên, khớp với Aggregation_ChromaDB

def initialize_external_documents_for_aggregation():
    """
    Khởi tạo vector database với tài liệu mẫu cho Aggregation_ChromaDB.
    Nó sẽ ghi vào collection mà Aggregation_ChromaDB sử dụng, tại đường dẫn bền vững.
    """
    try:
        logger.info(f"Initializing external documents in collection '{AGGREGATION_KNOWLEDGE_COLLECTION_NAME}' at path '{AGGREGATION_DB_STORAGE_PATH}' for Aggregation_ChromaDB...")
        
        os.makedirs(AGGREGATION_DB_STORAGE_PATH, exist_ok=True)
        client = chromadb.PersistentClient(path=AGGREGATION_DB_STORAGE_PATH)
        
        collection = client.get_or_create_collection(name=AGGREGATION_KNOWLEDGE_COLLECTION_NAME)
        encoder = SentenceTransformer(AGGREGATION_ENCODER_MODEL_NAME)

        sample_documents = [
            "DoS attacks (Denial-of-Service attacks) attempt to make a machine or network resource unavailable to its intended users by temporarily or indefinitely interrupting or suspending services of a host connected to the Internet. They are typically accomplished by flooding the targeted machine or resource with superfluous requests in an attempt to overload systems and prevent some or all legitimate requests from being fulfilled.",
            "Benign network traffic refers to the normal, legitimate data flow within a computer network. It is characterized by expected patterns, protocols, and data volumes that do not pose a security threat. Distinguishing benign traffic from malicious traffic is a primary goal of Intrusion Detection Systems.",
            "A SYN Flood is a form of denial-of-service attack in which an attacker sends a succession of SYN requests to a target's system in an attempt to consume enough server resources to make the system unresponsive to legitimate traffic. It exploits part of the normal three-way handshake used to establish a TCP connection.",
            "Port scanning is a method for determining which ports on a network are open and could be receiving or sending data. It is also a process for sending packets to specific ports on a host and analyzing responses to identify vulnerabilities. While system administrators can use it to verify security policies, attackers use it to identify running services and potential entry points into a system."
        ]
        doc_ids = [f"agg_doc_{i}" for i in range(len(sample_documents))]
        existing_docs_result = collection.get(ids=doc_ids, include=[])
        existing_ids_set = set(existing_docs_result['ids'])
        new_docs_to_add_content = []
        new_ids_to_add = []
        for i, content in enumerate(sample_documents):
            current_id = doc_ids[i]
            if current_id not in existing_ids_set:
                new_docs_to_add_content.append(content)
                new_ids_to_add.append(current_id)
        if new_docs_to_add_content:
            embeddings = encoder.encode(new_docs_to_add_content).tolist()
            collection.add(documents=new_docs_to_add_content, embeddings=embeddings, ids=new_ids_to_add)
            logger.info(f"Added {len(new_docs_to_add_content)} new external documents to Aggregation_ChromaDB's collection '{AGGREGATION_KNOWLEDGE_COLLECTION_NAME}' at '{AGGREGATION_DB_STORAGE_PATH}'.")
        else:
            logger.info(f"All sample external documents already exist in '{AGGREGATION_KNOWLEDGE_COLLECTION_NAME}' at '{AGGREGATION_DB_STORAGE_PATH}'. No new documents added.")

    except Exception as e:
        logger.error(f"Failed to initialize external documents for Aggregation_ChromaDB: {str(e)}", exc_info=True)


async def initialize_ltm_with_conditional_save(validation_file: str):
    """
    Khởi tạo Long-term Memory (LTM) từ tập xác thực.
    Sử dụng Aggregation_ChromaDB (phiên bản không LLM summary) để có dự đoán.
    Chỉ lưu vào LTM nếu dự đoán khớp nhãn gốc.
    """

    try:
        df = pd.read_csv(validation_file)
        logger.info(f"Read validation file: {validation_file} with {len(df)} rows.")
    except FileNotFoundError:
        logger.error(f"Validation file not found: {validation_file}")
        return
    except Exception as e:
        logger.error(f"Error reading validation file {validation_file}: {e}", exc_info=True)
        return
        
    # LTMRetriever sử dụng đường dẫn mặc định LTM_DB_PATH (./ChromaDB/LTM_storage)
    ltm_retriever = LTMRetriever() 
    # Aggregation_ChromaDB sử dụng đường dẫn mặc định AGGREGATION_DB_PATH (./ChromaDB/Aggregation_Knowledge_storage)
    aggregation_tool_instance = Aggregation_ChromaDB()

    added_to_ltm_count = 0
    processed_rows_count = 0

    feature_names_for_x_vector = [
       'Src Port', 'Dst Port', 'Protocol', 'Flow IAT Mean', 'Flow IAT Max',
       'Flow IAT Min', 'Fwd IAT Mean', 'Fwd IAT Max', 'Fwd IAT Min',
       'SYN Flag Count', 'RST Flag Count', 'Down/Up Ratio',
       'Subflow Fwd Packets', 'FWD Init Win Bytes', 'Fwd Seg Size Min',
       'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Flow Bytes/s',
       'Connection Type'
    ]
    if len(feature_names_for_x_vector) != 20: 
        logger.critical(f"feature_names_for_x_vector length is {len(feature_names_for_x_vector)}, expected 20.")

    classifier_model_names = ["logistic_regression_ACI", "decision_tree_ACI", "k_nearest_neighbors_ACI", "multi_layer_perceptrons_ACI", "random_forest_ACI", "svc_ACI" ]

    for idx, row in df.iterrows():
        processed_rows_count += 1
        if processed_rows_count % 50 == 0:
            logger.info(f"Processing row {processed_rows_count}/{len(df)}...")

        raw_data_dict = row.to_dict()
        original_raw_data_for_observations = raw_data_dict.copy()
        ground_truth_text_label = raw_data_dict.get("Label")
        
        preprocessed_data_output = await data_preprocessing(raw_data_dict.copy())
        if "error" in preprocessed_data_output:
            logger.warning(f"Row {idx} (LTM Init): Preprocessing failed - {preprocessed_data_output['error']}")
            continue
        
        features_for_classification = {
            k: v for k, v in preprocessed_data_output.items() if k != 'Label'
        }

        classification_step_outputs = {}
        for model_name in classifier_model_names:
            result = await classify(model_name, features_for_classification.copy())
            classification_step_outputs[model_name] = result 
            if "error" in result:
                 logger.warning(f"Row {idx} (LTM Init), Model {model_name}: Classification error - {result['error']}")
        
        agent_final_prediction_output_dict = None
        agent_final_predicted_text_label = None
        mock_ltm_retrieval_for_aggregation = {"previous_results": []} 
        simulated_reasoning_trace_for_aggregation = ["data_extraction_sim", "preprocessing_sim", "classification_sim"]

        try:
            agent_final_prediction_output_dict = await aggregation_tool_instance.aggregate_results(
                classification_results=classification_step_outputs,
                memory=mock_ltm_retrieval_for_aggregation, 
                line_number=int(idx),
                reasoning_trace=simulated_reasoning_trace_for_aggregation
            )
            if "error" in agent_final_prediction_output_dict:
                logger.warning(f"Row {idx} (LTM Init): Aggregation failed - {agent_final_prediction_output_dict['error']}")
                continue
            agent_final_predicted_text_label = agent_final_prediction_output_dict.get("predicted_label_top_1")
        except Exception as agg_ex:
            logger.error(f"Row {idx} (LTM Init): Exception during aggregation - {agg_ex}", exc_info=True)
            continue 
            
        if agent_final_predicted_text_label is None:
            logger.warning(f"Row {idx} (LTM Init): Aggregation_ChromaDB did not provide 'predicted_label_top_1'. Output: {agent_final_prediction_output_dict}")
            if ground_truth_text_label is None:
                 logger.info(f"Row {idx} (LTM Init): No ground truth. Agent prediction from aggregation N/A. Not added to LTM.")
            continue

        if ground_truth_text_label is not None:
            if agent_final_predicted_text_label == ground_truth_text_label:
                reasoning_trace_simplified_for_ltm = ["data_extraction_sim", "preprocessing_sim", "classification_sim", "aggregation_placeholder_summary", "correct_decision_for_ltm"]
                actions_simplified_for_ltm = reasoning_trace_simplified_for_ltm
                feature_vector_x_for_ltm = [float(features_for_classification.get(f_name, 0.0)) for f_name in feature_names_for_x_vector]
                observations_for_ltm_entry = [
                    original_raw_data_for_observations, preprocessed_data_output, 
                    classification_step_outputs, agent_final_prediction_output_dict 
                ]
                entry_to_save_in_ltm = {
                    "t": datetime.now().isoformat(), "x": feature_vector_x_for_ltm,
                    "R": reasoning_trace_simplified_for_ltm, "A": actions_simplified_for_ltm,
                    "O": observations_for_ltm_entry, "ŷ": agent_final_predicted_text_label 
                }
                try:
                    ltm_retriever.add_ltm_entry(entry_to_save_in_ltm)
                    added_to_ltm_count += 1
                except Exception as e_add:
                    logger.error(f"Row {idx} (LTM Init): Failed to add correct prediction to LTM - {e_add}", exc_info=True)
        
    logger.info(f"LTM conditional initialization finished. Processed rows: {processed_rows_count}. Entries added to LTM: {added_to_ltm_count}.")

if __name__ == "__main__":

    validation_csv_path = "train/sample_5_percent.csv"
    
    logger.info("=== Starting Initialization Process ===")
    
    logger.info("--- Initializing External Documents for Aggregation_ChromaDB ---")
    initialize_external_documents_for_aggregation() # Lưu vào ./ChromaDB/Aggregation_Knowledge_storage
    
    logger.info("--- Initializing Long-Term Memory (LTM) Conditionally ---")
    # LTMRetriever sẽ lưu vào ./ChromaDB/LTM_storage
    asyncio.run(initialize_ltm_with_conditional_save(validation_csv_path))
    
    logger.info("=== Initialization Process Complete ===")