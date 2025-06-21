# Bên trong file: core/tools/classification.py
import pickle
import numpy as np
import os
from typing import Dict, List, Union
import logging
import pandas as pd 
logger = logging.getLogger(__name__)

# --- DEFINISI LABEL MAP DAN REVERSE LABEL MAP ---
LABEL_MAP_CLASSIFICATION = {
    "ARP Spoofing": 0, "Benign": 1, "DNS Flood": 2, "Dictionary Attack": 3,
    "ICMP Flood": 4, "OS Scan": 5, "Ping Sweep": 6, "Port Scan": 7,
    "SYN Flood": 8, "Slowloris": 9, "UDP Flood": 10, "Vulnerability Scan": 11
}
REVERSE_LABEL_MAP_CLASSIFICATION = {v: k for k, v in LABEL_MAP_CLASSIFICATION.items()}

# --- DANH SÁCH 20 FEATURES CHÍNH XÁC MÀ MODEL ĐÃ HỌC, THEO ĐÚNG THỨ TỰ ---
EXPECTED_FEATURE_ORDER_FOR_MODELS = [
    'Src Port', 'Dst Port', 'Protocol', 'Flow IAT Mean', 'Flow IAT Max',
    'Flow IAT Min', 'Fwd IAT Mean', 'Fwd IAT Max', 'Fwd IAT Min',
    'SYN Flag Count', 'RST Flag Count', 'Down/Up Ratio',
    'Subflow Fwd Packets', 'FWD Init Win Bytes', 'Fwd Seg Size Min',
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min',
    'Connection Type' 
]
# Khởi tạo và tải mô hình 
class ClassificationTool:
    def __init__(self, model_dir: str = 'D:/UIT 2025-2026/Hocmay/Project/IDS_Agent/models/ACI_IoT_23/'): # Cập nhật đường dẫn nếu cần
        self.model_dir = model_dir
        self.loaded_models: Dict[str, object] = {}

    def _load_model_from_file(self, model_name: str) -> object: 
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        # Kiểm tra xem model_dir có tồn tại không
        if not os.path.isdir(self.model_dir):
            logger.error(f"Model directory not found: {self.model_dir}")
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        model_path = os.path.join(self.model_dir, f'{model_name}.pkl')
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path} for model name {model_name}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                self.loaded_models[model_name] = model
                logger.info(f"Model '{model_name}' loaded successfully from {model_path}")
                return model
        except FileNotFoundError: # Bắt lỗi cụ thể hơn
            logger.error(f"Could not find model file: {model_path} for model '{model_name}'")
            raise # Ném lại lỗi để hàm gọi xử lý
        except Exception as e:
            logger.error(f"Error loading model '{model_name}' from {model_path}: {e}", exc_info=True)
            raise # Ném lại lỗi

    # Chuẩn bị Dữ liệu đầu vào cho Mô hình
    def _prepare_input_features(self, features_dict: dict, model_name_for_log: str) -> Union[np.ndarray, None]:
        """Chuẩn bị mảng NumPy đầu vào cho model từ dictionary features, theo đúng thứ tự."""
        try:
            feature_values = []
            missing_features = []
            for feature_name in EXPECTED_FEATURE_ORDER_FOR_MODELS:
                value = features_dict.get(feature_name)
                if value is None:
                    # Ghi log feature bị thiếu, và có thể gán giá trị mặc định nếu phù hợp
                    # HOẶC báo lỗi nếu tất cả features là bắt buộc.
                    # logger.warning(f"Model {model_name_for_log}: Missing feature '{feature_name}', using 0.0 as default.")
                    missing_features.append(feature_name)
                    feature_values.append(0.0) # Tạm thời dùng 0.0, CẨN THẬN!
                else:
                    feature_values.append(value)
            
            if missing_features:
                logger.warning(f"Model {model_name_for_log}: Missing features {missing_features} from input dictionary. Used 0.0 as default for them.")


            features_np = np.array(feature_values).reshape(1, -1) # Luôn reshape thành 2D array

            # Kiểm tra số lượng features cuối cùng
            if features_np.shape[1] != len(EXPECTED_FEATURE_ORDER_FOR_MODELS):
                error_msg = (f"Model {model_name_for_log}: Feature count mismatch. "
                             f"Expected {len(EXPECTED_FEATURE_ORDER_FOR_MODELS)}, got {features_np.shape[1]}. "
                             f"Input dict keys: {list(features_dict.keys())}")
                logger.error(error_msg)
                return None # Trả về None nếu lỗi
            return features_np
        except Exception as e:
            logger.error(f"Model {model_name_for_log}: Error preparing input features: {e}", exc_info=True)
            return None

    # Dịch kết quả dự đoán thành văn bản 
    def _get_top_k_text_labels(self, model: object, probabilities: np.ndarray, k: int = 3) -> List[str]:
        """Lấy top-k nhãn dạng text từ xác suất dự đoán."""
        if not hasattr(model, 'classes_'):
            logger.warning("Model does not have 'classes_' attribute. Cannot determine top-k text labels accurately if classes are not standard integers.")
            # Fallback nếu không có classes_ 
            top_k_indices = np.argsort(probabilities[0])[::-1][:k]
            return [REVERSE_LABEL_MAP_CLASSIFICATION.get(idx, f"RawIdx_{idx}") for idx in top_k_indices]

        class_labels_from_model = model.classes_ 
        
        # Lấy top k indices từ probabilities ( probabilities là 2D array [[p0, p1, ...]])
        # [::-1] để đảo ngược thành giảm dần
        top_k_indices_in_proba_array = np.argsort(probabilities[0])[::-1][:k] 
        
        top_k_text_labels = []
        for index_in_proba in top_k_indices_in_proba_array:
            # index_in_proba là vị trí trong mảng probabilities 
            numeric_label = class_labels_from_model[index_in_proba]
            text_label = REVERSE_LABEL_MAP_CLASSIFICATION.get(numeric_label, f"UnknownNumericLabel({numeric_label})")
            top_k_text_labels.append(text_label)
        
        # Đảm bảo luôn trả về đủ k nhãn, có thể là "Unknown"
        while len(top_k_text_labels) < k:
            top_k_text_labels.append("Unknown (less than k predictions)")
            
        return top_k_text_labels
    # ---- HÀM DỰ ĐOÁN CHÍNH -------
    async def predict(self, model_name: str, features_dict: Union[dict, pd.Series]) -> Dict[str, Union[str, List, None]]:
        # Tải mô hình
        try:
            model = self._load_model_from_file(model_name)
        except FileNotFoundError: # Bắt lỗi cụ thể từ _load_model_from_file
             return {"error": f"Model file for '{model_name}' not found or model directory issue."}
        except Exception as e_load:
            return {"error": f"Failed to load model '{model_name}': {str(e_load)}"}

        if isinstance(features_dict, pd.Series): # Chuyển Series thành dict nếu cần
            features_dict = features_dict.to_dict()
        
        if not isinstance(features_dict, dict):
            return {"error": "Input 'features_dict' must be a dictionary or pandas Series."}

        # Chuẩn bị dữ liệu đầu vào
        prepared_features_np = self._prepare_input_features(features_dict, model_name)
        if prepared_features_np is None: # Lỗi đã được log trong _prepare_input_features
            return {"error": f"Failed to prepare input features for model {model_name}."}

        # Dự đoán xác suất
        probabilities_list = None
        try:
            if hasattr(model, "predict_proba"):
                probabilities_np = model.predict_proba(prepared_features_np) # probabilities_np là 2D array, ví dụ: [[0.1, 0.9, ...]]
                probabilities_list = probabilities_np.tolist() # Chuyển thành list Python
            else: # Xử lý mô hình không có predict_proba 
                logger.warning(f"Model {model_name} does not have predict_proba. Using proxy for probabilities.")
                num_classes = len(getattr(model, 'classes_', [])) # Cố gắng lấy số lớp
                if num_classes == 0 and hasattr(prepared_features_np, 'shape'): # Thử cách khác nếu model.classes_ không có
                  
                    # mặc định số lớp là 2 nếu không rõ
                    logger.warning(f"Cannot determine number of classes for {model_name} predict_proba proxy. Assuming 2 classes if structure allows.")
                    if hasattr(model, "feature_importances_") and len(model.feature_importances_) > 0:
                        mean_importance = np.mean(model.feature_importances_)
                        probabilities_np = np.array([[0.5] * 2]) # Giả định 2 lớp
                        logger.warning("Used very basic probability proxy due to missing model.classes_ and predict_proba.")

                    else:
                        # probabilities_np = np.ones((1, num_classes_guess)) / num_classes_guess if num_classes_guess > 0 else np.array([[0.5, 0.5]])
                        probabilities_np = np.array([[0.5] * 2]) # Giả định 2 lớp
                        logger.warning("Used very basic probability proxy due to missing model.classes_ and predict_proba.")
                    probabilities_list = probabilities_np.tolist()


            if probabilities_list is None or not probabilities_list[0]:
                 logger.error(f"Model {model_name} failed to produce valid probabilities.")
                 return {"error": f"Model {model_name} failed to produce probabilities."}


        except Exception as e_proba:
            logger.error(f"Error getting probabilities from model {model_name}: {e_proba}", exc_info=True)
            return {"error": f"Error getting probabilities from model {model_name}: {str(e_proba)}"}
        
        # Lấy top-k nhãn dạng TEXT
        try:
            top_text_labels = self._get_top_k_text_labels(model, np.array(probabilities_list), k=3)
        except Exception as e_label:
            logger.error(f"Error getting top-k text labels for model {model_name}: {e_label}", exc_info=True)
            return {"error": f"Error getting top-k text labels: {str(e_label)}"}
        # Trả về kết quả
        result = {
            'predicted_label_top_1': top_text_labels[0],
            'predicted_label_top_2': top_text_labels[1],
            'predicted_label_top_3': top_text_labels[2],
            'probabilities': probabilities_list[0] # Trả về list xác suất cho sample đầu tiên (và duy nhất)
        }
        return result

# Hàm classify để gọi từ main_initialize_memory.py
async def classify(classifier_name: str, preprocessed_features_dict: dict) -> Dict[str, Union[str, List, None]]:
    """
    Phân loại đặc trưng đã tiền xử lý bằng ClassificationTool.
    Args:
        classifier_name: Tên mô hình (e.g., 'logistic_regression_ACI').
        preprocessed_features_dict: Dictionary chứa đặc trưng đã tiền xử lý (đã scale, không có 'Label').
    Returns:
        Dictionary chứa kết quả dự đoán (text labels) hoặc lỗi.
    """
    try:
        tool = ClassificationTool() # model_dir sẽ lấy giá trị mặc định
        # logger.debug(f"Classify called for {classifier_name} with features: {list(preprocessed_features_dict.keys())}")
        prediction_result = await tool.predict(classifier_name, preprocessed_features_dict)
        
        if "error" not in prediction_result:
            logger.info(f"Classification result for {classifier_name}: predicted '{prediction_result.get('predicted_label_top_1')}'")
        else:
            logger.error(f"Classification for {classifier_name} returned an error: {prediction_result['error']}")
        return prediction_result
    except Exception as e:
        logger.error(f"Outer exception in classify function for {classifier_name}: {e}", exc_info=True)
        return {"error": f"General error in classify for {classifier_name}: {str(e)}"}