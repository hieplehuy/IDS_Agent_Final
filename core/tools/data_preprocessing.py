import pandas as pd
import numpy as np
import pickle
import os
import logging

logger = logging.getLogger(__name__)
# Ánh xạ sang số theo như trong quá trình tiền xử lý dataset
label_map = {
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
    "Vulnerability Scan": 11
}
connection_type_map = {
    "wired": 0,
    "wireless": 1
}
# Chuyển dữ liệu thô thành DataFrame và loại bỏ các cột không cần thiết

async def data_preprocessing(raw_data: dict):
    try:
        df = pd.DataFrame([raw_data])
        
        cols_to_drop = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp']
        cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        df.drop(columns=cols_to_drop, inplace=True)
        logger.info(f"Dropped irrelevant columns: {cols_to_drop}")
        # Chuyển đổi cột Label và Connection Type sang dạng số
        if 'Label' in df.columns:
            df['Label'] = df['Label'].map(label_map)
            if df['Label'].isnull().any():
                raise ValueError("Found unmapped Label values")
            logger.info("Mapped 'Label' column")

        if 'Connection Type' in df.columns:
            df['Connection Type'] = df['Connection Type'].map(connection_type_map)
            if df['Connection Type'].isnull().any():
                raise ValueError("Found unmapped Connection Type values")
            logger.info("Mapped 'Connection Type' column")
        # Lựa chọn các đặc trưng quan trọng kết quả khi dùng Ftest
        selected_features = [
                 'Src Port', 'Dst Port', 'Protocol', 'Flow IAT Mean', 'Flow IAT Max',
                 'Flow IAT Min', 'Fwd IAT Mean', 'Fwd IAT Max', 'Fwd IAT Min',
                 'SYN Flag Count', 'RST Flag Count', 'Down/Up Ratio',
                 'Subflow Fwd Packets', 'FWD Init Win Bytes', 'Fwd Seg Size Min',
                 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Connection Type'
        ]

        selected_features = [f for f in selected_features if f in df.columns]
        if not selected_features:
            raise ValueError("No selected features available in data")

        X_selected = df[selected_features]
        # Xử lý các giá trị vô cực và thiếu (NaN)
        X_selected = X_selected.replace([np.inf, -np.inf], np.nan)

        for col in X_selected.columns:
            if col == 'Flow Bytes/s':
                X_selected[col] = X_selected[col].fillna(420451.2619465897)
            else:
                X_selected[col] = X_selected[col].fillna(X_selected[col].mean())
        # Chuẩn hóa dữ liệu bằng scaler đã được huấn luyện
        scaler_path = "models/ACI_IoT_23/scaler.pkl"
        if not os.path.exists(scaler_path):
            raise FileNotFoundError("Scaler file not found")

        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        X_scaled = scaler.transform(X_selected)

        # Chuyển đổi dữ liệu đã xử lý về dạng dictionary và trả về
        processed = pd.DataFrame(X_scaled, columns=selected_features).iloc[0].to_dict()

        if 'Label' in df.columns:
            processed['Label'] = df['Label'].iloc[0]

        logger.info(f"Preprocessed data: {processed}")
        return processed

    except Exception as e:
        logger.error(f"Data preprocessing failed: {str(e)}")
        return {"error": str(e)}
