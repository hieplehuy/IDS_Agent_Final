# ---- View ChromaDB ----------
import asyncio  
import chromadb
import logging
import json
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from typing import List, Dict, Optional

# Cấu hình logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[logging.StreamHandler()]
)

LTM_DB_PATH = "D:/UIT 2025-2026/Hocmay/Project/IDS_Agent/ChromaDB/"

try:
    from core.tools.long_term_memory_retrieval import LTMRetriever
except ImportError as e:
    logger.error(f"Lỗi nhập LTMRetriever: {e}. Đảm bảo file core/tools/long_term_memory_retrieval.py tồn tại.")
    exit(1)

def retrieve_all_ltm_entries() -> List[Dict]:
    """
    Truy xuất tất cả mục LTM từ ChromaDB.

    Returns:
        Danh sách các dictionary chứa cấu trúc ϕ = {t, x, R, A, O, ŷ}.
    """
    try:
        ltm_retriever = LTMRetriever()
        if not ltm_retriever.collection:
            logger.error("LTMRetriever không khởi tạo được collection.")
            return []

        entries = ltm_retriever.collection.get(include=["metadatas", "documents"])
        ltm_entries = []

        for i, metadata in enumerate(entries["metadatas"]):
            try:
                entry = {
                    "t": metadata["timestamp"],
                    "x": json.loads(metadata["features"]),
                    "R": json.loads(metadata["reasoning_trace"]),
                    "A": json.loads(metadata["actions"]),
                    "ŷ": metadata["final_label"],
                    "O": json.loads(entries["documents"][i])
                }
                ltm_entries.append(entry)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Lỗi phân tích mục LTM ID {entries['ids'][i]}: {e}")
                continue

        logger.info(f"Truy xuất {len(ltm_entries)} mục LTM từ collection 'ltm_iot'.")
        return ltm_entries
    except Exception as e:
        logger.error(f"Truy xuất LTM thất bại: {e}")
        return []

def retrieve_ltm_by_filter(
    timestamp_start: Optional[str] = None,
    timestamp_end: Optional[str] = None,
    label: Optional[str] = None
) -> List[Dict]:
    """
    Truy xuất mục LTM theo bộ lọc thời gian hoặc nhãn.

    Args:
        timestamp_start: Thời gian bắt đầu (ISO format, ví dụ: "2025-05-30T00:00:00").
        timestamp_end: Thời gian kết thúc (ISO format).
        label: Nhãn cần lọc (ví dụ: "Benign", "DoS").

    Returns:
        Danh sách các mục LTM phù hợp.
    """
    try:
        ltm_retriever = LTMRetriever()
        if not ltm_retriever.collection:
            logger.error("LTMRetriever không khởi tạo được collection.")
            return []

        entries = ltm_retriever.collection.get(include=["metadatas", "documents"])
        filtered_entries = []

        for i, metadata in enumerate(entries["metadatas"]):
            try:
                timestamp = metadata["timestamp"]
                final_label = metadata["final_label"]

                # Lọc theo thời gian
                if timestamp_start:
                    start = datetime.fromisoformat(timestamp_start.replace("Z", "+00:00"))
                    if datetime.fromisoformat(timestamp.replace("Z", "+00:00")) < start:
                        continue
                if timestamp_end:
                    end = datetime.fromisoformat(timestamp_end.replace("Z", "+00:00"))
                    if datetime.fromisoformat(timestamp.replace("Z", "+00:00")) > end:
                        continue

                # Lọc theo nhãn
                if label and final_label != label:
                    continue

                # Phân tích metadata và document
                entry = {
                    "t": timestamp,
                    "x": json.loads(metadata["features"]),
                    "R": json.loads(metadata["reasoning_trace"]),
                    "A": json.loads(metadata["actions"]),
                    "ŷ": final_label,
                    "O": json.loads(entries["documents"][i])
                }
                filtered_entries.append(entry)
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Lỗi phân tích mục LTM ID {entries['ids'][i]}: {e}")
                continue

        logger.info(f"Truy xuất {len(filtered_entries)} mục LTM với bộ lọc: timestamp_start={timestamp_start}, timestamp_end={timestamp_end}, label={label}.")
        return filtered_entries
    except Exception as e:
        logger.error(f"Truy xuất LTM theo bộ lọc thất bại: {e}")
        return []

def display_ltm_entries(entries: List[Dict], limit: int = 5):
    """
    Hiển thị các mục LTM một cách dễ đọc.

    Args:
        entries: Danh sách các mục LTM.
        limit: Số lượng mục tối đa để hiển thị.
    """
    if not entries:
        logger.info("Không có mục LTM nào để hiển thị.")
        return

    for i, entry in enumerate(entries[:limit]):
        print(f"\n=== Mục LTM {i+1} ===")
        print(f"Thời gian (t): {entry['t']}")
        print(f"Vector đặc trưng (x): {entry['x'][:5]}... (độ dài: {len(entry['x'])})")
        print(f"Lịch sử suy luận (R): {entry['R']}")
        print(f"Hành động (A): {entry['A']}")
        print(f"Nhãn dự đoán (ŷ): {entry['ŷ']}")
        print(f"Quan sát (O):")
        for j, obs in enumerate(entry['O']):
            print(f"  Quan sát {j+1}: {json.dumps(obs, indent=2)[:200]}...")
    logger.info(f"Hiển thị {min(limit, len(entries))} trong tổng số {len(entries)} mục LTM.")

async def main():
    """
    Hàm chính để test truy xuất LTM.
    """
    logger.info("Bắt đầu truy xuất dữ liệu LTM...")

    # Truy xuất tất cả mục
    logger.info("\n--- Truy xuất tất cả mục LTM ---")
    all_entries = retrieve_all_ltm_entries()
    display_ltm_entries(all_entries)

    # Truy xuất theo bộ lọc (ví dụ: trong 24 giờ qua, nhãn "Benign")
    logger.info("\n--- Truy xuất mục LTM với bộ lọc ---")
    timestamp_end = datetime.now().isoformat()
    timestamp_start = (datetime.now() - timedelta(hours=24)).isoformat()
    filtered_entries = retrieve_ltm_by_filter(
        timestamp_start=timestamp_start,
        timestamp_end=timestamp_end,
        label="Benign"
    )
    display_ltm_entries(filtered_entries)

if __name__ == "__main__":
    asyncio.run(main())