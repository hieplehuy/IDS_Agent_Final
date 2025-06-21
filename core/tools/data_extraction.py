import pandas as pd
import logging
from typing import Dict, Any # Thay Union[int, str] bằng Any hoặc str cho identifier

logger = logging.getLogger(__name__)

async def data_extraction(file_path: str, identifier: str, by: str = "line_number") -> Dict:
    """
    Extracts a traffic record from a CSV file based on either line number or flow ID.
    'identifier' is expected to be a string from the LLM.
    If 'by' is 'line_number', 'identifier' should be a string representation of an integer.
    If 'by' is 'flow_id', 'identifier' is the flow ID string.

    Args:
        file_path (str): Path to the CSV file.
        identifier (str): The identifier string.
        by (str, optional): Specify "line_number" or "flow_id". Defaults to "line_number".

    Returns:
        Dict: A dictionary containing the traffic record, or an error message.
    """
    logger.info(f"Attempting data extraction from '{file_path}' by '{by}' with identifier string '{identifier}'")

    try:
        df = pd.read_csv(file_path)

        if by == "line_number":
            try:
                # đổi identifier (là string) thành số nguyên nếu là line_number
                line_num_int = int(identifier)
            except ValueError:
                raise ValueError(f"Identifier '{identifier}' is not a valid integer string for line_number extraction.")
            
            # Kiểm tra phạm vi của số dòng
            if line_num_int < 1 or line_num_int > len(df):
                raise ValueError(f"Line number {line_num_int} out of range (1 to {len(df)}) for file {file_path}")
            
            row = df.iloc[line_num_int - 1].to_dict()  # pandas iloc dùng 0-based indexing
        
        elif by == "flow_id":
            if 'Flow ID' not in df.columns:
                raise KeyError("Column 'Flow ID' not found in CSV. Please check the actual column names in your CSV file.")

            # Tìm dòng có cột 'Flow ID' bằng với identifier
            # Đảm bảo kiểu dữ liệu của cột 'Flow ID' trong DataFrame phù hợp để so sánh với identifier (string) tring tools.json
            matching_rows = df[df['Flow ID'].astype(str) == identifier] # Ép kiểu cột 'Flow ID' thành str
            
            if matching_rows.empty:
                raise ValueError(f"Flow ID '{identifier}' not found in file {file_path}")
            
            row = matching_rows.iloc[0].to_dict() # Lấy dòng đầu tiên nếu có nhiều kết quả
        else:
            raise ValueError(f"Invalid value for 'by': {by}. Must be 'line_number' or 'flow_id'")

        logger.info(f"Successfully extracted data: {row}")
        return row

    except FileNotFoundError:
        error_msg = f"File not found: {file_path}"
        logger.error(error_msg)
        return {"error": error_msg}
    except ValueError as ve:
        error_msg = str(ve)
        logger.error(error_msg)
        return {"error": error_msg}
    except KeyError as ke:
        error_msg = f"Column access error: {str(ke)}. This might be due to an incorrect column name."
        logger.error(error_msg)
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"An unexpected error occurred during data extraction: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"error": error_msg}