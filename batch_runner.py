# CHẠY TỰ ĐỘNG QUA LOẠT DATASET NO LABEL ĐỂ GHI KẾT QUẢ DỰ ĐOÁN 
import asyncio
import json
import os
import pandas as pd
import logging
from main import process_traffic_line

# Cấu hình logging cho batch_runner
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
batch_logger = logging.getLogger("batch_runner")


async def run_batch_processing(start_line: int, end_line: int, input_csv_path: str, output_csv_path: str):
    """
    Chạy pipeline IDS-Agent cho một loạt các dòng và lưu kết quả dự đoán.
    """
    results_list = [] 
    
    batch_logger.info(f"Starting batch processing for lines {start_line} to {end_line} from '{input_csv_path}'.")
    batch_logger.info(f"Results will be saved to '{output_csv_path}'.")

 
    try:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    except FileExistsError:
        pass
    except Exception as e:
        batch_logger.error(f"Could not create output directory {os.path.dirname(output_csv_path)}: {e}")
        return

    if os.path.exists(output_csv_path) and start_line == default_start_line: 
        try:
            os.remove(output_csv_path)
            batch_logger.info(f"Removed existing output file: {output_csv_path} for a fresh start.")
        except Exception as e:
            batch_logger.warning(f"Could not remove existing output file {output_csv_path}: {e}. Will attempt to append/overwrite.")


    for current_line_number in range(start_line, end_line + 1):
        batch_logger.info(f"--- Processing line: {current_line_number} ---")
        
        pipeline_output = None
        try:
            pipeline_output = await process_traffic_line(current_line_number, input_csv_path)
            
            predicted_label = "Error_Or_Unknown_Format"
            
            if isinstance(pipeline_output, dict):
                if "error" in pipeline_output:
                    error_message = str(pipeline_output['error'])
                    predicted_label = f"Error: {error_message[:150]}"
                    batch_logger.error(f"Line {current_line_number}: Processing error - {error_message}")
                elif "predicted_label_top_1" in pipeline_output:
                    predicted_label = pipeline_output["predicted_label_top_1"]
                    batch_logger.info(f"Line {current_line_number}: Successfully processed. Prediction: {predicted_label}")
                elif "final_llm_thought_or_error" in pipeline_output:
                    thought_or_error = str(pipeline_output['final_llm_thought_or_error'])
                    predicted_label = f"LLM_End_Thought: {thought_or_error[:150]}"
                    batch_logger.info(f"Line {current_line_number}: LLM ended with thought/error - {thought_or_error[:150]}...")
                else:
                    predicted_label = f"Unknown_Output_Structure: {str(pipeline_output)[:150]}"
                    batch_logger.warning(f"Line {current_line_number}: Unknown output structure - {str(pipeline_output)[:200]}...")
            else:
                predicted_label = f"Non_Dict_Output: {str(pipeline_output)[:150]}"
                batch_logger.warning(f"Line {current_line_number}: Non-dictionary output from pipeline - {str(pipeline_output)[:200]}...")

            results_list.append({"line_number_processed": current_line_number, "predicted_label_top_1": predicted_label})
            
        except Exception as e:
            batch_logger.error(f"Line {current_line_number}: CRITICAL error during call to process_traffic_line - {str(e)}", exc_info=True)
            results_list.append({"line_number_processed": current_line_number, "predicted_label_top_1": f"CRITICAL_Error: {str(e)[:150]}"})
            
        batch_logger.info(f"--- Finished attempt for line {current_line_number} ---")
        
        # Delay
        delay_seconds = 15
        batch_logger.info(f"Waiting for {delay_seconds} seconds before processing next line...")
        await asyncio.sleep(delay_seconds) 

    if results_list:
        df_results = pd.DataFrame(results_list)
        try:
            file_exists = os.path.exists(output_csv_path)
            df_results.to_csv(output_csv_path, mode='a', header=not file_exists, index=False, encoding='utf-8-sig')
            batch_logger.info(f"Successfully appended/saved results to {output_csv_path}")
        except Exception as e:
            batch_logger.error(f"Failed to save final results to CSV: {e}")
    else:
        batch_logger.info("No new results to save at the end of this batch run.")

    batch_logger.info(f"--- Batch processing complete for lines {start_line} to {end_line}. Results saved to {output_csv_path} ---")

if __name__ == "__main__":
    default_start_line = 1
    default_end_line = 205 
    default_input_csv = "D:/UIT 2025-2026/Hocmay/IDS_Agent/dataset/ACI-IoT-2023_test_no_labels.csv"
    default_output_csv = "D:/UIT 2025-2026/Hocmay/IDS_Agent/predictions/batch_predicted_labels.csv"

    print("--- IDS Agent Batch Processing Runner ---")
    try:
        start_line_input = input(f"Enter start line number (default: {default_start_line}): ")
        start_line = int(start_line_input) if start_line_input.isdigit() else default_start_line

        end_line_input = input(f"Enter end line number (default: {default_end_line}): ")
        end_line = int(end_line_input) if end_line_input.isdigit() else default_end_line

        input_csv_path_input = input(f"Enter input CSV file path (default: {default_input_csv}): ")
        input_csv_path = input_csv_path_input if input_csv_path_input else default_input_csv

        output_csv_path_input = input(f"Enter output CSV file path (default: {default_output_csv}): ")
        output_csv_path = output_csv_path_input if output_csv_path_input else default_output_csv

        if start_line > end_line:
            print("Error: Start line cannot be greater than end line.")
        elif not os.path.exists(input_csv_path):
            batch_logger.critical(f"CRITICAL ERROR: Input CSV file not found at {os.path.abspath(input_csv_path)}")
        else:
            batch_logger.info(f"Configuration: Start={start_line}, End={end_line}, Input='{input_csv_path}', Output='{output_csv_path}'")
            confirmation = input("Proceed with this configuration? (yes/no): ").lower()
            if confirmation == 'yes':
                asyncio.run(run_batch_processing(start_line, end_line, input_csv_path, output_csv_path))
            else:
                print("Batch processing cancelled by user.")

    except ValueError:
        print("Invalid input for line numbers. Please enter integers.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")