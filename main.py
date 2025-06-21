import os
import json
import asyncio
import logging
import google.generativeai as genai
import numpy as np 
from typing import Any 

# Import module
from core.llm.llm_germini import get_gemini_model
from core.tools.data_extraction import data_extraction
from core.tools.data_preprocessing import data_preprocessing
from core.tools.classification import classify
from core.tools.long_term_memory_retrieval import long_term_memory_retrieval_tool_function
from core.tools.knowledge_retrieval import knowledge_retrieval_tool_function
from core.tools.aggregation import aggregate_results_tool_function

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)

# --- HÀM TIỆN ÍCH ---
def convert_int_to_float_in_dict(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: convert_int_to_float_in_dict(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_int_to_float_in_dict(v) for v in data]
    elif isinstance(data, (int, np.integer)): # Xử lý cả int Python và int NumPy
        return float(data)
    elif isinstance(data, (float, np.floating)): # Giữ nguyên float
        return float(data)
    return data

# --- HÀM XỬ LÝ CHÍNH ---
async def process_traffic_line(line_number_to_process: int, csv_file_path: str):
    model = get_gemini_model()
    if not model:
        logger.error("Failed to get Gemini model.")
        return {"error": "Failed to get Gemini model"}

    try:
        with open(os.path.join(os.path.dirname(__file__), "config/tools.json"), "r") as f:
            tool_function_declarations = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load tools.json: {e}")
        return {"error": "Failed to load tool definitions"}

    general_prompt = f"""
You are IDS-Agent, an intelligent cybersecurity analyst specializing in intrusion detection within IoT networks. Your mission is to analyze a network traffic record (specifically line number {line_number_to_process} from file '{csv_file_path}'), determine if it is benign or belongs to a specific attack category, and provide a detailed analysis for your decision.

You have the following tools at your disposal (names match those in the tool configuration):

1.  `data_extraction`:
    *   Description: Extracts a specific network traffic record from a CSV file based on the file path, an identifier (line number as a string, or flow ID as a string), and the extraction method ('line_number' or 'flow_id').
    *   When to use: At the beginning of analyzing a newly requested record. For this task, you need to extract line number {line_number_to_process} from the file '{csv_file_path}' using 'line_number' as the 'by' method and providing the line number as a string for the 'identifier'.

2.  `data_preprocessing`:
    *   Description: Preprocesses the currently loaded raw traffic data for classification. This tool uses the internally stored raw_data from the previous `data_extraction` step.
    *   When to use: After raw data has been extracted. You do not need to pass any arguments to this tool.

3.  `classify`:
    *   Description: Classifies the currently preprocessed traffic features using a specified machine learning model. This tool uses the internally stored preprocessed_data.
    *   Parameters: `classifier_name` (STRING, enum: ["random_forest_ACI", "svc_ACI", "multi_layer_perceptrons_ACI", "decision_tree_ACI", "k_nearest_neighbors_ACI", "logistic_regression_ACI"]). (<<< UPDATE THIS LIST TO MATCH YOUR tools.json!)
    *   When to use: After data has been preprocessed. Provide only the `classifier_name`. You SHOULD use multiple different classifiers to get a multi-faceted view, one at a time.

4.  `knowledge_retrieval_tool_function`:
    *   Description: Retrieves relevant knowledge about a given query from local and external sources.
    *   Parameters: `query` (STRING).
    *   When to use: When classifiers provide conflicting results, to understand an attack type, or to enrich explanations.

5.  `long_term_memory_retrieval_tool_function`:
    *   Description: Retrieves relevant past successful classification experiences (LTM) based on current classification results. This tool uses the internally stored classification_results.
    *   When to use: After obtaining results from several classifiers, to compare and learn from previous decisions. You do not need to pass any arguments to this tool.

6.  `aggregation`:
    *   Description: Aggregates current classification results, LTM, and external knowledge to produce a final, explained decision. This tool uses internally stored states.
    *   Parameters: `line_number` (NUMBER - the line number being analyzed), `reasoning_trace_summary` (STRING - a brief summary of your reasoning).
    *   When to use: As the final step to draw a conclusion.

Suggested workflow for this specific request (analyzing line {line_number_to_process} from '{csv_file_path}'):
1.  **Extract Data:** Call `data_extraction` with `file_path='{csv_file_path}', identifier='{str(line_number_to_process)}', by='line_number'`.
2.  **Preprocess Data:** Call `data_preprocessing`.
3.  **Initial Classification:** Call `classify` one by one for a few different `classifier_name`s (e.g., 'random_forest_ACI', then 'multi_layer_perceptrons_ACI').
4.  **Evaluate Classification Results & Retrieve Context:**
    a.  After classifications, call `long_term_memory_retrieval_tool_function`.
    b.  If results are conflicting or you need more info, call `knowledge_retrieval_tool_function` with relevant queries.
5.  **Aggregate & Decide:** Call `aggregation` with the `line_number` and a `reasoning_trace_summary`.

Your final output from the `aggregation` tool must be a JSON object including: `line_number`, `analysis`, `predicted_label_top_1`, `predicted_label_top_2`, and `predicted_label_top_3`.
Start by calling `data_extraction`.
"""

    initial_user_message = f"Please analyze line number {line_number_to_process} from the file '{csv_file_path}' following the IDS-Agent protocol, starting with data_extraction."
    
    contents = [
        {"role": "user", "parts": [{"text": general_prompt + "\n\n" + initial_user_message}]}
    ]

    current_raw_data = None
    current_preprocessed_data = None
    current_classification_results = {}
    current_memory_retrieved = None
    current_reasoning_trace = []

    MAX_TURNS = 15
    turn_count = 0

    while turn_count < MAX_TURNS:
        turn_count += 1
        logger.info(f"\n--- Turn {turn_count} ---")
        try:
            logger.debug(f"Sending to Gemini, current contents length: {len(contents)}")
            response = await model.generate_content_async(
                contents,
                tools=tool_function_declarations,
                tool_config={"function_calling_config": {"mode": "AUTO"}}
                # generation_config={"temperature": 0.7} 
            )
        except Exception as e:
            logger.error(f"Gemini API error: {e}", exc_info=True)
            return {"error": f"Failed to call Gemini API: {str(e)}"}

        try:
            candidate = response.candidates[0]
        except (IndexError, AttributeError, TypeError) as e:
            logger.error(f"Invalid response structure or no candidates from Gemini: {response}", exc_info=True)
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                logger.error(f"Prompt Feedback: {response.prompt_feedback}")
            return {"error": "Invalid response structure or no candidates from Gemini"}
        
        # Xử lý phản hồi của model
        model_called_tool_this_turn = False
        model_response_part_for_history = None 

        if candidate.content and candidate.content.parts:
            # Ưu tiên xử lý function_call 
            called_function_part = None
            text_part_content = None

            for part in candidate.content.parts: # Lặp để tìm function call hoặc text
                if part.function_call:
                    called_function_part = part.function_call
                    break # Chỉ lấy function_call đầu tiên
                elif hasattr(part, 'text') and part.text:
                    if text_part_content is None: # Chỉ lấy text đầu tiên nếu không có function_call
                        text_part_content = part.text
            
            if called_function_part:
                model_called_tool_this_turn = True
                tool_name = called_function_part.name
                args_dict = {k: v for k, v in called_function_part.args.items()}
                
                model_response_part_for_history = {"function_call": {"name": tool_name, "args": args_dict}}
                contents.append({"role": "model", "parts": [model_response_part_for_history]})
                
                logger.info(f"LLM called tool: {tool_name} with args: {args_dict}")
                current_reasoning_trace.append(f"Turn {turn_count}: LLM called {tool_name} with args: {json.dumps(args_dict, default=str)}")

                tool_result = None
                # --- THỰC THI TOOL ---
                if tool_name == "data_extraction":
                    try:
                        tool_result = await data_extraction(
                            file_path=args_dict.get("file_path", csv_file_path),
                            identifier=str(args_dict.get("identifier", str(line_number_to_process))), # Đảm bảo là string
                            by=args_dict.get("by", "line_number")
                        )
                        if "error" not in tool_result: current_raw_data = tool_result
                        logger.info(f"Data extraction result: {tool_result}")
                    except Exception as e:
                        logger.error(f"Error during data_extraction: {e}", exc_info=True)
                        tool_result = {"error": f"Error during data_extraction: {str(e)}"}
                
                elif tool_name == "data_preprocessing":
                    if current_raw_data and "error" not in current_raw_data:
                        tool_result = await data_preprocessing(current_raw_data)
                        if "error" not in tool_result: current_preprocessed_data = tool_result
                        logger.info(f"Data preprocessing result: {tool_result}")
                    else:
                        tool_result = {"error": "No valid raw_data for preprocessing."}
                        logger.error(tool_result["error"])
                
                elif tool_name == "classify":
                    if current_preprocessed_data and "error" not in current_preprocessed_data:
                        features_for_classification = {k: v for k, v in current_preprocessed_data.items() if k != 'Label'}
                        classifier_name_arg = args_dict.get("classifier_name")
                        tool_result = await classify(
                            classifier_name=classifier_name_arg,
                            preprocessed_features_dict=features_for_classification
                        )
                        if "error" not in tool_result and classifier_name_arg:
                            current_classification_results[classifier_name_arg] = tool_result
                        logger.info(f"Classification ({classifier_name_arg}) result: {tool_result}")
                    else:
                        tool_result = {"error": "No valid preprocessed_data for classification."}
                        logger.error(tool_result["error"])

                elif tool_name == "knowledge_retrieval_tool_function":
                    tool_result = await knowledge_retrieval_tool_function(query=args_dict.get("query"))
                    logger.info(f"Knowledge retrieval result: {(tool_result.get('retrieved_knowledge') or 'No knowledge')[:200]}...")

                elif tool_name == "long_term_memory_retrieval_tool_function":
                    if current_classification_results:
                        tool_result = await long_term_memory_retrieval_tool_function(
                            classifier_names=list(current_classification_results.keys()),
                            classification_results=current_classification_results
                        )
                        if "error" not in tool_result: current_memory_retrieved = tool_result
                        logger.info(f"LTM retrieval result: {tool_result}")
                    else:
                        tool_result = {"error": "No classification_results for LTM retrieval."}
                        logger.error(tool_result["error"])
                
                elif tool_name == "aggregation":
                    if current_classification_results and current_memory_retrieved is not None:
                        tool_result = await aggregate_results_tool_function(
                            classification_results=current_classification_results,
                            memory=current_memory_retrieved,
                            line_number=int(args_dict.get("line_number", float(line_number_to_process))), # Ép kiểu về int
                            reasoning_trace=args_dict.get("reasoning_trace_summary", current_reasoning_trace) # Sửa lại tên tham số nếu cần
                        )
                        logger.info(f"Aggregation result: {tool_result}")
                        if "error" not in tool_result and "analysis" in tool_result:
                            return tool_result # Kết thúc pipeline
                    else:
                        missing_info = []
                        if not current_classification_results: missing_info.append("classification_results")
                        if current_memory_retrieved is None: missing_info.append("memory")
                        error_msg = f"Missing {', '.join(missing_info)} for aggregation."
                        tool_result = {"error": error_msg}
                        logger.error(error_msg)
                else:
                    tool_result = {"error": f"Unknown tool: {tool_name}"}
                    logger.error(tool_result["error"])

                # Chuẩn bị và thêm function response
                processed_tool_result_for_llm = convert_int_to_float_in_dict(tool_result)
                contents.append({
                    "role": "function",
                    "parts": [{
                        "function_response": {
                            "name": tool_name,
                            "response": {"result": processed_tool_result_for_llm if processed_tool_result_for_llm is not None else {"error": "Tool returned None"}}
                        }
                    }]
                })
                logger.debug(f"Appended function response for {tool_name} to contents.")

            elif text_part_content: # Model trả về text, không gọi tool
                model_response_part_for_history = {"text": text_part_content}
                contents.append({"role": "model", "parts": [model_response_part_for_history]})
                logger.info(f"LLM response (text): {text_part_content[:200]}...")
                
                # Kiểm tra xem text này có phải JSON kết quả cuối không
                try:
                    final_result_text_check = text_part_content
                    if final_result_text_check.strip().startswith("```json"):
                        final_result_text_check = final_result_text_check.strip()[7:-3].strip()
                    elif final_result_text_check.strip().startswith("```"):
                         final_result_text_check = final_result_text_check.strip()[3:-3].strip()

                    final_result = json.loads(final_result_text_check)
                    if "line_number" in final_result and "analysis" in final_result:
                        logger.info(f"Final result received from LLM text: {final_result}")
                        return final_result
                except json.JSONDecodeError:
                    logger.debug("Model text is not final JSON, continuing turn.")
                except Exception as e:
                    logger.error(f"Error processing model text response as JSON: {e}", exc_info=True)
                # Nếu không phải JSON cuối cùng, vòng lặp sẽ tiếp tục

            else: # Không có function call và không có text
                logger.warning("Model response part was empty or unhandled.")
                # contents.append({"role": "model", "parts": [{}]})


        else: # candidate.content hoặc candidate.content.parts không tồn tại
            logger.warning("Model response did not have parsable content or parts. Assuming end or error.")
            return {"final_llm_thought_or_error": "Model returned no parsable content.", "turn_count": turn_count}

        if not model_called_tool_this_turn and not (model_response_part_for_history and model_response_part_for_history.get("text")):
            logger.warning("Model neither called a tool nor provided text. Ending interaction.")
            return {"final_llm_thought_or_error": "Model provided no actionable response.", "turn_count": turn_count}


    logger.warning(f"Max turns ({MAX_TURNS}) reached. Returning current state or error.")
    return {"error": "Max turns reached", "reasoning_trace": current_reasoning_trace, "last_tool_result": tool_result if 'tool_result' in locals() else None}

if __name__ == "__main__":
    csv_file = "D:/UIT 2025-2026/Hocmay/IDS_Agent/dataset/ACI-IoT-2023_test_no_labels.csv"

    if not os.path.exists(csv_file):
        print(f"ERROR: CSV file not found at {os.path.abspath(csv_file)}")
    elif not os.path.exists(os.path.join(os.path.dirname(__file__), "config/tools.json")):
        print(f"ERROR: tools.json not found at {os.path.abspath(os.path.join(os.path.dirname(__file__), 'config/tools.json'))}")
    else:
        user_input = input()
        try:
            target_line = int(user_input)
            if target_line < 1:
                raise ValueError("Số dòng phải >= 1")
        except ValueError:
            print("ERROR: Nhập một số nguyên dương.")
            exit(1)

        print(f"Attempting to process line {target_line} from {csv_file}...")
        try:
            final_answer = asyncio.run(process_traffic_line(target_line, csv_file))
            print("\n--- Final Answer from IDS-Agent Pipeline ---")
            print(json.dumps(final_answer, indent=2, ensure_ascii=False))
        except TypeError:
            print(final_answer)