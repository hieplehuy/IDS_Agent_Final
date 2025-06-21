import google.generativeai as genai
import os
def get_gemini_model():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Not found GEMINI_API_KEY!")
    
    genai.configure(api_key=api_key)
    # Khởi tạo mô hình
    model = genai.GenerativeModel("gemini-2.0-flash")
    return model
