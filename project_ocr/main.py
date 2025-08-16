# main.py

import os
import json
import io
import re  # --- NEW: Import the regular expression module ---

# Web server framework
from fastapi import FastAPI, File, UploadFile, HTTPException

# OCR and Image processing
from PIL import Image
import pytesseract

# Google Gemini AI
import google.generativeai as genai

# For managing environment variables
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# --- FastAPI App Initialization ---
app = FastAPI(
    title="OCR to Structured Data API",
    description="Upload an image, and this API will perform OCR and use Google Gemini to structure the extracted text into JSON.",
    version="1.0.0"
)


# --- Gemini Configuration and Helper Functions ---

# Configure the model
try:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)
    MODEL_NAME = 'gemini-1.5-flash-latest'
    model = genai.GenerativeModel(MODEL_NAME)
    print(f"Successfully configured Gemini with model: {MODEL_NAME}")
except Exception as e:
    print(f"FATAL ERROR: Could not configure Gemini. {e}")
    model = None

# --- NEW: Helper function to clean the model's response ---
def extract_json_from_response(text: str) -> str:
    """
    Extracts a JSON object from a string that might be wrapped in Markdown fences.
    """
    # Use a regular expression to find the content between ```json and ```
    # The re.DOTALL flag allows the '.' to match newline characters.
    match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    
    # If it finds a match in the markdown format, return the extracted JSON part
    if match:
        return match.group(1)
    
    # If no markdown fences are found, assume the text is the JSON itself
    # and strip any leading/trailing whitespace.
    return text.strip()


def create_prompt(text: str) -> str:
    """Creates a detailed prompt for the Gemini model."""
    # I've slightly improved the prompt to be even more direct.
    return f"""
    You are an expert data extraction API. Your sole purpose is to convert unstructured text into a structured JSON object.

    Analyze the following OCR-extracted text. Return a single, valid JSON object containing all relevant data.
    
    IMPORTANT RULES:
    1. Your entire response MUST be the JSON object itself.
    2. Do NOT use Markdown formatting (like ```json).
    3. Begin your response with `{{` and end it with `}}`.

    Text to analyze:
    ---
    {text}
    ---
    """

# --- API Endpoint Definition ---

@app.post("/process-image/", summary="Process a single image for OCR and structuring")
async def process_image_and_structure(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=500, detail="Gemini model is not configured. Check server logs.")

    # --- Part 1: OCR ---
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        extracted_text = pytesseract.image_to_string(image)
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="OCR could not extract any text from the provided image.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during OCR processing: {str(e)}")

    # --- Part 2: Structuring ---
    try:
        prompt = create_prompt(extracted_text)
        response = await model.generate_content_async(prompt)
        
        # --- MODIFIED: Use the helper function to clean the response ---
        cleaned_json_text = extract_json_from_response(response.text)
        
        # Parse the cleaned JSON response
        structured_data = json.loads(cleaned_json_text)
        
        return structured_data

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500, 
            detail={
                "error": "Failed to parse Gemini's response as JSON, even after cleaning.",
                "raw_response": response.text
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred with the Gemini API: {str(e)}")

@app.get("/")
def read_root():
    return {"status": "API is running. Use the /docs endpoint to see the documentation."}