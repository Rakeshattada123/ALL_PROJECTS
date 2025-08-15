# main.py

import os
import json
import io

# Web server framework
from fastapi import FastAPI, File, UploadFile, HTTPException

# OCR and Image processing
from PIL import Image
import pytesseract

# Google Gemini AI
import google.generativeai as genai

# --- FastAPI App Initialization ---

# Initialize the FastAPI app
# This 'app' variable is what Uvicorn will look for.
app = FastAPI(
    title="OCR to Structured Data API",
    description="Upload an image, and this API will perform OCR and use Google Gemini to structure the extracted text into JSON.",
    version="1.0.0"
)


# --- Gemini Configuration and Helper Function ---

# It's best practice to configure the model once when the app starts.
try:
    # Get the API key from Render's environment variables
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)
    # Initialize the model
    MODEL_NAME = 'gemini-1.5-flash-latest'
    model = genai.GenerativeModel(MODEL_NAME)
    print(f"Successfully configured Gemini with model: {MODEL_NAME}")
except Exception as e:
    # If the app can't start due to a config error, it's better to know immediately.
    print(f"FATAL ERROR: Could not configure Gemini. {e}")
    model = None

def create_prompt(text: str) -> str:
    """Creates a detailed prompt for the Gemini model."""
    return f"""
    You are an expert in document understanding. Analyze the following OCR-extracted text and return all relevant structured data as a single JSON object.
    Include any fields you can identify, such as company, address, product, nutrition, invoice details, contact, etc.

    Only return valid JSON. Do not add any text before or after the JSON object. Do not use Markdown backticks like ```json.

    Text:
    ---
    {text}
    ---
    """

# --- API Endpoint Definition ---

@app.post("/process-image/", summary="Process a single image for OCR and structuring")
async def process_image_and_structure(file: UploadFile = File(...)):
    """
    Accepts an image file, performs OCR, sends the text to Gemini for structuring,
    and returns the resulting JSON.
    """
    if not model:
        raise HTTPException(status_code=500, detail="Gemini model is not configured. Check server logs.")

    # --- Part 1: OCR (Logic from pipeline.py) ---
    try:
        # Read the image file content in memory
        image_bytes = await file.read()
        
        # Open the image using Pillow
        image = Image.open(io.BytesIO(image_bytes))
        
        # Use pytesseract to extract text
        extracted_text = pytesseract.image_to_string(image)

        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="OCR could not extract any text from the provided image.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during OCR processing: {str(e)}")


    # --- Part 2: Structuring (Logic from ocr_gemini.py) ---
    try:
        prompt = create_prompt(extracted_text)
        
        # Use the async version of generate_content for better performance in a web server
        response = await model.generate_content_async(prompt)
        
        # Clean the response text to get the raw JSON
        response_text = response.text.strip()
        
        # Parse the JSON response from the model
        structured_data = json.loads(response_text)
        
        # Return the final structured data
        return structured_data

    except json.JSONDecodeError:
        # This happens if Gemini returns something that isn't valid JSON
        raise HTTPException(
            status_code=500, 
            detail={
                "error": "Failed to parse Gemini's response as JSON.",
                "raw_response": response.text
            }
        )
    except Exception as e:
        # Catch any other errors from the Gemini API call
        raise HTTPException(status_code=500, detail=f"An error occurred with the Gemini API: {str(e)}")

# Add a root endpoint for basic health checks
@app.get("/")
def read_root():
    return {"status": "API is running. Use the /docs endpoint to see the documentation."}
    