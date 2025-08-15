import os
import json
import google.generativeai as genai
from tqdm import tqdm

# --- CONFIGURATION ---

# The script will first try to get the API key from an environment variable.
# If you don't set the environment variable, uncomment and replace the line below.
# WARNING: Do not commit your API key to public repositories!
# os.environ['GOOGLE_API_KEY'] = "YOUR_API_KEY_HERE"

# The input file from your OCR pipeline
INPUT_JSON_FILE = 'output_list.json'

# The final, structured output file
OUTPUT_JSON_FILE = 'structured_output.json'

# Configure the Gemini model
# 'gemini-1.5-flash-latest' is fast and cost-effective for this task.
# For higher accuracy on complex documents, consider 'gemini-1.5-pro-latest'.
MODEL_NAME = 'gemini-1.5-flash-latest'

# --- SCRIPT LOGIC ---

def create_prompt(text):
    """Creates a detailed prompt for the Gemini model."""
    
    # This prompt is key to getting good results.
    # It instructs the model on what to do and how to format the output.
    prompt = f"""
    You are an expert in document understanding. Analyze the following OCR-extracted text and return all relevant structured data as JSON.
Include any fields you can identify, such as company, address, product, nutrition, invoice details, contact, etc.

Only return valid JSON. Do not explain.

Text:
    ---
    {text}
    ---
    """
    return prompt

def process_with_gemini():
    """
    Reads OCR output, sends it to Gemini for structuring, and saves the result.
    """
    # 1. Configure the API Key
    try:
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    except KeyError:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        print("Please set the environment variable or add the key directly to the script.")
        return

    # 2. Load the intermediate OCR data
    try:
        with open(INPUT_JSON_FILE, 'r', encoding='utf-8') as f:
            ocr_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{INPUT_JSON_FILE}'")
        print("Please run the OCR pipeline first to generate this file.")
        return

    # 3. Initialize the Gemini model
    model = genai.GenerativeModel(MODEL_NAME)
    print(f"Successfully initialized Gemini model: {MODEL_NAME}")

    # 4. Process each text entry
    structured_results = []
    print(f"Processing {len(ocr_data)} text entries with Gemini...")

    for item in tqdm(ocr_data, desc="Structuring with Gemini"):
        filename = item.get("filename")
        text_to_process = item.get("extracted_text", "")
        
        final_obj = {"source_filename": filename}

        if not text_to_process or "error" in item:
            final_obj["structured_data"] = {"error": item.get("error", "No text extracted during OCR.")}
            structured_results.append(final_obj)
            continue

        # Create the prompt and call the API
        try:
            prompt = create_prompt(text_to_process)
            response = model.generate_content(prompt)
            
            # Clean up the response to ensure it's valid JSON
            cleaned_response_text = response.text.strip().replace("```json", "").replace("```", "")
            
            # Parse the JSON response from the model
            structured_data = json.loads(cleaned_response_text)
            final_obj["structured_data"] = structured_data

        except json.JSONDecodeError:
            print(f"\nWarning: Gemini did not return valid JSON for {filename}. Storing raw text.")
            final_obj["structured_data"] = {"error": "Failed to parse LLM response", "raw_response": response.text}
        except Exception as e:
            print(f"\nAn error occurred while processing {filename}: {e}")
            final_obj["structured_data"] = {"error": str(e)}

        structured_results.append(final_obj)

    # 5. Save the final structured data
    with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(structured_results, f, ensure_ascii=False, indent=4)

    print(f"\nProcessing complete! Structured data saved to '{OUTPUT_JSON_FILE}'")


if __name__ == "__main__":
    process_with_gemini()