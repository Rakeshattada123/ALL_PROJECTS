import os
import json
from PIL import Image
import pytesseract
from tqdm import tqdm

# --- CONFIGURATION ---

# IMPORTANT: If Tesseract is not in your system's PATH, specify its location.
# Example for Windows:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Define the path to the directory containing your images
INPUT_DIR = 'input_images'

# Define the path for the output JSON file
OUTPUT_FILE = 'output_list.json'

# Define allowed image extensions
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'}

# --- SCRIPT LOGIC ---

def ocr_pipeline_list():
    """
    Processes all images in the input directory and saves the results as a
    list of JSON objects in a single file.
    """
    # Check if the input directory exists
    if not os.path.isdir(INPUT_DIR):
        print(f"Error: Input directory not found at '{INPUT_DIR}'")
        return

    # Get a list of all image files in the directory
    image_files = [f for f in os.listdir(INPUT_DIR) if os.path.splitext(f)[1].lower() in ALLOWED_EXTENSIONS]

    if not image_files:
        print(f"No images found in '{INPUT_DIR}'")
        return

    # Initialize a list to store OCR result objects
    # The structure will be: [ {"filename": "...", "extracted_text": "..."}, ... ]
    ocr_results_list = []

    print(f"Found {len(image_files)} images to process. Starting OCR...")

    # Process each image with a progress bar
    for filename in tqdm(image_files, desc="Processing Images"):
        image_path = os.path.join(INPUT_DIR, filename)
        
        # Create a dictionary for the current image's result
        result_obj = {"filename": filename}

        try:
            # Open the image file using Pillow
            with Image.open(image_path) as img:
                # Use pytesseract to extract text
                text = pytesseract.image_to_string(img)
                
                # Add the extracted text to our result object
                result_obj["extracted_text"] = text.strip()

        except Exception as e:
            # If an error occurs, add an error message to the result object
            print(f"\nCould not process {filename}. Error: {e}")
            result_obj["error"] = f"Error processing file: {e}"
        
        # Append the result object for the current image to our master list
        ocr_results_list.append(result_obj)

    # Save the list of result objects to a single JSON file
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            # Use indent=4 for a pretty, readable JSON format
            json.dump(ocr_results_list, f, ensure_ascii=False, indent=4)
        
        print(f"\nSuccessfully processed {len(ocr_results_list)} images.")
        print(f"Results saved to '{OUTPUT_FILE}'")
    except Exception as e:
        print(f"\nError saving results to JSON file: {e}")

# --- EXECUTION ---

if __name__ == "__main__":
    ocr_pipeline_list()