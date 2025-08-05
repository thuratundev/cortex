import os
import json
import time
from google import genai
from google.genai.types import EmbedContentConfig


# GOOGLE_API_KEY = 'AIzaSyDiidatii6YDYpp-gR-aRdhx7u79x0MK3I'
GOOGLE_API_KEY = os.getenv('API_KEY')

# Model name is simpler in this library. This is the equivalent of gemini-embedding-001
MODEL_NAME = "gemini-embedding-001" 

# --- File Paths ---
MASTER_DATA_FILE = "./data/feed_master_data.json"
# We name the output file based on the library used for clarity
OUTPUT_EMBEDDINGS_FILE = "./data/embedded_data.json"




def generate_and_save_embeddings():
    """
    Reads procedure descriptions, generates embeddings using the Google AI
    (genai) SDK, and saves them to a new JSON file.
    """
    # 1. Configure the API client with your key
    print("Configuring the Google AI client...")
    if not GOOGLE_API_KEY:
        raise ValueError("FATAL ERROR: GOOGLE_API_KEY is not set. Please set the environment variable or paste the key directly in the script.")
    client = genai.Client(api_key=GOOGLE_API_KEY)

    
    # 2. Read the master data file
    print(f"Reading master data from: {MASTER_DATA_FILE}")
    try:
        with open(MASTER_DATA_FILE, 'r', encoding='utf-8') as f:
            report_definitions = json.load(f)
    except FileNotFoundError:
        print(f"FATAL ERROR: Master data file not found at '{MASTER_DATA_FILE}'. Please create it first.")
        return

    # 3. Generate embeddings for each procedure
    final_embeddings_list = []
    total_records = len(report_definitions)
    print(f"Found {total_records} records. Starting embedding generation with model '{MODEL_NAME}'...")

    for i, definition in enumerate(report_definitions):
        report_name = definition.get("ReportName", "UnknownReportName")
        description = definition.get("Description", "")
        
        if not description:
            print(f"  - WARNING: Skipping '{report_name}' due to empty description.")
            continue
        
        print(f"  - ({i+1}/{total_records}) Processing: {report_name}")

        try:
            # The API call is much simpler in this library!
            response = client.models.embed_content(
                model=MODEL_NAME,
                contents=description,
                config=EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",  # Optional
                    output_dimensionality=3072,  # Optional
                    title=report_name,  # Optional
                ),
            )
            
            # Extract the vector from the response
            # vector = response['embedding']
            vector = response.embeddings[0].values  # Adjusted for the genai library
            
            final_embeddings_list.append({
                "ReportName": report_name,
                "Vector": vector
            })
            
            # A small delay to be respectful to the API on large jobs
            time.sleep(0.5)

        except Exception as e:
            print(f"  - ERROR: Failed to generate embedding for '{report_name}'. Error: {e}")

    # 4. Save the final list of embeddings to the output file
    if final_embeddings_list:
        print(f"\nSaving {len(final_embeddings_list)} generated embeddings to {OUTPUT_EMBEDDINGS_FILE}...")
        try:
            with open(OUTPUT_EMBEDDINGS_FILE, 'w', encoding='utf-8') as f:
                json.dump(final_embeddings_list, f, indent=4, ensure_ascii=False)
            print("Embedding generation and saving complete!")
        except Exception as e:
            print(f"FATAL ERROR: Could not write to output file. Error: {e}")
    else:
        print("No embeddings were generated. Output file not created.")


if __name__ == "__main__":
    generate_and_save_embeddings()