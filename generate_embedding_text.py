import os
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai.types import EmbedContentConfig



load_dotenv()
GOOGLE_API_KEY = os.getenv('API_KEY')

MODEL_NAME = "gemini-embedding-001" 

# --- File Paths ---
MASTER_DATA_FILE = "./data/feed_master_data.json"

# Output file for the generated embeddings
OUTPUT_EMBEDDINGS_FILE = "./data/embedded_data.json"




def generate_and_save_embeddings():

    if not GOOGLE_API_KEY:
        raise ValueError("FATAL ERROR: GOOGLE_API_KEY is not set. Please set the environment variable or paste the key directly in the script.")
    client = genai.Client(api_key=GOOGLE_API_KEY)

    try:
        with open(MASTER_DATA_FILE, 'r', encoding='utf-8') as f:
            report_definitions = json.load(f)
    except FileNotFoundError:
        print(f"FATAL ERROR: Master data file not found at '{MASTER_DATA_FILE}'. Please create it first.")
        return

    final_embeddings_list = []
    total_records = len(report_definitions)
    print(f"Found {total_records} records. Starting embedding generation with model '{MODEL_NAME}'...")

    for i, definition in enumerate(report_definitions):
        report_name = definition.get("ReportName", "UnknownReportName")
        description = definition.get("Description", "")
        keywords = definition.get("Keywords", [])
        sample_queries = definition.get("SampleQueries", [])

        keywords_text = ", ".join(keywords)
        sample_queries_text = ", ".join(sample_queries)

        combined_text_for_embedding = (
            f"Report Name: {report_name}. "
            f"Description: {description}. "
            f"Keywords: {keywords_text}. "
            f"Typical User Questions: {sample_queries_text}"
        )
        
        if not description:
            print(f"  - WARNING: Skipping '{report_name}' due to empty description.")
            continue
        
        print(f"  - ({i+1}/{total_records}) Processing: {report_name}")

        try:
            # The API call is much simpler in this library!
            response = client.models.embed_content(
                model=MODEL_NAME,
                contents=combined_text_for_embedding,
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
                "Description": description,
                "Vector": vector
            })
            
            # A small delay to be respectful to the API on large jobs
            time.sleep(0.5)

        except Exception as e:
            print(f"  - ERROR: Failed to generate embedding for '{report_name}'. Error: {e}")

    
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