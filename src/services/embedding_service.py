
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import  app_config 
from agents import embedding_agent 
from agents import db_agent

def load_master_data():
    try:
        with open(app_config.MASTER_DATA_FILE, 'r', encoding='utf-8') as f:
            report_definitions = json.load(f)
            return report_definitions
    except FileNotFoundError:
        print(f"FATAL ERROR: Master data file not found at '{app_config.MASTER_DATA_FILE}'. Please create it first.")
        return []
    
async def generate_embedded_data() -> list[dict] | None:
    final_embeddings_list = []
    try:
        feed_data = load_master_data()
        total_records = len(feed_data)
        for i, definition in enumerate(feed_data):
            report_id = definition.get("Id", 0)
            version_no = definition.get("Version", 1.0)
            report_name = definition.get("ReportName", "UnknownReportName")
            description = definition.get("Description", "")
            keywords = definition.get("Keywords", [])
            sample_queries = definition.get("SampleQueries", [])

            keywords_text = ", ".join(keywords)
            sample_queries_text = ", ".join(sample_queries)

            combined_text_for_embedding = (
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
                embedder = embedding_agent.get_embedding_model()
                vector = await embedder.aembed_query(combined_text_for_embedding)

                final_embeddings_list.append({
                    "id": report_id,
                    "reportname": report_name,
                    "description": description,
                    "keywords": keywords,
                    "samplequeries": sample_queries,
                    "modelname": "Gemini",
                    "verno": version_no,
                    "embeddedvector": vector
                })

                print(f"  final embedded list length: {len(final_embeddings_list)}.")

            except Exception as e:
                print(f"  - ERROR: Failed to generate embedding for '{report_name}'. Error: {e}")

    except Exception as e:
        print(f"FATAL ERROR: Failed to generate embedded data. Error: {e}")
    return final_embeddings_list

async def save_embeddings_to_db(embeddings: list[dict]) -> bool:
    try:
        db = db_agent.db_agent() 
        db.insert_document(embeddings)
        return True
    except Exception as e:
        print(f"ERROR: Failed to save embeddings to database. Error: {e}")
        return False
    
async def execute():
    embeddings = await generate_embedded_data()
    if embeddings:
        await save_embeddings_to_db(embeddings)
        return len(embeddings)
    else:
        return 0

if __name__ == "__main__":
    execute()