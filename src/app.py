import os
import json
from app_config import MASTER_DATA_FILE
from app_config import OUTPUT_EMBEDDINGS_FILE
import agents.embedding_agent as embedding_agent
import agents.db_agent as db_agent


def startup():
    try:
        with open(MASTER_DATA_FILE, 'r', encoding='utf-8') as f:
            report_definitions = json.load(f)
    except FileNotFoundError:
        print(f"FATAL ERROR: Master data file not found at '{MASTER_DATA_FILE}'. Please create it first.")
        return

    final_embeddings_list = []
    total_records = len(report_definitions)
    for i, definition in enumerate(report_definitions):
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
            vector = embedder.embed_query(combined_text_for_embedding)

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

    if final_embeddings_list:
        print(f"Saving {len(final_embeddings_list)} embeddings to database...")
        db = db_agent.db_agent()
        db.insert_document(final_embeddings_list)
    else:
        print("No valid embeddings found.")

if __name__ == "__main__":
    startup()