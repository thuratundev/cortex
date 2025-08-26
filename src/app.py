import os
import json
import chromadb
import uuid
from app_config import MASTER_DATA_FILE
from app_config import OUTPUT_EMBEDDINGS_FILE
import chroma_agent

_chroma_agent = chroma_agent.chroma_agent(db_path="./db", collection_name="reports")

def testChroma():

    # client = chromadb.PersistentClient(path="./db")
    # collection = client.get_or_create_collection(name="reports")


    _documentid = _chroma_agent.add_document("North America is a continent[b] in the Northern and Western hemispheres.[c] North America is bordered to the north by the Arctic Ocean, to the east by the Atlantic Ocean, to the southeast by South America and the Caribbean Sea, and to the south and west by the Pacific Ocean. The region includes Middle America (comprising the Caribbean, Central America, and Mexico) and Northern America.", {"source": "continent"})
    print(f"Added document with ID: {_documentid}")
    # collection.add(
    #     ids=[str(uuid.uuid4())],
    #     documents=["Candy is Sweet"],
    #     metadatas=[{"source": "test"}]
    # )

def testQuery():

    results = _chroma_agent.query("Which continent is Canada in?", n_results=1)
    # Access the embeddings from the results
    if 'embeddings' in results and results['embeddings']:
        # retrieved_embeddings = results['embeddings']
        # print("Retrieved Embeddings:", retrieved_embeddings)
        for i, document in enumerate(results['documents']):
            print(f"Retrieved Document {i+1}: {document}")
    else:
        print("No embeddings found in the query results.")

def startup():
    try:
        try:
            with open(MASTER_DATA_FILE, 'r', encoding='utf-8') as f:
                report_definitions = json.load(f)
        except FileNotFoundError:
            print(f"FATAL ERROR: Master data file not found at '{MASTER_DATA_FILE}'. Please create it first.")
            return

        
        total_records = len(report_definitions)
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
            break  ## test code for only one report
                
        
    except Exception as e:
        print("Error loading master data file:", e)


if __name__ == "__main__":
    testQuery()