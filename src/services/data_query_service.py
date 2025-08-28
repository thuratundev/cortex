import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import  app_config 
from agents import embedding_agent 
from agents import db_agent

def get_similar(user_query: str) -> list[dict] | None:
    try:
        embedder = embedding_agent.get_embedding_model()
        vector = embedder.embed_query(user_query)
        
        db = db_agent.db_agent()
        similar_reports = db.get_similar_documents(vector)
        return similar_reports
    except Exception as e:
        print(f"ERROR: Failed to retrieve similar reports. Error: {e}")
        return None

if __name__ == "__main__":
    user_query = "Invoice အလိုက် ရောင်းအားစာရင်း ထုတ်ပေးပါ။"
    similar_reports = get_similar(user_query)
    for report in similar_reports:
        print(report['reportname'], report['similarity'])