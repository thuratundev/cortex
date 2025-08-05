import os
import json
import numpy as np
from google import genai
from google.genai.types import EmbedContentConfig

GOOGLE_API_KEY = 'AIzaSyDiidatii6YDYpp-gR-aRdhx7u79x0MK3I'


MODEL_NAME = "gemini-embedding-001" 

# --- File Paths ---
EMBEDDED_FILE_NAME = "./data/embedded_data.json"

def get_query_embedding(query_text: str, model) -> list[float] | None:
    """အသုံးပြုသူရဲ့ မေးခွန်းအတွက် embedding ဖန်တီးပေးသည်"""

    print("Configuring the Google AI client...")
    if not GOOGLE_API_KEY:
        raise ValueError("FATAL ERROR: GOOGLE_API_KEY is not set. Please set the environment variable or paste the key directly in the script.")
    client = genai.Client(api_key=GOOGLE_API_KEY)

    try:
           
        response = client.models.embed_content(
                model=MODEL_NAME,
                contents=query_text,
                config=EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",  # Optional
                ),
            )        

        return response.embeddings[0].values
    except Exception as e:
        print(f"မေးခွန်းကို embedding ဖန်တီးရာတွင် အမှားအယွင်းဖြစ်ပေါ်ပါသည်: {e}")
        return None
    
def cosine_similarity(vec1, vec2):
    """vector နှစ်ခုကြားက cosine similarity ကို တွက်ချက်ပေးသည်"""
    
    dot_product = np.dot(vec1, vec2)
    # np.linalg.norm က vector တစ်ခုရဲ့ အလျား (magnitude) ကို ရှာပေးတာပါ
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
        
    return dot_product / (norm_vec1 * norm_vec2)    

def find_most_similar_procedures(user_query: str, top_k: int = 5):
    """အသုံးပြုသူရဲ့ မေးခွန်းနဲ့ အနီးစပ်ဆုံးတူညီတဲ့ Procedure တွေကို ရှာဖွေပေးသည်"""


    # 1. Reading the embedded data file
    print(f" Reading file from '{EMBEDDED_FILE_NAME}'...")
    try:
        with open(EMBEDDED_FILE_NAME, 'r', encoding='utf-8') as f:
            data_embeddings = json.load(f)
        print(f" Successfully read embedding file ({len(data_embeddings)} entries)")
    except FileNotFoundError:
        print(f"'{EMBEDDED_FILE_NAME}' file not found. Please run the embedding generator script first.")
        return

    # 2. convert user query to embedding vector
    print(f"\nConverting user query '{user_query}' to vector...")
    query_vector = get_query_embedding(user_query, MODEL_NAME)
    if not query_vector:
        return

    # 3. Calculate similarity scores
    print("Calculating similarity scores...")
    scores = []
    for proc in data_embeddings:
        proc_name = proc["ReportName"]
        proc_vector = proc["Vector"]
        score = cosine_similarity(query_vector, proc_vector)
        scores.append({"ReportName": proc_name, "Score": score})

    # 4. Sort and display the top K results
    sorted_scores = sorted(scores, key=lambda x: x["Score"], reverse=True)

    print("\n--- Similar Procedures ---")
    for i in range(min(top_k, len(sorted_scores))):
        print(f"{i+1}. {sorted_scores[i]['ReportName']} (Score: {sorted_scores[i]['Score']:.4f})")


if __name__ == "__main__":
 
    # Sample query in Burmese
    test_query_burmese = "ကုန်ပစ္စည်း အလိုက် အရောင်းစာရင်းကို ကြည့်ချင်တယ်"
    # test_query_burmese = "အသုံးစရိတ် စားရင်း ဘယ်လိုကြည့်ရမလဲ"
    # test_query_burmese = "Invoice အလိုက် ရောင်းအားစာရင်း ထုတ်ပေးပါ။"
    # test_query_burmese = "ပစ္စည်းအရောင်းစာရင်းကို ကြည့်ချင်တယ်"

    # Non Sense Query
    # test_query_burmese = "အိမ်မှာ ဘာလုပ်ရမလဲ"
    # test_query_burmese = "ကျောင်းမှန်မှန်တတ် စာမခက်"
    
    find_most_similar_procedures(test_query_burmese)        

