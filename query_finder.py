import os
import json
from dotenv import load_dotenv
import numpy as np
from google import genai
from google.genai.types import EmbedContentConfig

load_dotenv()
GOOGLE_API_KEY = os.getenv('API_KEY')


MODEL_NAME = "gemini-embedding-001" 

# --- File Paths ---
EMBEDDED_FILE_NAME = "./data/embedded_data.json"

def get_query_embedding(query_text: str, model) -> list[float] | None:
   
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

    dot_product = np.dot(vec1, vec2)
    # np.linalg.norm က vector တစ်ခုရဲ့ အလျား (magnitude) ကို ရှာပေးတာပါ
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
        
    return dot_product / (norm_vec1 * norm_vec2)    

def find_most_similar_result(user_query: str, top_k: int = 5) -> list[dict] | None:
   
    try:
        with open(EMBEDDED_FILE_NAME, 'r', encoding='utf-8') as f:
            data_embeddings = json.load(f)
        print(f" Successfully read embedding file ({len(data_embeddings)} entries)")
    except FileNotFoundError:
        print(f"'{EMBEDDED_FILE_NAME}' file not found. Please run the embedding generator script first.")
        return

    print(f"\nConverting user query '{user_query}' to vector...")
    query_vector = get_query_embedding(user_query, MODEL_NAME)
    if not query_vector:
        return

  
    print("Calculating similarity scores...")
    scores = []
    for proc in data_embeddings:
        proc_name = proc["ReportName"]
        proc_description = proc["Description"]
        proc_vector = proc["Vector"]
        score = cosine_similarity(query_vector, proc_vector)
        scores.append({"ReportName": proc_name,"Description": proc_description,"Score": score})

    sorted_scores = sorted(scores, key=lambda x: x["Score"], reverse=True)

    return sorted_scores[:top_k]

def find_best_response_with_gemini(user_query : str,similar_results: list[dict]) -> str:
    
    print(f"Finding best response for query: {user_query}")
    
    if not similar_results:
        return "No similar results found."

    context_for_prompt = ""
    for i, report in enumerate(similar_results):
        report_name = report['ReportName']
        description = report["Description"]
        
        context_for_prompt += f"--- Report {i+1} ---\n"
        context_for_prompt += f"Report အမည်: {report_name}\n"
        context_for_prompt += f"Report ဖော်ပြချက်: {description}\n\n"

    
    prompt = f"""
        သင်သည် ERP စနစ်အတွက် အထူးပြုမေးခွန်းများဖြေဆိုပေးသော Retrieval-Augmented Generation (RAG) AI တစ်ဦးဖြစ်ပါသည်။
        သင်၏ အဓိကတာဝန်မှာ အောက်တွင်ပေးထားသော "ရှာဖွေတွေ့ရှိထားသော အချက်အလက် (Context)" ကိုသာ အသုံးပြု၍ "အသုံးပြုသူ၏ မေးခွန်း" ကို တိကျစွာဖြေဆိုရန် ဖြစ်သည်။

        **လိုက်နာရန် စည်းမျဉ်းများ:**

        1.  **Context ကိုသာ အခြေခံပါ:** သင်၏အဖြေသည် ပေးထားသော "ရှာဖွေတွေ့ရှိထားသော အချက်အလက် (Context)" ၏ ဘောင်အတွင်း၌သာ ရှိရမည်။ သင်၏ ကိုယ်ပိုင်အသိပညာ သို့မဟုတ် အခြားသော အချက်အလက်ရင်းမြစ်များကို **လုံးဝအသုံးမပြုပါနှင့်**။
        2.  **တိုက်ရိုက်နှင့် တိကျပါ:** မေးခွန်းကို တိုက်ရိုက်ဖြေဆိုပါ။ အဖြေကို တိုတို၊ လိုရင်းနှင့် တိကျအောင်ရေးပါ။ မလိုအပ်သော နိဒါန်းပျိုးခြင်း၊ နှုတ်ဆက်ခြင်း သို့မဟုတ် အပိုစကားများ ထည့်သွင်းရန်မလိုအပ်ပါ။
        3.  **အချက်အလက်မရှိက ဝန်ခံပါ:** အကယ်၍ ပေးထားသော Context သည် မေးခွန်းကိုဖြေဆိုရန် လုံလောက်သော အချက်အလက်မပါဝင်ပါက **"ပေးထားသော အချက်အလက်များအရ ဤမေးခွန်းကို ဖြေဆိုနိုင်ခြင်းမရှိပါ"** ဟု တိကျစွာဖြေဆိုပါ။ မမှန်ကန်သော အချက်အလက်များကို မှန်းဆ၍ **လုံးဝမဖြေဆိုပါနှင့်**။
        4.  **ပေါင်းစပ်ဖြေဆိုခြင်း:** မေးခွန်းက အချက်အလက်များကို ပေါင်းစပ်သုံးသပ်ရန် လိုအပ်ပါက (ဥပမာ - နှိုင်းယှဉ်ခိုင်းခြင်း)၊ Context ထဲမှ သက်ဆိုင်ရာ အချက်အလက်များကိုသာ ပေါင်းစပ်ပြီး အဖြေတစ်ခုအဖြစ် ပြန်လည်ဖွဲ့စည်းပါ။ Context တွင်မပါသော အချက်အလက်အသစ်များကို မထည့်သွင်းပါနှင့်။

        =========================
        **အသုံးပြုသူ၏ မေးခွန်း:** "{user_query}"
        =========================

        **ရှာဖွေတွေ့ရှိထားသော Report များ:**
        {context_for_prompt}
        =========================

        **သင်၏ အဖြေ:**
    """    
    
    # Call Gemini to get the best response
    client = genai.Client(api_key=GOOGLE_API_KEY)
    response = client.models.generate_content(
        model='gemini-2.0-flash-001', contents=f'{prompt}',
    )

    return response.text.strip()

def run_rag_query(user_query: str):
   
    similar_results = find_most_similar_result(user_query)
    
    if not similar_results:
        return "မေးခွန်းနှင့် ဆက်စပ်သော Report များကို ရှာမတွေ့ပါ။"

    final_choice = find_best_response_with_gemini(user_query, similar_results)
    

    print("\n==============================================")
    print(f"👤  User Query: \"{user_query}\"")
    print(f"🤖  Cortex's Final Recommendation: {final_choice}")
    print("==============================================")


