import data_query_service
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import  app_config 
from agents import chatting_agent

def get_chat_response(user_query: str) -> str | None:
   similar_results = data_query_service.get_similar(user_query)
   return chatting_agent.chat(user_query, similar_results)

if __name__ == "__main__":
    user_query = "ရောင်းအားအကောင်းဆုံး ပစ္စည်းကို ဘယ်လိုသိနိုင်မလဲ။"
    response = get_chat_response(user_query)


    print("\n==============================================")
    print(f"👤  User Query: \"{user_query}\"")
    print(f"🤖  Cortex's Final Recommendation: {response}")
    print("==============================================")
    