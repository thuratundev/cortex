import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import  app_config 
from agents import chatting_agent
from services import data_query_service

async def get_chat_response(user_query: str) -> str | None:
   similar_results = await data_query_service.get_similar(user_query)
   return await chatting_agent.chat(user_query, similar_results)


    