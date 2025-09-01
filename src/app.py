from fastapi import FastAPI
from pydantic import BaseModel
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import chatting_service

class QueryData(BaseModel):
    query: str

app = FastAPI()



@app.get("/")
def greeting():
    return {"message": "Hello, World!"}

@app.post("/query")
async def get_query(data: QueryData):
    result = await chatting_service.get_chat_response(data.query)
    return {"query": data.query, "response": result}


