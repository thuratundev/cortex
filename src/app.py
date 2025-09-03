from fastapi import FastAPI
from pydantic import BaseModel
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import chatting_service
from services import embedding_service

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


@app.post("/feed")
async def feed_data():
    result = await embedding_service.execute()
    return {"status": "completed", "records_processed": (f"Successfully embedded and saved {result} records.")}