from fastapi import FastAPI
import uvicorn
# from pydantic import BaseModel
import os
import sys
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import chatting_service
from services import embedding_service
from models.data import QueryData , FeedData

# class QueryData(BaseModel):
#     query: str

app = FastAPI()

@app.get("/")
def greeting():
    return {"message": "Hello, World!"}

@app.post("/feed")
async def receive_feed_data(data: list[FeedData]):
    datajson = json.dumps([item.model_dump() for item in data])
    result = await embedding_service.execute(_isBulkFeed=False, _feedJson= datajson)
    return {"status": "Feed completed", "records_processed": (f"Successfully embedded and saved {result} records.")}

@app.post("/query")
async def get_query(data: QueryData):
    result = await chatting_service.get_chat_response(data.query)
    return {"query": data.query, "response": result}


@app.post("/feedbulk")
async def feed_data():
    result = await embedding_service.execute(_isBulkFeed=True)
    return {"status": "Bulk Feed completed", "records_processed": (f"Successfully embedded and saved {result} records.")}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)