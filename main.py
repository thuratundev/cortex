from fastapi import FastAPI
from pydantic import BaseModel
import query_finder


class QueryData(BaseModel):
    query: str


app = FastAPI()

@app.get("/")
def greeting():
    return {"message": "Hello, World!"}

@app.post("/query")
def process_query(data: QueryData):
    query_result = query_finder.run_rag_query(data.query)
    return {"result": query_result}   