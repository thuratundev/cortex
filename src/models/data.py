from pydantic import BaseModel

class QueryData(BaseModel):
    query: str

class FeedData(BaseModel):
    Id: int
    ReportName: str
    Description: str
    Keywords: list[str]
    SampleQueries: list[str]
    Version: float
        