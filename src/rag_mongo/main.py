from fastapi import FastAPI
from pydantic import BaseModel

from rag_mongo.rag import get_query_result

app = FastAPI(
    title="RAG Mongo API",
    description="RAG API using MongoDB",
    version="1.0.0"
)


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def home():
    return {
        "message": "RAG API is running"
    }


@app.post("/chat")
def ask_question(data: ChatRequest):
    answer = get_query_result(data.message)

    print(answer, "question answer")

    return {
        "answer": answer
    }