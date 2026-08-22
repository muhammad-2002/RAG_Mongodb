from fastapi import FastAPI
from pydantic import BaseModel

from rag_mongo.rag import get_query_result

app = FastAPI(
    title="RAG Mongo API",
    description="RAG API using MongoDB",
    version="1.0.0"
)
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "https://muhammad-jade.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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