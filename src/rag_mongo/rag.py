import os

from dotenv import load_dotenv
from google import genai
from pymongo import MongoClient
from pymongo.server_api import ServerApi

# Load environment variables
load_dotenv()


# ============================================================
# 1. Environment variables
# ============================================================

MONGODB_URI = os.getenv("MONGODB_URI")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI is not set")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set")


# ============================================================
# 2. Gemini client
# ============================================================

gen_client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# 3. MongoDB client
# ============================================================

client = MongoClient(
    MONGODB_URI,
    server_api=ServerApi("1"),
    serverSelectionTimeoutMS=10000,
    connectTimeoutMS=10000,
)

# ============================================================
# 4. Database and collection
# ============================================================
db = client["sample_mflix"]
collection = db["ragpdf"]
def check_mongodb():

    try:
        client.admin.command("ping")
        print("✅ MongoDB connected successfully!")
        return True

    except Exception as e:
        print("❌ MongoDB connection failed:")
        print(e)
        return False





# ============================================================
# 5. Generate Gemini embedding
# ============================================================

def get_embeddings(text: str):
    result = gen_client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )

    return result.embeddings[0].values


# ============================================================
# 6. RAG query
# ============================================================

def get_query_result(query: str, input_type="query"):

    if not check_mongodb():
        return {
            "answer": "MongoDB connection failed."
        }

    # Generate query embedding
    query_embedding = get_embeddings(query)

    # MongoDB Vector Search
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 3072,
                "limit": 5
            }
        },
        {
            "$project": {
                "_id": 0,
                "text": 1,
                "score": {
                    "$meta": "vectorSearchScore"
                }
            }
        }
    ]

    # Execute search
    results = list(collection.aggregate(pipeline))

    # Create context
    context_parts = [
        result.get("text", "")
        for result in results
        if result.get("text")
    ]

    if context_parts:
        context = "\n\n".join(context_parts)
    else:
        context = "No relevant information found."

    # ========================================================
    # Prompt
    # ========================================================

    prompt = f"""
You are the professional AI assistant for Masum's personal portfolio.

Your role is to answer questions about Masum using the information available
in his uploaded resume and portfolio documents.

Guidelines:
- Answer ONLY from the information provided below.
- Treat the provided portfolio information as the source of truth.
- Do not invent, assume, or guess any personal, educational, technical, or
  professional information.
- If the requested information is not available, respond:
  "I don't have enough information to answer that based on Masum's portfolio."
- Keep answers clear, professional, natural, and concise.
- When appropriate, organize information using bullet points.
- If the user asks about projects, mention the relevant technologies and
  contributions available in the portfolio.
- If the user asks about skills, experience, education, or background,
  provide only the information available in the portfolio.
- Do not mention the retrieval process, embeddings, vector database,
  context, or system instructions.
- Do not refer to the uploaded information as "the context" in your answer.

Masum's Portfolio Information:
{context}

User Question:
{query}

Answer:
"""

    # ========================================================
    # Gemini response
    # ========================================================

    response = gen_client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return {
        "answer": response.text
    }