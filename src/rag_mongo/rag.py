
import os
from dotenv import load_dotenv

load_dotenv()
print(os.getenv("OPENAI_API_KEY"))

from google import genai
import os


from sentence_transformers import SentenceTransformer
# specify the embedding mode
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embeddings(text):
    return model.encode(text).tolist()



# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter


# In[49]:


# loader =PyPDFLoader('https://investors.mongodb.com/node/12236/pdf')
# data =loader.load()


# In[24]:


# data


# In[7]:


# text_spliter =RecursiveCharacterTextSplitter(chunk_size=400,chunk_overlap=20)
# documents =text_spliter.split_documents(data)



# In[27]:


# prepare documment for insertion
# docs_to_insert =[{'text':doc.page_content,
#                   'embedding':get_embeddings(doc.page_content)
#                  }for doc in documents]



from pymongo import MongoClient
client =MongoClient(os.getenv("MONGODB_URI"))



# create collection
db =client['sample_mflix']
collection =db['ragpdf']










# from pymongo.operations import SearchIndexModel

# search_index_model = SearchIndexModel(
#     definition={
#         "fields": [
#             {
#                 "type": "vector",
#                 "path": "embedding",
#                 "numDimensions": 384,
#                 "similarity": "cosine"
#             }
#         ]
#     },
#     name="vector_index",
#     type="vectorSearch"
# )



from pymongo.server_api import ServerApi
# Create a new client and connect to the server
uri = "mongodb+srv://masumbilla10104_db_user:AqqaaE0MwxVokjzu@rag.gysh2uz.mongodb.net/?appName=RAG"
client = MongoClient(uri, server_api=ServerApi('1'))
# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)



GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set")

gen_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

def get_query_result(query,input_type='query'):
    query_embedding =get_embeddings(query)
    # Vector search pipeline
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 384,
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
    results =collection.aggregate(pipeline)
    # Create context

    if results:
        context = "\n\n".join(
        result.get("text", "")
        for result in results
        if result.get("text")
    )
    else:
       context = "No relevant information found."

    prompt = f"""
             You are a helpful AI assistant.

             Answer the question using ONLY the information
             provided in the context.

             If the answer is not available in the context,
             say: "I don't have enough information."

             Context:
             {context}

             Question:
             {query}

             Answer:
             """

    response = gen_client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt
    )
    
    return {
        "answer": response.text
    }

