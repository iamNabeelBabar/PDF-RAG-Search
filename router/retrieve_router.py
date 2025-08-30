from fastapi import APIRouter, Header
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone
import os

# Load Pinecone API key from environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Router object
router = APIRouter()

# Request schema
class QueryRequest(BaseModel):
    index_name: str
    namespace: str
    query: str
    top_k: int = 5


@router.post("/rag-search")
def rag_search(request: QueryRequest, openai_api_key: str = Header(...)):
    # Initialize OpenAI client with header-provided API key
    client = OpenAI(api_key=openai_api_key)

    # 1. Encode query using OpenAI embeddings
    embed = client.embeddings.create(
        model="text-embedding-3-small",
        input=request.query
    )
    query_vector = embed.data[0].embedding

    # 2. Connect to Pinecone index
    index = pc.Index(request.index_name)

    # 3. Search Pinecone
    results = index.query(
        namespace=request.namespace,
        vector=query_vector,
        top_k=request.top_k,
        include_metadata=True
    )

    # 4. Collect retrieved text chunks
    retrieved_chunks = [
        match["metadata"].get("text", "")
        for match in results["matches"]
    ]
    context = "\n".join(retrieved_chunks)

    # 5. Use GPT-4o-mini for final answer
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant using RAG."},
            {"role": "user", "content": f"Question: {request.query}\n\nContext:\n{context}"}
        ]
    )

    final_answer = completion.choices[0].message.content

    return {
        "query": request.query,
        # "retrieved_chunks": retrieved_chunks,
        "answer": final_answer
    }