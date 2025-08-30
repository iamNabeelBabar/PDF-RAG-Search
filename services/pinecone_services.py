# utils/index_utils.py
import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") # Ensure the Pinecone API key is set in the environment

def index_creation(index_name: str = "index-name", dimension: int = 1536):

    # Load API key from environment
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is not set in environment variables.")

    # Initialize Pinecone client
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Check if index exists, if not create
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            vector_type="dense",
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"Index '{index_name}' created successfully!")
    else:
        print(f"Index '{index_name}' already exists.")

    # Return the index object for use
    return pc.Index(index_name)



