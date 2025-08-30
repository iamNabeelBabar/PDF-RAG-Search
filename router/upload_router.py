from fastapi import APIRouter, File, UploadFile, Header
from utils.upload_utils import load_file, clean_data, splitted_data
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from services.pinecone_services import index_creation
import tempfile
import os

router = APIRouter(tags=["Upload"])

@router.post("/uploadfile/")
async def create_upload_file(
    file: UploadFile = File(...),
    index_name: str = "main",
    namespace: str = "default",
    openai_api_key: str = Header(...),
    pinecone_api_key: str = Header(...)
):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    
    # Load and process the file
    pages = load_file(tmp_path)

    # Cleaning
    clean_pages = clean_data(pages)

    # Splitting / chunking
    doc_list = splitted_data(clean_pages)

    # Embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=openai_api_key
    )

    # Ensure Pinecone index exists
    index = index_creation(index_name=index_name, dimension=1536, api_key=pinecone_api_key)

    # Store chunks into Pinecone
    vectorstore = PineconeVectorStore.from_documents(
        documents=doc_list,
        embedding=embeddings,
        index_name=index_name,
        namespace=namespace
    )

    # Clean up temporary file
    os.unlink(tmp_path)

    return {
        "filename": file.filename,
        "num_pages": len(clean_pages),
        "num_chunks": len(doc_list),
        "status": f"Inserted into Pinecone index '{index_name}' under namespace '{namespace}'"
    }
