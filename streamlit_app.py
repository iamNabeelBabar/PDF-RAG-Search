import streamlit as st
import requests
from io import BytesIO
import os

# FastAPI backend URL
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:4545")

# Streamlit app configuration
st.set_page_config(page_title="PDF Upload and RAG Search", layout="wide")

# Sidebar for API keys
st.sidebar.header("API Keys")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password", value=st.session_state.get("openai_api_key", ""))
pinecone_api_key = st.sidebar.text_input("Pinecone API Key", type="password", value=st.session_state.get("pinecone_api_key", ""))

# Store API keys in session state
st.session_state.openai_api_key = openai_api_key
st.session_state.pinecone_api_key = pinecone_api_key

# Title and description
st.title("PDF Upload and RAG Search App")
st.markdown("Upload a PDF file to store its contents in Pinecone and query it using Retrieval-Augmented Generation (RAG).")

# Create two tabs: Upload and Search
tab1, tab2 = st.tabs(["Upload PDF", "Search"])

# Upload PDF Tab
with tab1:
    st.header("Upload PDF")
    with st.form("upload_form"):
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
        index_name = st.text_input("Index Name", value="main")
        namespace = st.text_input("Namespace", value="default")
        submit_button = st.form_submit_button("Upload")

        if submit_button:
            if not openai_api_key or not pinecone_api_key:
                st.error("Please provide both OpenAI and Pinecone API keys in the sidebar.")
            elif uploaded_file is not None:
                try:
                    # Prepare file for upload
                    files = {"file": (uploaded_file.name, uploaded_file.read(), "application/pdf")}
                    params = {
                        "index_name": index_name,
                        "namespace": namespace
                    }
                    headers = {
                        "OpenAI-API-Key": openai_api_key,
                        "Pinecone-API-Key": pinecone_api_key
                    }
                    
                    # Send POST request to FastAPI /uploadfile/ endpoint
                    response = requests.post(
                        f"{API_BASE_URL}/files/uploadfile/",
                        files=files,
                        params=params,
                        headers=headers,
                        timeout=120
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"File '{result['filename']}' uploaded successfully!")
                        st.write(f"Number of pages: {result['num_pages']}")
                        st.write(f"Number of chunks: {result['num_chunks']}")
                        st.write(f"Status: {result['status']}")
                    else:
                        st.error(f"Upload failed: {response.json().get('detail', 'Unknown error')}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Error connecting to the server: {str(e)}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")
            else:
                st.warning("Please upload a PDF file before submitting.")

# Search Tab
with tab2:
    st.header("Search with RAG")
    with st.form("search_form"):
        query = st.text_area("Enter your query")
        index_name = st.text_input("Index Name", value="main", key="search_index")
        namespace = st.text_input("Namespace", value="default", key="search_namespace")
        top_k = st.number_input("Top K Results", min_value=1, max_value=10, value=5, step=1)
        search_button = st.form_submit_button("Search")

        if search_button:
            if not openai_api_key or not pinecone_api_key:
                st.error("Please provide both OpenAI and Pinecone API keys in the sidebar.")
            elif query.strip():
                try:
                    # Prepare payload for RAG search
                    payload = {
                        "index_name": index_name,
                        "namespace": namespace,
                        "query": query,
                        "top_k": int(top_k)
                    }
                    headers = {
                        "OpenAI-API-Key": openai_api_key,
                        "Pinecone-API-Key": pinecone_api_key
                    }
                    
                    # Send POST request to FastAPI /rag-search endpoint
                    response = requests.post(
                        f"{API_BASE_URL}/retrieve/rag-search",
                        json=payload,
                        headers=headers,
                        timeout=120
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.write("**Query:**")
                        st.write(result["query"])
                        st.write("**Answer:**")
                        st.markdown(result["answer"])
                    else:
                        st.error(f"Search failed: {response.json().get('detail', 'Unknown error')}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Error connecting to the server: {str(e)}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")
            else:
                st.warning("Please enter a query before submitting.")
