import streamlit as st
import requests
import os
from io import BytesIO

# FastAPI backend URL (set via environment variable for deployment)
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:4545")

# Streamlit app configuration
st.set_page_config(page_title="PDF Upload and RAG Search", layout="wide")

# Sidebar for API keys and configuration
st.sidebar.header("Configuration")
st.sidebar.markdown(
    """
    Enter your API keys below. For Streamlit Community Cloud, set `API_BASE_URL`, 
    `OPENAI_API_KEY`, and `PINECONE_API_KEY` in the Streamlit Cloud dashboard under 'Secrets'.
    """
)
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password", value=st.session_state.get("openai_api_key", ""))
pinecone_api_key = st.sidebar.text_input("Pinecone API Key", type="password", value=st.session_state.get("pinecone_api_key", ""))

# Store API keys in session state
st.session_state.openai_api_key = openai_api_key
st.session_state.pinecone_api_key = pinecone_api_key

# Warning if using local API URL
if API_BASE_URL == "http://127.0.0.1:4545":
    st.sidebar.warning(
        """
        The API_BASE_URL is set to 'http://127.0.0.1:4545', which is for local testing only. 
        For the deployed app, update the API_BASE_URL in Streamlit Cloud secrets to your 
        deployed FastAPI backend URL (e.g., https://my-fastapi-backend.onrender.com).
        """
    )

# Title and description
st.title("PDF Upload and RAG Search App")
st.markdown(
    f"""
    Upload a PDF file to store its contents in Pinecone and query it using Retrieval-Augmented Generation (RAG).
    **Connected Backend:** {API_BASE_URL}
    """
)

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
                    # Read file content
                    file_content = uploaded_file.read()
                    files = {"file": (uploaded_file.name, file_content, "application/pdf")}
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
                        error_detail = response.json().get('detail', 'Unknown error')
                        st.error(f"Upload failed: {error_detail}")
                except requests.exceptions.ConnectionError:
                    st.error(f"Failed to connect to the backend at {API_BASE_URL}. Ensure the FastAPI server is running and the URL is correct.")
                except requests.exceptions.Timeout:
                    st.error("Request timed out. Check your network or backend server performance.")
                except requests.exceptions.RequestException as e:
                    st.error(f"Error connecting to the server: {str(e)}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")
            else:
                st.warning("Please upload a PDF file before submitting.")

# Search Tab
with tab2:
