from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Loaders
def load_file(file_path):
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        return pages
    except Exception as e:
        raise ValueError(f"Failed to load PDF file: {str(e)}")

# Cleaning
def clean_data(pages):
    for page in pages:
        text = page.page_content
        page.page_content = " ".join(text.split())
    return pages

# Splitting
re_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200
)

# Recursive character text splitter
def splitted_data(cleaned_data):
    doc_list = []
    for page in cleaned_data:
        if not hasattr(page, 'page_content') or not isinstance(page.metadata, dict):
            raise ValueError("Invalid page data or metadata")
        pg_split = re_splitter.split_text(page.page_content)
        for pg_sub_split in pg_split:
            metadata = {"source": "AI Embedding", "page_no": page.metadata.get("page", 0) + 1}
            doc_string = Document(page_content=pg_sub_split, metadata=metadata)
            doc_list.append(doc_string)
    return doc_list