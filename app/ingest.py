# app/ingest.py
from langchain.document_loaders import PyPDFLoader
from .embeddings import get_embeddings
from langchain_community.vectorstores import Chroma

DB_DIR = "db"

def ingest_pdf_chunk(pdf_path):
    """Ingest a single PDF into Chroma DB"""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    embeddings = get_embeddings()
    vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    vectordb.add_documents(docs)
    return True
