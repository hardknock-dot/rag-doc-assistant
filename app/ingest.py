# app/ingest.py
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from .embeddings import get_embeddings
import os

def ingest():
    print("Step 1: Loading documents...")
    
    # Assuming your PDFs are in the "data/" folder
    docs = []
    for filename in os.listdir("data"):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join("data", filename))
            docs.extend(loader.load())

    print(f"Loaded {len(docs)} documents. Splitting into chunks...")
    
    # Chunks (basic splitting)
    chunks = []
    for doc in docs:
        text = doc.page_content
        chunk_size = 500
        for i in range(0, len(text), chunk_size):
            chunks.append(doc.__class__(page_content=text[i:i+chunk_size], metadata=doc.metadata))

    print(f"{len(chunks)} chunks created. Creating embeddings and saving to Chroma...")

    embeddings = get_embeddings()
    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="db")
    vectordb.persist()
    print("Ingestion completed and saved to Chroma.")

if __name__ == "__main__":
    ingest()
