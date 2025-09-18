from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

def load_and_chunk_docs(data_folder="data", chunk_size=500, chunk_overlap=50):
    all_docs = []

    for filename in os.listdir(data_folder):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_folder, filename))
            docs = loader.load()
            all_docs.extend(docs)

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunked_docs = text_splitter.split_documents(all_docs)
    print(f"Loaded {len(all_docs)} documents and split into {len(chunked_docs)} chunks.")
    return chunked_docs

if __name__ == "__main__":
    chunks = load_and_chunk_docs()
