from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from ingest import load_and_chunk_docs

print("Step 1: Loading documents...")
chunks = load_and_chunk_docs(data_folder="data")

print(f"Step 2: {len(chunks)} chunks loaded. Creating embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Step 3: Storing chunks in Chroma vector DB...")
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="db"
)

vectordb.persist()
print("✅ Day 3 complete: Documents embedded and stored in Chroma vector DB.")
