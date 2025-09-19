from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Step 1: Load embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Step 2: Load existing Chroma DB
vectordb = Chroma(
    persist_directory="db",
    embedding_function=embeddings
)

# Step 3: Create a retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# Step 4: Test a query
query = "What is a cat?"
print(f"Query: {query}")
results = retriever.get_relevant_documents(query)

print("\nTop results:")
for i, doc in enumerate(results, start=1):
    print(f"\nResult {i}:")
    print(doc.page_content[:300])  # print first 300 chars
