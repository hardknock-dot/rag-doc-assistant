# app/qa.py
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from app.embeddings import get_embeddings
  # make sure this file returns HuggingFaceEmbeddings

# Load embeddings
embeddings = get_embeddings()

# Load vector database and provide embeddings
persist_directory = "db"
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# Create retriever
retriever = vectordb.as_retriever()

# Load HuggingFace LLM
llm_pipeline = HuggingFacePipeline.from_model_id(
    model_id="google/flan-t5-base",
    task="text2text-generation",
    model_kwargs={"max_length": 256}  # remove 'temperature' to avoid warning
)

# Create QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm_pipeline,
    retriever=retriever
)

if __name__ == "__main__":
    query = input("Enter your question: ")
    result = qa.invoke(query)
    print("Answer:", result["result"])
