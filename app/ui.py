import streamlit as st
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from app.embeddings import get_embeddings  # make sure this returns HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# --- Load embeddings and vectorstore ---
persist_directory = "db"  # directory where Chroma DB is saved
embeddings = get_embeddings()

vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# --- Load LLM pipeline ---
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    device=-1  # CPU
)

llm = HuggingFacePipeline(pipeline=generator)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# --- Streamlit UI ---
st.title("📚 RAG QA System")
st.write("Ask questions based on your documents!")

query = st.text_input("Enter your question:")

if query:
    with st.spinner("Generating answer..."):
        result = qa.run(query)
    st.subheader("Answer")
    st.write(result)
