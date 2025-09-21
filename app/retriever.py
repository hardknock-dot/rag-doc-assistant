# app/retriever.py
# Minimal adapter to return answer + sources.
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# import the embedding factory and llm-qa build if you have them
# Adjust these imports to match your project module names (relative imports)
from .embeddings import get_embeddings  # must return HuggingFaceEmbeddings or equivalent
from .qa import build_qa  # we'll assume qa.py exposes build_qa()

_DB_DIR = "db"

def load_vectordb():
    embeddings = get_embeddings()
    vectordb = Chroma(persist_directory=_DB_DIR, embedding_function=embeddings)
    return vectordb

# build_qa should accept a retriever and return a RetrievalQA chain
def answer(query: str, k: int = 3):
    """
    Returns dict: { "answer": str, "source_documents": [ {source, snippet}, ... ] }
    """
    vectordb = load_vectordb()
    retriever = vectordb.as_retriever(search_kwargs={"k": k})

    qa = build_qa(retriever)  # delegates LLM selection to qa.py
    # use invoke to be compatible with newer langchain versions
    result = qa.invoke({"query": query})
    answer_text = result.get("result") or result.get("answer") or ""
    sources = []
    for d in result.get("source_documents", []):
        sources.append({
            "source": d.metadata.get("source", "unknown"),
            "snippet": (d.page_content[:300] + "...") if len(d.page_content) > 300 else d.page_content
        })
    return {"answer": answer_text, "sources": sources}
