# app/retriever.py
from pathlib import Path
from langchain_community.vectorstores import Chroma
from .embeddings import get_embeddings
from .qa import build_qa

_DB_DIR = "db"

def load_vectordb():
    embeddings = get_embeddings()
    vectordb = Chroma(persist_directory=_DB_DIR, embedding_function=embeddings)
    return vectordb

def answer(query: str, k: int = 3):
    """
    If docs exist → run RetrievalQA.
    If no docs → fallback to plain LLM response.
    """
    vectordb = load_vectordb()

    # check if vectordb is empty
    try:
        collection = vectordb._collection
        num_docs = collection.count()
    except Exception:
        num_docs = 0

    if num_docs == 0:
        # fallback: plain LLM
        from .qa import build_plain_llm
        llm = build_plain_llm()
        resp = llm.invoke(query)
        return {"answer": resp, "sources": []}

    # normal retrieval QA
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    qa = build_qa(retriever)
    result = qa.invoke({"query": query})

    answer_text = result.get("result") or result.get("answer") or ""
    sources = []
    for d in result.get("source_documents", []):
        sources.append({
            "source": d.metadata.get("source", "unknown"),
            "snippet": (d.page_content[:300] + "...") if len(d.page_content) > 300 else d.page_content
        })

    return {"answer": answer_text, "sources": sources}
