# app/qa.py
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

def build_qa(retriever):
    generator = pipeline("text2text-generation", model="google/flan-t5-base", device=-1)
    llm = HuggingFacePipeline.from_model_id(
        model_id="google/flan-t5-base",
        task="text2text-generation",
        model_kwargs={"max_length": 256}
    )
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    return qa

def build_plain_llm():
    """Return a plain LLM pipeline for non-RAG mode."""
    generator = pipeline("text2text-generation", model="google/flan-t5-base", device=-1)
    llm = HuggingFacePipeline.from_model_id(
        model_id="google/flan-t5-base",
        task="text2text-generation",
        model_kwargs={"max_length": 256}
    )
    return llm
