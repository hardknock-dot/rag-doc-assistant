# app/qa.py
from langchain.chains import RetrievalQA
# choose your preferred HuggingFace wrapper import present in your env
from langchain_huggingface import HuggingFacePipeline  # ensure package is installed
from transformers import pipeline

def build_qa(retriever):
    """
    Build and return a RetrievalQA chain using a local HF model.
    We avoid creating the retriever here so the caller (retriever.py) can pass it.
    """
    # local LLM via transformers
    generator = pipeline("text2text-generation", model="google/flan-t5-base", device=-1)
    llm = HuggingFacePipeline.from_model_id(
        model_id="google/flan-t5-base",
        task="text2text-generation",
        model_kwargs={"max_length": 256}
    )
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    return qa
