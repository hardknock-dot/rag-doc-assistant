# app/qa_improved.py - Improved QA with better model selection
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from langchain_openai import ChatOpenAI
from transformers import pipeline
import torch
from .config import LLMConfig

def build_qa_improved(retriever):
    """Build QA with the best available model."""
    model_type = LLMConfig.get_best_available_model()
    
    if model_type == "openai":
        return _build_openai_qa(retriever)
    else:
        return _build_hf_qa(retriever)

def build_plain_llm_improved():
    """Build plain LLM with the best available model."""
    model_type = LLMConfig.get_best_available_model()
    
    if model_type == "openai":
        return _build_openai_llm()
    else:
        return _build_hf_llm()

def _build_openai_qa(retriever):
    """Build QA using OpenAI API."""
    try:
        llm = ChatOpenAI(
            model_name=LLMConfig.OPENAI_MODEL,
            temperature=LLMConfig.TEMPERATURE,
            max_tokens=LLMConfig.MAX_LENGTH,
            api_key=LLMConfig.OPENAI_API_KEY
        )
        print(f"Using OpenAI model: {LLMConfig.OPENAI_MODEL}")
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
        return qa
    except Exception as e:
        print(f"OpenAI QA failed: {e}")
        print("Falling back to HuggingFace models...")
        return _build_hf_qa(retriever)

def _build_openai_llm():
    """Build plain LLM using OpenAI API."""
    try:
        llm = ChatOpenAI(
            model_name=LLMConfig.OPENAI_MODEL,
            temperature=LLMConfig.TEMPERATURE_PLAIN,
            max_tokens=LLMConfig.MAX_LENGTH,
            api_key=LLMConfig.OPENAI_API_KEY
        )
        print(f"Using OpenAI plain LLM: {LLMConfig.OPENAI_MODEL}")
        return llm
    except Exception as e:
        print(f"OpenAI plain LLM failed: {e}")
        print("Falling back to HuggingFace models...")
        return _build_hf_llm()

def _build_hf_qa(retriever):
    """Build QA using HuggingFace models."""
    for model_name in LLMConfig.HF_MODELS:
        try:
            print(f"Trying HuggingFace model: {model_name}")
            llm = HuggingFacePipeline.from_model_id(
                model_id=model_name,
                task="text2text-generation",
                model_kwargs={
                    "max_length": LLMConfig.MAX_LENGTH,
                    "temperature": LLMConfig.TEMPERATURE
                }
            )
            print(f"Successfully loaded: {model_name}")
            qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
            return qa
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            continue
    
    raise Exception("Could not load any HuggingFace model for QA")

def _build_hf_llm():
    """Build plain LLM using HuggingFace models."""
    for model_name in LLMConfig.HF_MODELS:
        try:
            print(f"Trying HuggingFace plain LLM: {model_name}")
            llm = HuggingFacePipeline.from_model_id(
                model_id=model_name,
                task="text2text-generation",
                model_kwargs={
                    "max_length": LLMConfig.MAX_LENGTH,
                    "temperature": LLMConfig.TEMPERATURE_PLAIN
                }
            )
            print(f"Successfully loaded plain LLM: {model_name}")
            return llm
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            continue
    
    raise Exception("Could not load any HuggingFace model for plain LLM")

