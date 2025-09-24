# app/qa_advanced.py - Alternative with better models
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

def build_qa_advanced(retriever):
    """Build QA with more capable models for better responses."""
    models_to_try = [
        {
            "name": "microsoft/DialoGPT-large",
            "task": "text-generation",
            "kwargs": {
                "max_length": 512,
                "temperature": 0.7,
                "do_sample": True,
                "pad_token_id": 50256
            }
        },
        {
            "name": "google/flan-t5-large",
            "task": "text2text-generation",
            "kwargs": {"max_length": 512, "temperature": 0.7}
        },
        {
            "name": "google/flan-t5-xl",
            "task": "text2text-generation", 
            "kwargs": {"max_length": 512, "temperature": 0.7}
        }
    ]
    
    for model_config in models_to_try:
        try:
            print(f"Trying advanced model: {model_config['name']}")
            llm = HuggingFacePipeline.from_model_id(
                model_id=model_config["name"],
                task=model_config["task"],
                model_kwargs=model_config["kwargs"]
            )
            print(f"Successfully loaded: {model_config['name']}")
            break
        except Exception as e:
            print(f"Failed to load {model_config['name']}: {e}")
            continue
    else:
        raise Exception("Could not load any advanced model")
    
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    return qa

def build_plain_llm_advanced():
    """Return a more capable plain LLM pipeline."""
    models_to_try = [
        {
            "name": "microsoft/DialoGPT-large",
            "task": "text-generation",
            "kwargs": {
                "max_length": 512,
                "temperature": 0.8,
                "do_sample": True,
                "pad_token_id": 50256
            }
        },
        {
            "name": "google/flan-t5-large",
            "task": "text2text-generation",
            "kwargs": {"max_length": 512, "temperature": 0.8}
        },
        {
            "name": "google/flan-t5-xl",
            "task": "text2text-generation",
            "kwargs": {"max_length": 512, "temperature": 0.8}
        }
    ]
    
    for model_config in models_to_try:
        try:
            print(f"Trying advanced plain LLM: {model_config['name']}")
            llm = HuggingFacePipeline.from_model_id(
                model_id=model_config["name"],
                task=model_config["task"],
                model_kwargs=model_config["kwargs"]
            )
            print(f"Successfully loaded advanced plain LLM: {model_config['name']}")
            return llm
        except Exception as e:
            print(f"Failed to load {model_config['name']}: {e}")
            continue
    
    raise Exception("Could not load any advanced model for plain LLM")

# Option to use OpenAI API (if you have API key)
def build_openai_llm():
    """Alternative using OpenAI API for much better responses."""
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=512
        )
        return llm
    except ImportError:
        print("OpenAI not available. Install with: pip install langchain-openai")
        return None
    except Exception as e:
        print(f"OpenAI setup failed: {e}")
        return None

