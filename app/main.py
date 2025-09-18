# app/main.py
from dotenv import load_dotenv
import os

# Load .env variables
load_dotenv()

# Access OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
print("OpenAI API Key loaded:", bool(openai_api_key))

def main():
    print("RAG Document Q&A setup working!")

if __name__ == "__main__":
    main()
