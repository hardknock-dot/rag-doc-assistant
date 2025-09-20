# app/ui.py
from .qa import qa

def main():
    print("Welcome to your RAG assistant!")
    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        result = qa.invoke(query)
        print("\nAnswer:", result["result"])

if __name__ == "__main__":
    main()