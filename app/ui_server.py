from flask import Flask, render_template, request, jsonify
from .qa import qa  # your QA chain
# from embeddings import get_embeddings  # if needed

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question")
    if not question:
        return jsonify({"answer": "Please enter a question."})
    
    # Call your QA chain
    result = qa.invoke(question)  # result should be a dict with key "result"
    answer = result.get("result", "No answer found.")
    
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
