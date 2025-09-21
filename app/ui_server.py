# app/ui_server.py
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
from pathlib import Path
import os
from .retriever import answer as get_answer
from .ingest import ingest_pdf_chunk  # we'll create a function to ingest a single PDF

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "replace-with-a-secure-random-key"  # needed for session
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    if "history" not in session:
        session["history"] = []
    return render_template("index.html", chat_history=session["history"])

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json or request.form
    query = data.get("query") if data else None
    if not query:
        return jsonify({"error": "No query provided"}), 400
    try:
        resp = get_answer(query, k=3)
        # append to session memory
        session["history"].append({"question": query, "answer": resp["answer"]})
        session.modified = True
        return jsonify(resp)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        # ingest the uploaded PDF
        ingest_pdf_chunk(filepath)
        return jsonify({"success": f"{filename} uploaded and ingested."})
    else:
        return jsonify({"error": "File type not allowed"}), 400

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
