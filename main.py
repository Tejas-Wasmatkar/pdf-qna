from flask import Flask, request, jsonify, render_template
import os
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
index = None
chunks = []

# Extract text from PDF
def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Split text into chunks
def split_text(text, chunk_size=500):
    sentences = text.split('. ')
    chunk_list, chunk = [], ""
    for sentence in sentences:
        if len(chunk) + len(sentence) < chunk_size:
            chunk += sentence + ". "
        else:
            chunk_list.append(chunk.strip())
            chunk = sentence + ". "
    if chunk:
        chunk_list.append(chunk.strip())
    return chunk_list

# Build FAISS index
def generate_index(chunks):
    embeddings = embedding_model.encode(chunks)
    vec_index = faiss.IndexFlatL2(embeddings.shape[1])
    vec_index.add(np.array(embeddings))
    return vec_index

# Call LLaMA 3 via Ollama
def generate_response(context, question):
    prompt = f"""Answer the question using the context below.

### Context:
{context}

### Question:
{question}

### Answer:"""
    res = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    })
    if res.ok:
        return res.json().get("response", "").strip()
    return "❌ Failed to get response from LLaMA."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    global index, chunks
    uploaded_file = request.files.get("file")
    if uploaded_file:
        try:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], uploaded_file.filename)
            uploaded_file.save(file_path)

            text = extract_text_from_pdf(file_path)
            if not text.strip():
                return jsonify({"success": False, "error": "PDF contains no text."}), 400

            chunks = split_text(text)
            if not chunks:
                return jsonify({"success": False, "error": "Text splitting failed."}), 400

            index = generate_index(chunks)
            return jsonify({"success": True})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500
    return jsonify({"success": False, "error": "No file uploaded."}), 400

@app.route("/ask", methods=["POST"])
def ask():
    global index, chunks
    data = request.get_json()
    question = data.get("question", "").strip()

    if not index or not chunks:
        return jsonify({"answer": "❌ Please upload a PDF first."})
    if not question:
        return jsonify({"answer": "❌ Question is empty."})

    try:
        query_embedding = embedding_model.encode([question])
        distances, indices = index.search(np.array(query_embedding), 5)
        context = "\n\n".join([chunks[i] for i in indices[0]])
        answer = generate_response(context, question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"answer": f"❌ Error: {str(e)}"})
    
if __name__ == "__main__":
    app.run(debug=True)
