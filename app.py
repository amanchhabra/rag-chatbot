from flask import Flask, flash, request, render_template, redirect, url_for, send_from_directory, Response, session
import os, pickle, faiss, numpy as np, json
from dotenv import load_dotenv
from ingest import build_index
from llm_provider import LLMProvider
import markdown
from datetime import datetime

load_dotenv()

# Paths
INDEX_PATH = "storage/index.faiss"
STORE_PATH = "storage/store.pkl"
CHAT_FILE = "storage/chat_history.json"
UPLOAD_FOLDER = "uploads"

# App init
app = Flask(__name__)
app.secret_key = "supersecret"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load FAISS + store
index, texts, metas = None, [], []
if os.path.exists(INDEX_PATH) and os.path.getsize(INDEX_PATH) > 0:
    try:
        index = faiss.read_index(INDEX_PATH)
    except Exception as e:
        print("Failed to load FAISS index:", e)
if os.path.exists(STORE_PATH):
    with open(STORE_PATH, "rb") as f:
        store = pickle.load(f)
    texts, metas = store.get("texts", []), store.get("metadatas", [])

# Chat history persistence
def load_chat_history():
    try:
        if os.path.exists(CHAT_FILE):
            with open(CHAT_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        print("Failed to load chat history:", e)
    return []

def save_chat_history(history):
    os.makedirs("storage", exist_ok=True)
    with open(CHAT_FILE, "w") as f:
        json.dump(history, f, indent=2)

chat_history = load_chat_history()

# Helpers
def get_llm():
    llm = LLMProvider(provider=session.get("llm_provider", "openai"))
    if not llm:
        flash("No API key found. Please set it in Settings.", "danger")
        return None
    return llm

def embed(texts):
    llm = get_llm()
    if not llm:
        return []
    arr = np.array(llm.embed(texts), dtype="float32")
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    return arr

def answer_question(query, top_k=10):
    llm = get_llm()
    if not llm:
        return "No LLM configured. Please check settings.", []

    if index is None or index.ntotal == 0:
        return "No documents found. Please upload a file or URL.", []

    qv = embed([query])
    scores, idxs = index.search(qv, top_k)

    sources = []
    for i in idxs[0]:
        meta = metas[i]
        sources.append({
            "source": meta["source"],
            "page": meta["page"],
            "link": meta.get("link", "#"),
            "preview": texts[i][:120] + "..."
        })

    context = "\n\n".join([texts[i] for i in idxs[0]])
    messages = [
        {"role": "system", "content": "Answer only using the context. If missing, say exactly 'No result found.'"},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]

    answer = llm.chat(messages, model=session.get("llm_model"))
    if "no result found." in answer.lower():
        sources = []
    return answer, sources

# Routes
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files.get("file")
        url = request.form.get("url")
        success = False

        if file and file.filename.endswith((".pdf", ".txt")):
            path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(path)
            success = build_index(file_path=path)
        elif url:
            success = build_index(url=url)
        else:
            flash("Please upload a PDF/TXT or provide a valid URL.", "danger")
            return redirect(url_for("upload"))

        # Reload
        global index, texts, metas
        index = faiss.read_index(INDEX_PATH)
        with open(STORE_PATH, "rb") as f:
            store = pickle.load(f)
        texts, metas = store["texts"], store["metadatas"]

        if success:
            flash("Source uploaded successfully.", "success")
        return redirect(url_for("home"))
    return render_template("upload.html")

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        q = request.form["query"]
        try:
            ans, sources = answer_question(q)
        except Exception as e:
            ans, sources = f"Error: {e}", []
        ans_html = markdown.markdown(ans)
        chat_history.insert(0, {
            "timestamp": datetime.utcnow().isoformat(),
            "query": q,
            "answer": ans_html,
            "sources": sources
        })
        save_chat_history(chat_history)
    return render_template("index.html", history=chat_history)

@app.route("/settings", methods=["GET", "POST"])
def settings():
    if request.method == "POST":
        # LLM settings
        session["llm_provider"] = request.form.get("llm_provider", "openai")
        session["llm_model"] = request.form.get("llm_model")

        # API keys
        for key in ["openai", "anthropic", "huggingface", "oci"]:
            session[f"{key}_key"] = request.form.get(f"{key}_key")
        session["oci_endpoint"] = request.form.get("oci_endpoint")

        # Integrations
        session["confluence_enabled"] = "confluence_enabled" in request.form
        session["confluence_key"] = request.form.get("confluence_key")
        session["jira_enabled"] = "jira_enabled" in request.form
        session["jira_key"] = request.form.get("jira_key")

        flash("Settings saved.", "success")
        return redirect(url_for("settings"))

    return render_template("settings.html",
                           llm_provider=session.get("llm_provider", "openai"),
                           llm_model=session.get("llm_model", "gpt-4o-mini"),
                           openai_key=session.get("openai_key", ""),
                           anthropic_key=session.get("anthropic_key", ""),
                           huggingface_key=session.get("huggingface_key", ""),
                           oci_key=session.get("oci_key", ""),
                           oci_endpoint=session.get("oci_endpoint", ""),
                           confluence_enabled=session.get("confluence_enabled", False),
                           confluence_key=session.get("confluence_key", ""),
                           jira_enabled=session.get("jira_enabled", False),
                           jira_key=session.get("jira_key", "")
                           )

@app.route("/documents")
def documents():
    with open(STORE_PATH, "rb") as f:
        store = pickle.load(f)
    docs = {}
    for m in store.get("metadatas", []):
        src = m["source"]
        if src not in docs:
            docs[src] = {"count": 0, "type": m.get("type", "file")}
        docs[src]["count"] += 1
    return render_template("documents.html", docs=docs)

@app.route("/delete/<path:doc>")
def delete_doc(doc):
    global index
    with open(STORE_PATH, "rb") as f:
        store = pickle.load(f)
    texts, metas = store.get("texts", []), store.get("metadatas", [])

    filtered = [(t, m) for t, m in zip(texts, metas)
                if m["link"] != doc and m["source"] != doc]

    if not filtered:
        index = None
        if os.path.exists(INDEX_PATH):
            os.remove(INDEX_PATH)
        with open(STORE_PATH, "wb") as f:
            pickle.dump({"texts": [], "metadatas": []}, f)
        flash(f"Deleted {doc}", "danger")
        return redirect(url_for("documents"))

    texts, metas = zip(*filtered)
    vecs = embed(list(texts))
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, INDEX_PATH)
    with open(STORE_PATH, "wb") as f:
        pickle.dump({"texts": list(texts), "metadatas": list(metas)}, f)
    flash(f"Deleted {doc}", "danger")
    return redirect(url_for("documents"))

@app.route("/refresh")
def refresh_doc():
    with open(STORE_PATH, "rb") as f:
        store = pickle.load(f)

    doc = request.args.get("doc")
    meta_entry = next((m for m in store["metadatas"] if m["source"] == doc), None)
    if not meta_entry:
        flash(f"No source found for {doc}", "warning")
        return redirect(url_for("documents"))

    if meta_entry["type"] == "file":
        flash(f"{doc} is a file. Re-upload to update.", "info")
        return redirect(url_for("documents"))

    if meta_entry["type"] == "url":
        build_index(url=meta_entry.get("link"))
    elif meta_entry["type"] == "confluence":
        build_index(confluence_id=meta_entry.get("link"))

    flash(f"Refreshed {doc}", "success")
    return redirect(url_for("documents"))

@app.route('/files/<path:filename>')
def serve_file(filename):
    return send_from_directory("uploads", filename)

@app.route("/delete_chat/<int:chat_id>", methods=["POST"])
def delete_chat(chat_id):
    global chat_history
    if 0 <= chat_id < len(chat_history):
        chat_history.pop(chat_id)
        save_chat_history(chat_history)
        flash("Chat deleted.", "warning")
    return redirect(url_for("home"))

@app.route("/delete_all_chats", methods=["POST"])
def delete_all_chats():
    global chat_history
    chat_history = []
    save_chat_history(chat_history)
    flash("All chats cleared.", "danger")
    return redirect(url_for("home"))

@app.route("/export_chat/<int:chat_id>", methods=["POST"])
def export_chat(chat_id):
    if 0 <= chat_id < len(chat_history):
        entry = chat_history[chat_id]
        content = f"Q: {entry['query']}\n\nA: {entry['answer']}\n\nSources: {entry.get('sources', [])}"
        return Response(content, mimetype="text/plain",
                        headers={"Content-Disposition": f"attachment;filename=chat_{chat_id}.txt"})
    flash("Chat not found.", "warning")
    return redirect(url_for("home"))

if __name__ == "__main__":
    print("Server running at http://127.0.0.1:5001")
    app.run(debug=True, host="0.0.0.0", port=5001)
