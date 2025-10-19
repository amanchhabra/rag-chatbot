import os, re, pickle, faiss, numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from flask import flash, session
import requests
from requests.auth import HTTPBasicAuth
from bs4 import BeautifulSoup
from llm_provider import LLMProvider


# ===============================
# CONFIG
# ===============================
CONFLUENCE_EMAIL = os.getenv("CONFLUENCE_EMAIL")
CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")
CONFLUENCE_BASE = os.getenv("CONFLUENCE_BASE")

INDEX_PATH = "storage/index.faiss"
STORE_PATH = "storage/store.pkl"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

load_dotenv()

# ===============================
# HELPERS
# ===============================
def fetch_url_text(url: str):
    """Download a web page or PDF from a URL and return text,
    or show a failure toast if restricted."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        if url.lower().endswith(".pdf"):
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            path = "temp_url.pdf"
            with open(path, "wb") as f:
                f.write(resp.content)
            return read_pdf(path)
        else:
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            return [{"source": url, "page": 1, "text": text}]
    except Exception as e:
        # Show failure toast
        flash(f"❌ Fetching data from this website is restricted: {url}", "danger")
        print(f"⚠️ URL fetch failed for {url}: {e}")
        return []


def fetch_confluence_page(page_id: str):
    """Fetch Confluence page content via REST API"""
    url = f"{CONFLUENCE_BASE}/rest/api/content/{page_id}?expand=body.storage"
    resp = requests.get(
        url,
        auth=HTTPBasicAuth(CONFLUENCE_EMAIL, CONFLUENCE_API_TOKEN),
        headers={"Accept": "application/json"}
    )
    resp.raise_for_status()
    data = resp.json()
    title = data["title"]
    html = data["body"]["storage"]["value"]
    text = re.sub(r"<[^>]+>", " ", html)  # strip HTML tags
    return [{"source": f"Confluence: {title}", "page": 1, "text": text}]


def get_llm():
    llm = LLMProvider(provider="openai")
    if not llm:
        flash("❌ No API key found. Please set it in Settings.", "danger")
        return None
    return llm;


def read_pdf(path):
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()
        pages.append({"source": os.path.basename(path), "page": i, "text": text})
    return pages


def read_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    text = re.sub(r"\s+", " ", text).strip()
    return [{"source": os.path.basename(path), "page": 1, "text": text}]


def chunk_text(text, size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def embed(texts):
    client = get_llm()
    if not client:
        return np.array([])
    arr = np.array(client.embed(texts), dtype="float32")
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    return arr


def build_index(file_path=None, url=None, confluence_id=None):
    # Load existing index + metadata if they exist
    if os.path.exists(INDEX_PATH) and os.path.exists(STORE_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(STORE_PATH, "rb") as f:
            store = pickle.load(f)
        texts, metas = store["texts"], store["metadatas"]
    else:
        index = None
        texts, metas = [], []

    if confluence_id:
        pages = fetch_confluence_page(confluence_id)
        link = f"https://your-confluence-site.atlassian.net/wiki/{confluence_id}"
        type = "confluence"
    elif url:
        pages = fetch_url_text(url)
        link = url
        type = "url"
    elif file_path and file_path.lower().endswith(".pdf"):
        pages = read_pdf(file_path)
        source_name = os.path.basename(file_path)
        link = f"/uploads/{source_name}"
        type = "file"
    elif file_path and file_path.lower().endswith(".txt"):
        pages = read_txt(file_path)
        source_name = os.path.basename(file_path)
        link = f"/uploads/{source_name}"
        type = "file"
    else:
        raise ValueError("Provide either file_path, url, or confluence_id")

    if not pages:
        flash("No content added from this source.", "warning")
        print(f"Skipped ingestion: {url or file_path or confluence_id}")
        return False

    records = []
    for page in pages:
        chunks = chunk_text(page["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        for ci, ch in enumerate(chunks):
            records.append({
                "text": ch,
                "meta": {
                    "source": page["source"],
                    "page": page["page"],
                    "chunk_id": ci,
                    "link": link,
                    "type": type
                }
            })

    new_texts = [r["text"] for r in records]
    new_metas = [r["meta"] for r in records]

    if not new_texts:
        flash("⚠️ No text extracted from the document.", "warning")
        print(f"⚠️ No chunks created from {url or file_path or confluence_id}")
        return False

    # Embed the new chunks
    vecs = embed(new_texts)
    if vecs.size == 0:
        return False

    # Create or extend FAISS index
    if index is None:
        index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    # Extend existing metadata
    texts.extend(new_texts)
    metas.extend(new_metas)

    # Save everything back
    os.makedirs("storage", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(STORE_PATH, "wb") as f:
        pickle.dump({"texts": texts, "metadatas": metas}, f)

    print(f"✅ Added {len(new_texts)} chunks from {url or file_path or confluence_id}")
    print(f"📦 Total chunks in index: {len(texts)}")
    return True


if __name__ == "__main__":
    import sys
    build_index(sys.argv[1])
