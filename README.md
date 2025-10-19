# RAG Chatbot

RAG Chatbot (Retrieval-Augmented Generation Chatbot) is an AI-powered question-answering application that ingests and indexes external knowledge from documents, URLs, and Confluence pages. It retrieves the most relevant context for any query and generates accurate, citation-backed answers using large language models.

## Features

- Knowledge ingestion: Upload PDF or text files, add URLs, or integrate Confluence pages as context.
- RAG pipeline: Automatically chunk, embed, and index content using FAISS for efficient retrieval.
- Intelligent Q&A: Query the knowledge base and receive context-aware answers with source references.
- Source citations: Each answer includes the source and a preview of the retrieved content.
- Multi-LLM support: Switch between providers such as OpenAI, Anthropic, Hugging Face, or OCI.
- Web interface: Clean UI built with Flask and Bootstrap for chat, document management, and settings.
- Chat history: Store, review, export, and manage previous conversations.
- Confluence integration: (Optional) Fetch pages and embed them into the knowledge base.
- URL ingestion: Add web links as data sources with graceful handling of restricted sites.

## Tech Stack

- Backend: Python, Flask, FAISS
- Embeddings and LLM: OpenAI, Hugging Face, Anthropic, OCI (configurable)
- Frontend: Jinja2, Bootstrap 5, Markdown
- Data storage: Pickle (prototype) or SQLite/Postgres (for production)
- Integrations: Confluence REST API, web page ingestion

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/rag-chatbot.git
cd rag-chatbot
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set environment variables
Create a `.env` file in the project root and add:
```
OPENAI_API_KEY=your_openai_api_key
CONFLUENCE_EMAIL=your_email
CONFLUENCE_API_TOKEN=your_confluence_token
CONFLUENCE_BASE=https://your-domain.atlassian.net/wiki
```

### 5. Run the application
```bash
python app.py
```
Access it in your browser at: `http://127.0.0.1:5001`

## Project Structure

```
rag-chatbot/
│
├── app.py                # Main Flask application
├── ingest.py             # Document ingestion and indexing logic
├── llm_provider.py       # LLM provider abstraction (OpenAI, Anthropic, etc.)
├── templates/            # HTML templates (Jinja2)
├── static/               # CSS/JS and frontend assets
├── storage/              # Indexed data and chat history
└── requirements.txt      # Python dependencies
```

## Usage

1. Go to the Upload page and add documents, URLs, or Confluence pages.
2. Once ingested, ask questions in the Chat interface.
3. View sources, copy answers, export chats, and manage conversations.
4. Use the Settings page to switch LLM providers or update API keys.

## Deployment

For production deployment, consider:
- Running the app in Docker or a cloud platform (Render, Railway, Azure App Service).
- Replacing pickle storage with SQLite or PostgreSQL for scalability.
- Securing API keys and credentials in a managed secrets store.

## Roadmap

- Enhanced Markdown rendering (tables, images, code blocks).
- Automatic refresh for URL and Confluence sources.
- Multi-user session support.
- Improved legal compliance (robots.txt and ToS checks).
- SaaS-ready version with billing and team access.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
