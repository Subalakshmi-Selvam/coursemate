# 🎓 CourseMate — AI Course Assistant (RAG Chatbot)

CourseMate is a Retrieval-Augmented Generation (RAG) chatbot that answers
student questions about a course — syllabus details, grading policy,
deadlines, assignment rules — by grounding every answer in the actual
course PDFs, not the model's general knowledge.

Ask it things like *"How much does the final exam count for?"* or
*"What's the late penalty on assignments?"* and it answers from your
uploaded documents, with sources shown.

---

## ✨ Features

- 🔍 **Grounded answers only** — retrieves relevant chunks from your course PDFs before answering; tells the student when something isn't in the materials instead of guessing
- 📤 **Upload PDFs at runtime** — add a new syllabus or handout from the sidebar and the index updates immediately, no restart needed
- 🔁 **Local or cloud LLM, your choice** — flip a single env var to switch between a fully offline **Ollama** model and the fast, free-tier **Groq** API
- 🧠 **Semantic search** via FAISS + sentence-transformer embeddings
- 💬 **Multi-turn chat** with conversation history
- 📄 **Source attribution** — every answer shows which document (and page) it came from, expandable in the UI
- 🐳 **Docker-ready** for one-command deployment
- ⚙️ **All tuning via `.env`** — no hunting through code to change chunk size, model, or retrieval depth

---

## 🛠 Stack

| Component | Tool |
|---|---|
| RAG framework | LangChain |
| Vector store | FAISS |
| Embeddings | Sentence-Transformers (`gte-small`) |
| LLM (local) | Ollama — any local model, default `gemma3:1b` |
| LLM (cloud) | Groq API — default `llama-3.1-8b-instant` |
| PDF parsing | `pypdf` |
| UI | Streamlit |
| Deployment | Docker / Docker Compose |

---

## 🗂 Project Structure

```
coursemate/
├── app.py                      # Streamlit chat UI (entry point)
├── ingest.py                   # CLI: build/rebuild the FAISS index from data/*.pdf
├── generate_sample_data.py     # Optional: creates demo course PDFs to try the app
├── config.py                   # All settings, read from environment variables
├── llm_provider.py             # Ollama / Groq provider switch
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example                # Copy to .env and fill in
├── .streamlit/config.toml      # UI theme
├── .github/workflows/ci.yml    # Basic lint + import check on push
├── data/                       # Course PDFs live here
└── faiss_index/                # Generated vector index (gitignored by default)
```

---

## 🚀 Quickstart

### 1. Clone and enter the project

```bash
git clone https://github.com/<your-username>/coursemate.git
cd coursemate
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
```

Open `.env` and choose your LLM provider:

**Option A — Local (Ollama), fully offline, no API key:**
```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=gemma3:1b
```
Then install [Ollama](https://ollama.com/download) and pull the model:
```bash
ollama pull gemma3:1b
ollama serve     # if it isn't already running in the background
```

**Option B — Cloud (Groq), fast, free tier, no local model download:**
```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.1-8b-instant
```
Get a free key at [console.groq.com/keys](https://console.groq.com/keys).

### 4. Add course documents

Either generate sample demo PDFs:
```bash
python generate_sample_data.py
```
or drop your own syllabus/notes/policy PDFs into `data/`.

### 5. Build the vector index

```bash
python ingest.py
```
Use `python ingest.py --rebuild` any time you want to regenerate the index from scratch (e.g. after changing `CHUNK_SIZE` in `.env`).

### 6. Run the app

```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser. You can also drag-and-drop new PDFs directly in the sidebar at any time — no need to re-run `ingest.py`.

---

## 🐳 Run with Docker

```bash
cp .env.example .env   # fill in GROQ_API_KEY if using groq
docker compose up -d
```
The app will be available at **http://localhost:8501**. The `data/` and `faiss_index/` folders are mounted as volumes so your documents and index persist across container restarts.

> To use local Ollama inside Docker instead of Groq, uncomment the `ollama` service in `docker-compose.yml` and set `OLLAMA_BASE_URL=http://ollama:11434` in `.env`.

---

## ☁️ Deploying

**Streamlit Community Cloud** (easiest, free):
1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo, set the main file to `app.py`.
3. In the app's "Secrets" settings, add:
   ```toml
   LLM_PROVIDER = "groq"
   GROQ_API_KEY = "your_key_here"
   GROQ_MODEL = "llama-3.1-8b-instant"
   ```
   (Streamlit Cloud has no Ollama runtime, so use the `groq` provider for cloud deploys.)
4. Commit a pre-built `faiss_index/` (or have users upload PDFs on first run via the sidebar).

**Any VM / cloud server (Docker):**
```bash
git clone https://github.com/<your-username>/coursemate.git
cd coursemate
cp .env.example .env   # configure provider + key
docker compose up -d
```
Put a reverse proxy (Caddy/Nginx) in front for HTTPS and a custom domain.

---

## ⚙️ Tunable Settings (`.env`)

| Variable | Default | Effect |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` (local) or `groq` (cloud) |
| `OLLAMA_MODEL` | `gemma3:1b` | Local model name |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model name |
| `LLM_TEMPERATURE` | `0.1` | Lower = more factual/deterministic |
| `EMBEDDING_MODEL` | `thenlper/gte-small` | Sentence-transformer model for embeddings |
| `CHUNK_SIZE` | `500` | Characters per text chunk |
| `CHUNK_OVERLAP` | `75` | Overlap between chunks |
| `RETRIEVAL_K` | `5` | Number of chunks retrieved per question |

---

## 🧠 How It Works

```
Student question
      ↓
FAISS retriever → top-k relevant chunks from indexed course PDFs
      ↓
Prompt template → injects context + grounding rules
      ↓
LLM (Ollama or Groq) → generates an answer using only that context
      ↓
Streamlit UI → answer + expandable source chunks
```

---

## 🔮 Possible Extensions

- Swap FAISS for a hosted vector DB (Pinecone, Qdrant, Weaviate) for multi-user deployments
- Add per-course namespaces so one deployment can serve several classes
- Add a feedback button (👍/👎) on answers to track retrieval quality
- Support `.docx` and lecture slide (`.pptx`) ingestion alongside PDFs

---

## 📄 License

MIT — see [LICENSE](LICENSE).
