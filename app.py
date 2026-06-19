"""
CourseMate — Streamlit chat UI.

A RAG-based course assistant that answers student questions about
syllabi, lecture notes, assignment policies, and other course PDFs.

Run from the project root:
    streamlit run app.py
"""

from pathlib import Path

import streamlit as st
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

import config
from llm_provider import get_llm, ConfigError

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

NOT_GROUNDED_MARKER = "I don't see that in the course materials I have"

# ── Custom styling ──────────────────────────────────────────────────────
# Design direction: a "study notebook" feel for an academic tool — warm
# paper background, a serif voice for the assistant (like reading typed
# notes), and an index-card treatment for cited source chunks. The one
# signature element is the grounded/not-grounded pill on every answer,
# since that distinction (real answer vs "not in my materials") is the
# entire point of a RAG assistant.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --paper: #FBF7EF;
    --paper-raised: #FFFFFF;
    --ink: #2B2622;
    --ink-soft: #6B6258;
    --rule: #E4DCCB;
    --accent: #A8432A;
    --accent-soft: #F1E0D6;
    --grounded: #3F6B4F;
    --grounded-bg: #E7EFE6;
    --ungrounded: #9C7A1F;
    --ungrounded-bg: #F5EDD9;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: var(--paper); }

[data-testid="stHeader"] { background: transparent; }

/* ── Hero ───────────────────────────────────────────────────────── */
.cm-hero {
    display: flex;
    align-items: center;
    gap: 18px;
    padding: 28px 4px 20px 4px;
    border-bottom: 1px solid var(--rule);
    margin-bottom: 28px;
}
.cm-hero-badge {
    width: 52px; height: 52px;
    border-radius: 12px;
    background: var(--accent);
    color: #fff;
    display: flex; align-items: center; justify-content: center;
    font-size: 26px;
    flex-shrink: 0;
    box-shadow: 0 3px 0 rgba(0,0,0,0.12);
}
.cm-hero-title {
    font-family: 'Source Serif 4', serif;
    font-weight: 600;
    font-size: 2.1rem;
    color: var(--ink);
    line-height: 1.1;
    margin: 0;
}
.cm-hero-sub {
    color: var(--ink-soft);
    font-size: 0.95rem;
    margin-top: 4px;
}

/* ── Chat bubbles ───────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 4px 0 !important;
}
.cm-msg-user {
    background: var(--ink);
    color: var(--paper);
    border-radius: 14px 14px 4px 14px;
    padding: 12px 16px;
    max-width: 80%;
    margin-left: auto;
    font-size: 0.96rem;
}
.cm-msg-assistant-wrap { max-width: 88%; }
.cm-msg-assistant {
    background: var(--paper-raised);
    border: 1px solid var(--rule);
    border-radius: 4px 14px 14px 14px;
    padding: 16px 18px;
    font-family: 'Source Serif 4', serif;
    font-size: 1.04rem;
    line-height: 1.55;
    color: var(--ink);
}
.cm-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 999px;
    margin-bottom: 10px;
}
.cm-pill-grounded { background: var(--grounded-bg); color: var(--grounded); }
.cm-pill-ungrounded { background: var(--ungrounded-bg); color: var(--ungrounded); }

/* ── Source index cards ─────────────────────────────────────────── */
.cm-source-card {
    background: var(--paper-raised);
    border: 1px solid var(--rule);
    border-left: 3px solid var(--accent);
    border-radius: 2px 8px 8px 2px;
    padding: 10px 14px;
    margin-bottom: 8px;
}
.cm-source-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    color: var(--accent);
    font-weight: 500;
    margin-bottom: 4px;
}
.cm-source-text {
    font-size: 0.85rem;
    color: var(--ink-soft);
    line-height: 1.5;
}

/* ── Sidebar ────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--ink);
}
[data-testid="stSidebar"] * { color: #EFE9DF !important; }
[data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] small { color: #A89C8C !important; }
.cm-side-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #A89C8C !important;
    margin-bottom: 6px;
}
.cm-doc-chip {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(255,255,255,0.06);
    border-radius: 8px;
    padding: 7px 10px;
    margin-bottom: 6px;
    font-size: 0.83rem;
}
.cm-provider-card {
    background: rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 18px;
}
.cm-provider-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #6FCF97;
    margin-right: 6px;
}

/* ── Empty state ────────────────────────────────────────────────── */
.cm-empty {
    text-align: center;
    padding: 60px 20px;
    color: var(--ink-soft);
}
.cm-empty-icon { font-size: 2.5rem; margin-bottom: 12px; }

/* Buttons */
.stButton button {
    border-radius: 8px !important;
    font-weight: 500 !important;
}

/* Chat input */
[data-testid="stChatInput"] textarea {
    font-family: 'Inter', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="cm-hero">
    <div class="cm-hero-badge">{config.APP_ICON}</div>
    <div>
        <p class="cm-hero-title">{config.APP_TITLE}</p>
        <p class="cm-hero-sub">Ask about the syllabus, deadlines, grading, or course material — every answer is grounded in your course PDFs.</p>
    </div>
</div>
""", unsafe_allow_html=True)

SYSTEM_TEMPLATE = """
You are {assistant_name}, an AI assistant that helps students with questions
about their course. Use ONLY the information in CONTEXT to answer.

Rules:
1) Use ONLY the provided context below — do not use outside knowledge.
2) If the answer isn't in the context, say "I don't see that in the course materials I have — please check with your instructor."
3) Be concise and direct. Quote key phrases (deadlines, policies, numbers) exactly.
4) When useful, mention which document the info came from.

CONTEXT:
{{context}}

STUDENT QUESTION:
{{question}}
""".format(assistant_name=config.ASSISTANT_NAME)


# ── Resource loading (cached) ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformerEmbeddings(model_name=config.EMBEDDING_MODEL)


def index_exists() -> bool:
    return (config.INDEX_DIR / "index.faiss").exists()


@st.cache_resource(show_spinner=False)
def load_vectordb(_embedder):
    return FAISS.load_local(
        str(config.INDEX_DIR), _embedder, allow_dangerous_deserialization=True
    )


def build_chain(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": config.RETRIEVAL_K})
    llm = get_llm()
    prompt = PromptTemplate(input_variables=["context", "question"], template=SYSTEM_TEMPLATE)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        verbose=False,
    )


def ingest_uploaded_pdfs(uploaded_files, embedder, vectordb):
    """Add newly uploaded PDFs into the existing (or new) FAISS index at runtime."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    new_chunks = []
    for uploaded in uploaded_files:
        dest = config.DATA_DIR / uploaded.name
        dest.write_bytes(uploaded.getbuffer())

        loader = PyPDFLoader(str(dest))
        pages = loader.load()
        for page in pages:
            page.metadata["source"] = uploaded.name
        new_chunks.extend(splitter.split_documents(pages))

    if vectordb is None:
        vectordb = FAISS.from_documents(new_chunks, embedder)
    else:
        vectordb.add_documents(new_chunks)

    vectordb.save_local(str(config.INDEX_DIR))
    return vectordb, len(new_chunks)


# ── Sidebar: provider info + document upload ───────────────────────────────
with st.sidebar:
    st.markdown(f'<p style="font-family:\'Source Serif 4\',serif;font-size:1.3rem;font-weight:600;margin-bottom:18px;">{config.APP_ICON} {config.APP_TITLE}</p>', unsafe_allow_html=True)

    active_model = config.OLLAMA_MODEL if config.LLM_PROVIDER == "ollama" else config.GROQ_MODEL
    active_mode = "local · Ollama" if config.LLM_PROVIDER == "ollama" else "cloud · Groq"
    st.markdown(f"""
    <div class="cm-provider-card">
        <div class="cm-side-label">Active model</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;">
            <span class="cm-provider-dot"></span>{active_model}
        </div>
        <div style="font-size:0.78rem;color:#A89C8C;margin-top:2px;">{active_mode}</div>
    </div>
    """, unsafe_allow_html=True)

    existing = sorted(p.name for p in config.DATA_DIR.glob("*.pdf"))
    st.markdown(f'<div class="cm-side-label">Course documents · {len(existing)} indexed</div>', unsafe_allow_html=True)
    if existing:
        for name in existing:
            st.markdown(f'<div class="cm-doc-chip">📄 {name}</div>', unsafe_allow_html=True)
    else:
        st.caption("No documents indexed yet.")

    st.write("")
    st.markdown('<div class="cm-side-label">Add course material</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Add new course PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    add_clicked = st.button("＋ Add to knowledge base", disabled=not uploaded_files, use_container_width=True)

# ── Load embedder + index ──────────────────────────────────────────────────
embedder = load_embedder()
vectordb = load_vectordb(embedder) if index_exists() else None

if add_clicked and uploaded_files:
    with st.spinner("Reading PDFs and updating the index..."):
        vectordb, n_chunks = ingest_uploaded_pdfs(uploaded_files, embedder, vectordb)
    load_vectordb.clear()
    st.sidebar.success(f"Added {len(uploaded_files)} file(s) ({n_chunks} chunks). Reloading...")
    st.rerun()

if vectordb is None:
    st.markdown("""
    <div class="cm-empty">
        <div class="cm-empty-icon">📚</div>
        <p style="font-family:'Source Serif 4',serif;font-size:1.2rem;color:#2B2622;">No course documents yet</p>
        <p>Upload a syllabus or course PDF in the sidebar to get started,<br>or run <code>python ingest.py</code> from the project root.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

try:
    chain = build_chain(vectordb)
except ConfigError as e:
    st.error(str(e))
    st.stop()

def render_user_message(text: str):
    st.markdown(f'<div class="cm-msg-user">{text}</div>', unsafe_allow_html=True)


def render_assistant_message(answer: str, sources: list):
    is_grounded = NOT_GROUNDED_MARKER not in answer
    pill_class = "cm-pill-grounded" if is_grounded else "cm-pill-ungrounded"
    pill_icon = "✓" if is_grounded else "i"
    pill_text = "Grounded in course materials" if is_grounded else "Not in course materials"

    st.markdown(f"""
    <div class="cm-msg-assistant-wrap">
        <div class="cm-msg-assistant">
            <span class="cm-pill {pill_class}">{pill_icon} {pill_text}</span><br>
            {answer}
        </div>
    </div>
    """, unsafe_allow_html=True)

    if sources:
        with st.expander(f"📎 {len(sources)} source chunk(s) retrieved"):
            for doc in sources:
                src = doc.metadata.get("source", "unknown")
                page = doc.metadata.get("page")
                label = src + (f" · page {page + 1}" if page is not None else "")
                st.markdown(f"""
                <div class="cm-source-card">
                    <div class="cm-source-label">{label}</div>
                    <div class="cm-source-text">{doc.page_content}</div>
                </div>
                """, unsafe_allow_html=True)


# ── Session state ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Render existing messages ────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else config.APP_ICON):
        if msg["role"] == "user":
            render_user_message(msg["content"])
        else:
            render_assistant_message(msg["content"], msg.get("sources", []))

# ── Chat input ───────────────────────────────────────────────────────────
if user_input := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑‍🎓"):
        render_user_message(user_input)

    with st.chat_message("assistant", avatar=config.APP_ICON):
        with st.spinner("Looking through your course materials..."):
            try:
                result = chain.invoke({
                    "question": user_input,
                    "chat_history": st.session_state.chat_history,
                })
                answer = result["answer"]
                sources = result.get("source_documents", [])
            except Exception as e:
                answer = f"Something went wrong talking to the LLM: {e}"
                sources = []

        render_assistant_message(answer, sources)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
    st.session_state.chat_history.append((user_input, answer))
