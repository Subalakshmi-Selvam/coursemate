"""
CourseMate — document ingestion script.

Builds (or rebuilds) the FAISS vector index from every PDF in data/.
Run this once before starting the app, and again any time you add or
change course documents.

Usage:
    python ingest.py                  # index everything in data/
    python ingest.py --data ./mydocs  # index a different folder
    python ingest.py --rebuild        # force a clean rebuild (ignore cache)
"""

import argparse
import logging
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest")


def load_pdfs(data_dir: Path):
    """Load every PDF in data_dir, tagging each chunk with its source filename."""
    pdf_paths = sorted(data_dir.glob("*.pdf"))
    if not pdf_paths:
        log.error("No PDF files found in %s — add course PDFs there first.", data_dir)
        sys.exit(1)

    documents = []
    for pdf_path in pdf_paths:
        log.info("Loading %s", pdf_path.name)
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for page in pages:
            page.metadata["source"] = pdf_path.name
        documents.extend(pages)

    log.info("Loaded %d pages from %d PDF(s)", len(documents), len(pdf_paths))
    return documents


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    log.info(
        "Split into %d chunks (chunk_size=%d, overlap=%d)",
        len(chunks), config.CHUNK_SIZE, config.CHUNK_OVERLAP,
    )
    return chunks


def build_index(chunks, index_dir: Path):
    log.info("Loading embedding model: %s (first run downloads it, can take a minute)", config.EMBEDDING_MODEL)
    embedder = SentenceTransformerEmbeddings(model_name=config.EMBEDDING_MODEL)

    log.info("Embedding %d chunks and building FAISS index ...", len(chunks))
    vectordb = FAISS.from_documents(chunks, embedder)

    index_dir.mkdir(parents=True, exist_ok=True)
    vectordb.save_local(str(index_dir))
    log.info("Index saved to %s", index_dir)


def main():
    parser = argparse.ArgumentParser(description="Build the CourseMate FAISS index from PDFs.")
    parser.add_argument(
        "--data", type=str, default=str(config.DATA_DIR),
        help="Folder containing course PDFs (default: data/)",
    )
    parser.add_argument(
        "--index", type=str, default=str(config.INDEX_DIR),
        help="Output folder for the FAISS index (default: faiss_index/)",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Force a full rebuild even if an index already exists",
    )
    args = parser.parse_args()

    data_dir = Path(args.data)
    index_dir = Path(args.index)

    if not args.rebuild and (index_dir / "index.faiss").exists():
        log.info(
            "An index already exists at %s. Use --rebuild to regenerate it. Exiting.",
            index_dir,
        )
        return

    documents = load_pdfs(data_dir)
    chunks = chunk_documents(documents)
    build_index(chunks, index_dir)
    log.info("Done. You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
