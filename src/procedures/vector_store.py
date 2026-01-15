import os
import pandas as pd
from typing import List
from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext
from chromadb.config import Settings as ChromaSettings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
import chromadb
import re

# ---------------- Env & Models (same as vector_store.py expects) ----------------
load_dotenv()

# Use the same LLM + embedding model via LlamaIndex Settings
Settings.llm = GoogleGenAI(
    model_name="gemini-2.5-flash",
    temperature=0.0,
    api_key=os.getenv("GOOGLE_GENAI_API_KEY"),
)
Settings.embed_model = GoogleGenAIEmbedding(
    model_name="gemini-embedding-001",
    api_key=os.getenv("GOOGLE_GENAI_API_KEY"),
)

PCS_COLLECTION = os.getenv("PCS_COLLECTION", "pcs_2026")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma")  # match vector_store.py default

_CLIENT: Optional[chromadb.PersistentClient] = None
_COLLECTION = None  # cache the actual collection object

_INDEX = None  # cache

def _get_index() -> VectorStoreIndex:
    """Open existing Chroma collection strictly; never (re)create or re-embed."""
    global _INDEX
    if _INDEX is not None:
        return _INDEX

    client = _get_client()
    collection = client.get_collection(PCS_COLLECTION)  # must already exist
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    _INDEX = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
    return _INDEX
# ---------------- Load PCS CSV and create documents ----------------
def load_procedures_csv(csv_path: str, num_docs: int = None) -> List[Document]:
    """
    Expects CSV columns: code,title,section,table,source
    """
    df = pd.read_csv(csv_path)

    required = {"code", "title", "section", "table", "source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    if num_docs:
        df = df.head(num_docs)

    docs: List[Document] = []
    for _, row in df.iterrows():
        code = str(row["code"]) if pd.notna(row["code"]) else ""
        title = str(row["title"]) if pd.notna(row["title"]) else ""
        section = str(row["section"]) if pd.notna(row["section"]) else ""
        table = str(row["table"]) if pd.notna(row["table"]) else ""
        source = str(row["source"]) if pd.notna(row["source"]) else ""

        # Text used for embeddings (title + light context)
        text = (
            f"{code} — {title}. "
            f"Section: {section}. "
            f"Table: {table}. "
            f"Source: {source}."
        )

        metadata = {
            "code": code,
            "title": title,
            "section": section,
            "table": table,
            "source": source,
        }

        docs.append(Document(text=text, metadata=metadata))

    return docs

# ---------------- Build & persist index ----------------
def build_and_save_index(
    docs: List[Document],
    collection_name: str = PCS_COLLECTION,
    persist_dir: str = CHROMA_DIR,
):
    # Create/open Chroma collection in the same location/env as vector_store.py
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = chroma_client.get_or_create_collection(collection_name)

    # IMPORTANT: pass the collection object to ChromaVectorStore
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        docs, storage_context=storage_context, show_progress=True
    )

    storage_context.persist(persist_dir=persist_dir)
    print(f"✅ Index built and stored in '{persist_dir}' with {len(docs)} documents (collection='{collection_name}')")
    return index
def _get_client() -> chromadb.PersistentClient:
    global _CLIENT
    if _CLIENT is None:
        chroma_path = Path(CHROMA_DIR)
        if not chroma_path.exists():
            raise FileNotFoundError(
                f"Chroma persist dir not found: {chroma_path}. "
                "Build the PCS index first."
            )
        _CLIENT = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=ChromaSettings(allow_reset=False),
        )
    return _CLIENT

def _get_collection():
    global _COLLECTION
    if _COLLECTION is None:
        client = _get_client()
        _COLLECTION = client.get_collection(PCS_COLLECTION)  # pass the name, get the object
    return _COLLECTION


def ensure_ingested(force: bool = False) -> int:
    """
    Compatibility shim: we DON'T ingest here; just return vector count.
    """
    col = _get_collection()
    try:
        return int(col.count())
    except Exception as e:
        raise RuntimeError(
            f"Unable to count vectors in collection '{PCS_COLLECTION}': {e}"
        )
    
def _get_query_embedding(text: str) -> List[float]:
    """
    Use the LlamaIndex embed model already set in Settings.embed_model.
    """
    embed_model: BaseEmbedding = Settings.embed_model  # type: ignore
    if embed_model is None:
        raise RuntimeError(
            "Settings.embed_model is not configured. "
            "Initialize LlamaIndex’s embedding model before calling query_note."
        )
    return embed_model.get_text_embedding(text)

def query_note(note: str, n_results: int = 8) -> Dict[str, Any]:
    """
    Retrieve via LlamaIndex retriever (same logic as icd10_query).
    Returns: {'results': [{code,title,section,table,score,document}, ...]}
    """
    index = _get_index()
    nodes = index.as_retriever(similarity_top_k=n_results).retrieve(note)

    out: List[Dict[str, Any]] = []
    for nw in nodes:
        meta = dict(nw.metadata or {})
        out.append({
            "code": meta.get("code"),
            "title": meta.get("title"),
            "section": meta.get("section"),
            "table": meta.get("table"),
            "score": getattr(nw, "score", None),        # higher is better
            "document": getattr(nw, "text", None) or nw.get_content() if hasattr(nw, "get_content") else None,
        })
    return {"results": out}
# ---------- Lightweight canonicalization & rewrites (generic) ----------
_STOP = {"a","an","and","the","of","in","on","with","without","w","for","to","by","at","from","or","via","into","than","as","per","using","use","none"}
_ALNUM = re.compile(r"[a-z0-9]+")

def _norm_tokens_basic(text: str):
    toks = _ALNUM.findall(text.lower())
    return [t for t in toks if len(t) > 1 and t not in _STOP]

_AND_PAIR = re.compile(r"\b([a-z0-9]+)\s+and\s+([a-z0-9]+)\b")
def _canon_and(text: str) -> str:
    def repl(m):
        a, b = m.group(1), m.group(2)
        return " ".join(sorted([a, b]))
    return _AND_PAIR.sub(repl, text.lower())

def _canon_of(text: str) -> str:
    s = " " + text.lower() + " "
    i = s.find(" of ")
    if i == -1:
        return text
    left, right = s[:i].strip(), s[i+4:].strip()
    if not left or not right:
        return text
    # head-first
    return (right + " " + left).strip()

def _canonicalize(text: str) -> str:
    return _canon_and(_canon_of(text.strip()))

def _anchors(tokens, k: int = 2, min_len: int = 7):
    long = [t for t in tokens if len(t) >= min_len]
    long.sort(key=len, reverse=True)
    seen, out = set(), []
    for t in long:
        if t not in seen:
            out.append(t); seen.add(t)
        if len(out) >= k:
            break
    return out

def _build_rewrites(note: str) -> list[str]:
    """Small, efficient set of rewrites (max 8)."""
    rewrites: list[str] = []
    raw = note.strip()
    can = _canonicalize(raw)
    toks = _norm_tokens_basic(can)

    if raw: rewrites.append(raw)
    if can and can != raw: rewrites.append(can)
    if toks: rewrites.append(" ".join(toks))

    if " of " in raw.lower():
        L, R = raw.lower().split(" of ", 1)
        L, R = L.strip(), R.strip()
        if L and R:
            rewrites.append(f"{R} {L}")

    anc = _anchors(toks, k=2, min_len=7)
    if anc:
        tail = [t for t in toks if t not in anc]
        rewrites.append(" ".join(anc + tail))

    # dedupe & cap
    seen, out = set(), []
    for r in rewrites:
        if r and r not in seen:
            out.append(r); seen.add(r)
    return out[:8]
def multi_query_note(note: str, n_per_query: int = 10) -> Dict[str, Any]:
    """
    RAG retriever: run several small queries, merge by code (keep max score and count hits).
    Returns: {'results': [{code,title,section,table,score,_hits,document}, ...]}
    """
    index = _get_index()
    rewrites = _build_rewrites(note)
    merged: Dict[str, Dict[str, Any]] = {}

    for rq in rewrites:
        nodes = index.as_retriever(similarity_top_k=n_per_query).retrieve(rq)
        for nw in nodes:
            meta = dict(nw.metadata or {})
            code = meta.get("code")
            if not code:
                continue
            score = float(getattr(nw, "score", 0.0) or 0.0)
            item = {
                "code": code,
                "title": meta.get("title"),
                "section": meta.get("section"),
                "table": meta.get("table"),
                "score": score,
                "document": getattr(nw, "text", None) or meta.get("title"),
                "_hits": 1,
            }
            if code not in merged or score > merged[code]["score"]:
                merged[code] = item
            else:
                merged[code]["_hits"] += 1

    results = sorted(merged.values(), key=lambda x: (x["score"], x["_hits"]), reverse=True)
    return {"results": results}

def main():
    # Point to your PCS CSV
    csv_path = r"C:\Users\Utilizador\Desktop\proj2\Deep-Agents\icd10pcs_generated_2026.csv"
    num_docs = None  # set an int to test on a subset

    print("📥 Loading PCS CSV…")
    docs = load_procedures_csv(csv_path, num_docs)
    print(f"📄 Loaded {len(docs)} procedure documents.")

    print("🚀 Building index and saving into ChromaDB…")
    build_and_save_index(
        docs,
        collection_name=PCS_COLLECTION,
        persist_dir=CHROMA_DIR,
    )
    print("✅ Done.")

if __name__ == "__main__":
    main()
