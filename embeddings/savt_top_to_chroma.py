# save_top_level_csv_to_chroma.py
import os
import ast
import json
import pandas as pd
from typing import List, Dict, Any
from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from dotenv import load_dotenv
import chromadb

# --- Env ---
load_dotenv()

# --- LLM / Embeddings ---
Settings.llm = GoogleGenAI(
    model_name="gemini-2.5-flash",
    temperature=0.0,
    api_key=os.getenv("GOOGLE_GENAI_API_KEY"),
)
Settings.embed_model = GoogleGenAIEmbedding(
    model_name="gemini-embedding-001",
    api_key=os.getenv("GOOGLE_GENAI_API_KEY"),
)

def _safe_list(x) -> list:
    return x if isinstance(x, list) else (list(x) if isinstance(x, tuple) else ([] if pd.isna(x) or x is None else [x]))

def _load_list_cell(cell):
    """Safely parse stringified Python lists from CSV into real lists."""
    if pd.isna(cell) or cell == "":
        return []
    if isinstance(cell, list):
        return cell
    if isinstance(cell, str):
        try:
            val = ast.literal_eval(cell)
            return val if isinstance(val, list) else [val]
        except Exception:
            # If it's a plain string, return as single-item list
            return [cell]
    return []

def _load_dict_cell(cell) -> Dict[str, Any]:
    if pd.isna(cell) or cell == "":
        return {}
    if isinstance(cell, dict):
        return cell
    if isinstance(cell, str):
        try:
            val = ast.literal_eval(cell)
            return val if isinstance(val, dict) else {}
        except Exception:
            return {}
    return {}

def load_tabular_csv_with_seven(csv_path: str, num_docs: int | None = None) -> List[Document]:
    df = pd.read_csv(csv_path)

    # Parse list-like and dict-like columns if they exist
    list_cols = [
        "inclusion_terms", "includes", "excludes1", "excludes2",
        "use_additional_code", "code_first", "notes", "parent_codes",
        "seven_char_note", "seven_char_def_notes",
    ]
    for col in list_cols:
        if col in df.columns:
            df[col] = df[col].apply(_load_list_cell)

    dict_cols = ["seven_char_def_extensions"]
    for col in dict_cols:
        if col in df.columns:
            df[col] = df[col].apply(_load_dict_cell)

    if num_docs:
        df = df.head(num_docs)

    docs: List[Document] = []
    for _, row in df.iterrows():
        # 7th char text assembly
        seven_note_txt = ""
        if "seven_char_note" in df.columns and row.get("seven_char_note"):
            seven_note_txt = " ".join(row["seven_char_note"])

        seven_def_map = row.get("seven_char_def_extensions") or {}
        seven_def_pairs = [f"{k}: {v}" for k, v in seven_def_map.items()]
        seven_def_txt = ", ".join(seven_def_pairs) if seven_def_pairs else ""

        seven_def_notes_txt = ""
        if "seven_char_def_notes" in df.columns and row.get("seven_char_def_notes"):
            seven_def_notes_txt = " ".join(row["seven_char_def_notes"])

        text = (
            f"{row['code']} — {row['description']}. "
            f"Chapter {row['chapter']}: {row['chapter_desc']}. "
            f"Section: {row['section_desc'] if pd.notna(row['section_desc']) and row['section_desc'] else ''} "
            f"{('Also known as: ' + ', '.join(row['inclusion_terms']) + '. ') if row.get('inclusion_terms') else ''}"
            f"{('Includes: ' + ', '.join(row['includes']) + '. ') if row.get('includes') else ''}"
            f"{('Excludes: ' + ', '.join(row['excludes1']) + '. ') if row.get('excludes1') else ''}"
            f"{('Excludes2: ' + ', '.join(row['excludes2']) + '. ') if row.get('excludes2') else ''}"
            f"{('Requires additional code: ' + ', '.join(row['use_additional_code']) + '. ') if row.get('use_additional_code') else ''}"
            f"{('Code must appear first: ' + ', '.join(row['code_first']) + '. ') if row.get('code_first') else ''}"
            f"{('Parent code(s): ' + ', '.join(row['parent_codes']) + '. ') if row.get('parent_codes') else ''}"
            f"{('7th-character note: ' + seven_note_txt + '. ') if seven_note_txt else ''}"
            f"{('7th-character defs: ' + seven_def_txt + '. ') if seven_def_txt else ''}"
            f"{('7th-character def notes: ' + seven_def_notes_txt + '. ') if seven_def_notes_txt else ''}"
        )

        metadata = {
            "code": row["code"],
            "chapter": row.get("chapter", ""),
            "chapter_desc": row.get("chapter_desc", ""),
            "section": row.get("section", ""),
            "section_desc": row.get("section_desc", ""),
            "inclusion_terms": ", ".join(row["inclusion_terms"]) if row.get("inclusion_terms") else "",
            "parent_codes": ", ".join(row["parent_codes"]) if row.get("parent_codes") else "",
            # Store raw 7th char info in metadata too (JSON string for dict)
            "seven_char_note": ", ".join(row.get("seven_char_note", [])),
            "seven_char_def_extensions": json.dumps(seven_def_map, ensure_ascii=False),
            "seven_char_def_notes": ", ".join(row.get("seven_char_def_notes", [])),
        }

        docs.append(Document(text=text, metadata=metadata))

    return docs

def build_and_save_index(
    docs: List[Document],
    collection_name: str = "icd10_tabular_top_level",
    persist_dir: str = "./chroma_store",
):
    # Use same DB directory — just a different collection name
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(docs, storage_context=storage_context, show_progress=True)
    storage_context.persist(persist_dir=persist_dir)
    print(f"✅ Index built into collection '{collection_name}' in '{persist_dir}' with {len(docs)} docs")
    return index

def main():
    # Point to your NEW CSV (the top-level-only one you generated)
    csv_path = r"C:\Users\Utilizador\Desktop\finalproj\data\icd10_tabular_top_level_only.csv"
    persist_dir = "./chroma_store"             # same directory as your existing DB
    collection_name = "icd10_tabular_top_level"  # NEW collection name

    print("📥 Loading CSV…")
    docs = load_tabular_csv_with_seven(csv_path)
    print(f"📄 Loaded {len(docs)} documents.")

    print("🚀 Building index and saving into ChromaDB…")
    build_and_save_index(docs, collection_name=collection_name, persist_dir=persist_dir)
    print("✅ Done.")

if __name__ == "__main__":
    main()
