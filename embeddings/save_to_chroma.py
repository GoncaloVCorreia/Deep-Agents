import os
import pandas as pd
from typing import List
from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from dotenv import load_dotenv
import chromadb

# Load environment variables from .env file
load_dotenv()
# --- Configuration for LLM + Embeddings (adjust to your setup) ---
# Example using Google GenAI as you earlier indicated
Settings.llm = GoogleGenAI(model_name="gemini-2.5-flash", temperature=0.0, api_key=os.getenv("GOOGLE_GENAI_API_KEY"))
Settings.embed_model = GoogleGenAIEmbedding(model_name="gemini-embedding-001", api_key=os.getenv("GOOGLE_GENAI_API_KEY"))

# --- Load CSV and create documents ---
def load_tabular_csv(csv_path: str, num_docs: int = None) -> List[Document]:
    # Convert list-columns from strings to actual lists
    df = pd.read_csv(csv_path, converters={
        "inclusion_terms": eval,
        "includes": eval,
        "excludes1": eval,
        "excludes2": eval,
        "use_additional_code": eval,
        "code_first": eval,
        "notes": eval,
        "parent_codes": eval,
    })
    if num_docs:
        df = df.head(num_docs)
    
    docs: List[Document] = []
    for idx, row in df.iterrows():
        text = (
            f"{row['code']} — {row['description']}.  "
            f"Chapter {row['chapter']}: {row['chapter_desc']}.  "
            f"Section: {row['section_desc'] if pd.notna(row['section_desc']) and row['section_desc'] else ''}  "
            f"{('Also known as: ' + ', '.join(row['inclusion_terms']) + '. ') if row['inclusion_terms'] else ''}"
            f"{('Includes: ' + ', '.join(row['includes']) + '. ') if row['includes'] else ''}"
            f"{('Excludes: ' + ', '.join(row['excludes1']) + '. ') if row['excludes1'] else ''}"
            f"{('Excludes2: ' + ', '.join(row['excludes2']) + '. ') if row['excludes2'] else ''}"
            f"{('Requires additional code: ' + ', '.join(row['use_additional_code']) + '. ') if row['use_additional_code'] else ''}"
            f"{('Code must appear first: ' + ', '.join(row['code_first']) + '. ') if row['code_first'] else ''}"
            f"{('Parent code(s): ' + ', '.join(row['parent_codes']) + '. ') if row['parent_codes'] else ''}"
        )
        metadata = {
            "code": row['code'],
            "chapter": row['chapter'],
            "chapter_desc": row['chapter_desc'],
            "section": row['section'],
            "section_desc": row['section_desc'],
            "inclusion_terms": ", ".join(row['inclusion_terms']) if row['inclusion_terms'] else "",
            "parent_codes": ", ".join(row['parent_codes']) if row['parent_codes'] else "",
        }
        doc = Document(text=text, metadata=metadata)
        docs.append(doc)
    return docs

def build_and_save_index(docs: List[Document], collection_name: str = "icd10_tabular", persist_dir: str = "./chroma_store"):
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(docs, storage_context=storage_context, show_progress=True)

    # Persist index data to disk
    storage_context.persist(persist_dir=persist_dir)

    print(f"✅ Index built and stored in '{persist_dir}' with {len(docs)} documents")
    return index

def main():
    csv_path = r"C:\Users\Utilizador\Desktop\finalproj\data\icd10_tabular_extracted.csv" # update path as needed
    num_docs = None  # or an integer limit if you want to test smaller subset
    
    print("📥 Loading CSV…")
    docs = load_tabular_csv(csv_path, num_docs)
    print(f"📄 Loaded {len(docs)} documents.")
    
    print("🚀 Building index and saving into ChromaDB…")
    index = build_and_save_index(docs, collection_name="icd10_tabular")
    
    # Optionally: save index/storage context for future reload
    # storage_context.persist(...) if your version supports it
    print("✅ Done.")

if __name__ == "__main__":
    main()
