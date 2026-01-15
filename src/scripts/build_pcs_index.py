# scripts/build_pcs_index.py
from src.procedures.vector_store import ensure_ingested
if __name__ == "__main__":
    n = ensure_ingested(force=True)   # rebuilds embeddings
    print(f"Ingested {n} PCS titles.")
