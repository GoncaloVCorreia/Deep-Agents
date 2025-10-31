import os
from typing import List
import pandas as pd

from llama_index.core import VectorStoreIndex, Settings, StorageContext, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from chromadb.config import Settings as ChromaSettings

from llama_index.llms.google_genai import GoogleGenAI  # for answer generation
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding  # still used for embeddings
from dotenv import load_dotenv
load_dotenv()
# Set up the LLM
Settings.llm = GoogleGenAI(model_name="gemini-2.5-flash",
                           temperature=0.0,
                           api_key=os.getenv("GOOGLE_GENAI_API_KEY"))
Settings.embed_model = GoogleGenAIEmbedding(model_name="gemini-embedding-001", api_key=os.getenv("GOOGLE_GENAI_API_KEY"))

def load_index(persist_dir: str, collection_name: str = "icd10_tabular") -> VectorStoreIndex:
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store,
                                               storage_context=storage_context)
    return index

def query_and_generate_answer(index: VectorStoreIndex, user_query: str, top_k: int = 5) -> str:
    # Get query engine and retrieve top documents
    query_engine = index.as_query_engine(similarity_top_k=top_k)
    response = query_engine.query(user_query)
    
    # The response object will include retrieved nodes + the LLM answer.
    # We can print or process the nodes and the answer text.
    print("🔍 Retrieved context:")
    for node in response.source_nodes:
        print(f"- Code: {node.metadata.get('code')} | Description: {node.metadata.get('chapter_desc')}")
    
    print("\n🧠 Generated answer:")
    print(response.response)  # the actual answer from Google GenAI model
    
    return response.response

def main():
    persist_dir = "./chroma_store"
    collection_name = "icd10_tabular"
    user_query = "Early congenital syphilitic pneumonia"

    print("📂 Loading index…")
    index = load_index(persist_dir, collection_name)
    print("🚀 Querying and generating answer…")
    answer = query_and_generate_answer(index, user_query, top_k=5)
    print("\n✅ Done.")

if __name__ == "__main__":
    main()
