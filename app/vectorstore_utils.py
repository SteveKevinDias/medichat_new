import os
import hashlib
from typing import List
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

VECTOR_DB_PATH = "vector_db"
HASH_FILE = os.path.join(VECTOR_DB_PATH, "text_hash.txt")

# Load embeddings ONCE (important optimization)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)


def _compute_text_hash(texts: List[str]) -> str:
    return hashlib.md5("".join(texts).encode("utf-8")).hexdigest()


def create_faiss_index(texts: List[str]) -> FAISS:
    """
    - If FAISS index exists AND text unchanged -> load only
    - If new text detected -> recreate embeddings
    """

    os.makedirs(VECTOR_DB_PATH, exist_ok=True)
    new_hash = _compute_text_hash(texts)

    index_path = os.path.join(VECTOR_DB_PATH, "index.faiss")

    # 🔹 Case 1: FAISS already exists
    if os.path.exists(index_path):
        with open(HASH_FILE, "r") as f:
            old_hash = f.read().strip()

        vectorstore = FAISS.load_local(
            VECTOR_DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        # No change → retrieval only
        if new_hash == old_hash:
            st.success("✅ Existing embeddings reused")
            return vectorstore

        st.success( "🔄 New content detected → rebuilding embeddings")

    # 🔹 Case 2: First time OR new content
    vectorstore = FAISS.from_texts(texts, embeddings)

    vectorstore.save_local(VECTOR_DB_PATH)
    with open(HASH_FILE, "w") as f:
        f.write(new_hash)

    return vectorstore


def retrieve_similar_documents(
    vectorstore: FAISS,
    query: str,
    k: int = 4
    ):
    return vectorstore.similarity_search(query, k=k)