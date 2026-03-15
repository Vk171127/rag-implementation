# src/modules/vector/vector_store.py

import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any
import os

# Why ChromaDB?
# - Runs 100% locally, no cloud needed
# - Stores documents + their embeddings in a local folder
# - Has a simple Python API
# - Same concepts as Vertex AI / Pinecone — just without the setup pain


class VectorStore:
    def __init__(self):
        self.client = None
        self.collection = None
        self.embedding_fn = None

    def initialize(self):
        """
        Set up ChromaDB with a local persistent folder.
        'persistent' means data survives restarts — stored in chroma_db/ folder.
        """

        # PersistentClient saves data to disk.
        # Every time the app restarts, the documents are still there.
        # path is relative to where you run uvicorn from (the src/ folder)
        self.client = chromadb.PersistentClient(path="../chroma_db")

        # This is the embedding model — the thing that converts
        # text → numbers. We use a small, fast local model.
        # First run will download it (~90MB), then it's cached.
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
            # Why this model? It's small (fast), free, runs locally,
            # and good enough for semantic search on short texts.
        )

        # A "collection" is like a table in SQL — a named group of documents.
        # get_or_create means: use it if it exists, create it if not.
        self.collection = self.client.get_or_create_collection(
            name="requirements_docs",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
            # cosine = measure similarity by angle between vectors,
            # not distance. Better for text.
        )

        print("✅ ChromaDB vector store ready")

    def add_documents(self, documents: List[str], metadatas: List[Dict], ids: List[str]):
        """
        Store documents in the vector store.
        ChromaDB automatically converts text → embeddings using our embedding_fn.

        documents  = list of text strings to store
        metadatas  = list of dicts with extra info (session_id, doc_type, etc.)
        ids        = unique string ID for each document
        """
        if not documents:
            return

        self.collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
            # upsert = insert if new, update if ID already exists.
            # Safer than add() which crashes on duplicate IDs.
        )

    def search(self, query: str, n_results: int = 5, where: Dict = None) -> List[Dict]:
        """
        Find the most semantically similar documents to a query string.

        query      = the text you're searching for
        n_results  = how many results to return
        where      = optional filter e.g. {"session_id": "abc123"}
        """
        query_params = {
            "query_texts": [query],  # ChromaDB embeds this automatically
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"]
            # distances = how far each result is from the query
            # lower distance = more similar
        }

        if where:
            query_params["where"] = where

        results = self.collection.query(**query_params)

        # Reformat ChromaDB's nested response into a clean list of dicts
        formatted = []
        for i in range(len(results["documents"][0])):
            formatted.append({
                "content":  results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score":    1 - results["distances"][0][i]
                # Convert distance → similarity score.
                # distance=0 means identical → score=1.0
                # distance=1 means opposite  → score=0.0
            })

        return formatted

    def delete_by_session(self, session_id: str):
        """Remove all documents belonging to a session."""
        self.collection.delete(
            where={"session_id": session_id}
        )


# Single shared instance
vector_store = VectorStore()