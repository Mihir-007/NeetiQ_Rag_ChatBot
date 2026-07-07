"""
Retriever module.

Generates query embeddings and retrieves the
most relevant document chunks from Pinecone.
"""

from retrieval.config.settings import Settings

import builtins

def _trace(msg):
    builtins.print(msg, flush=True)

_trace("retriever.py: Importing Embedder")
from retrieval.embeddings.embedder import Embedder

_trace("retriever.py: Importing PineconeStore")
from retrieval.vectordb.pinecone_store import PineconeStore


class Retriever:
    """
    Orchestrates semantic search and hybrid search.
    """

    def __init__(self):
        _trace("Retriever.__init__: Instantiating Embedder")
        self.embedder = Embedder()

        _trace("Retriever.__init__: Instantiating PineconeStore")
        self.vector_store = PineconeStore()

        _trace("Retriever.__init__: Done")

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list:
        """
        Retrieve the most relevant chunks for a user query.
        """

        if top_k is None:
            top_k = Settings.TOP_K

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        print("Retriever: before encode", flush=True)

        query_embedding = self.embedder.encode(query)

        print("Retriever: encode finished", flush=True)

        print("Retriever: searching Pinecone", flush=True)

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        print("Retriever: Pinecone search finished", flush=True)
        print(f"Retriever: retrieved {len(results)} results", flush=True)

        return results