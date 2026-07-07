def retrieve(
    self,
    query: str,
    top_k: int = Settings.TOP_K,
) -> list:
    """
    Retrieve the most relevant chunks for a user query.

    Args:
        query (str): User query.
        top_k (int): Number of results to retrieve.

    Returns:
        list: Top-k retrieved document chunks.
    """

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