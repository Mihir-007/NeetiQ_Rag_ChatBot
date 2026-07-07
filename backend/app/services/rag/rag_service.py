def run_pipeline():

    print("PIPELINE: started", flush=True)

    try:
        print("PIPELINE: getting retriever", flush=True)

        retriever = self.retrieval_service._get_retriever()

        print("PIPELINE: retriever ready", flush=True)

        print("PIPELINE: retrieving...", flush=True)

        results_dicts = retriever.retrieve(
            query,
            top_k=5
        )

        print("PIPELINE: retrieval complete", flush=True)

    except Exception as e:

        print("=" * 80, flush=True)
        print("RETRIEVAL FAILED", flush=True)
        print(str(e), flush=True)
        print("=" * 80, flush=True)

        results_dicts = []

    print("\n" + "=" * 80)
    print("Retrieved", len(results_dicts), "chunks")

    for i, r in enumerate(results_dicts, 1):
        print(f"\nChunk {i}")
        print("Chunk ID:", r.get("chunk_id"))
        print("Score:", r.get("score"))
        print("Metadata keys:", list((r.get("metadata") or {}).keys()))
        print("Page Content:")
        print(r.get("page_content", "")[:500])

    print("=" * 80)

    chunks_text = "\n".join(
        r.get("page_content", "")
        for r in results_dicts
    )

    citations = [
        CitationResponse(
            chunk_id=r.get("chunk_id"),
            score=r.get("score"),
            content=r.get("page_content"),
            document_id=(r.get("metadata") or {}).get("doc_id"),
        )
        for r in results_dicts
    ]

    if results_dicts:
        print("PIPELINE: Running Gemini WITH RAG context", flush=True)
    else:
        print("PIPELINE: Running Gemini WITHOUT RAG context", flush=True)

    prompt = build_prompt(
        question=query,
        context=chunks_text,
        history=history
    )

    print("\nPROMPT SENT TO GEMINI")
    print("=" * 80)
    print(prompt)
    print("=" * 80)

    try:
        answer = gemini_generate(prompt)

    except ValueError as e:
        raise HTTPException(
            status_code=503,
            detail=f"LLM Configuration Error: {str(e)}"
        )

    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Gemini unavailable: {str(e)}"
        )

    return answer, citations