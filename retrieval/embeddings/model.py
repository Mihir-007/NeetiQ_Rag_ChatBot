"""
Embedding model loader.

Loads the SentenceTransformer model only once
and reuses it throughout the application.
"""

from typing import Optional, Any

from retrieval.config.settings import Settings


class EmbeddingModel:
    """Singleton loader for the embedding model."""

    _model: Optional[Any] = None

    @classmethod
    def get_model(cls):
        """
        Returns the shared embedding model instance.
        """
        import builtins

        builtins.print("EmbeddingModel.get_model: Called", flush=True)

        if cls._model is None:

            builtins.print(
                "EmbeddingModel.get_model: Model is None, initializing...",
                flush=True
            )

            print("=" * 60)
            print(f"Loading Embedding Model : {Settings.EMBEDDING_MODEL}")
            print(f"Embedding Dimension     : {Settings.EMBEDDING_DIMENSION}")
            print("=" * 60)

            try:
                builtins.print(
                    "EmbeddingModel.get_model: importing SentenceTransformer",
                    flush=True
                )

                builtins.print("Importing torch...", flush=True)
                import torch
                builtins.print(f"Torch version: {torch.__version__}", flush=True)

                builtins.print("Importing transformers...", flush=True)
                import transformers
                builtins.print(f"Transformers version: {transformers.__version__}", flush=True)

                builtins.print("Importing sentence_transformers...", flush=True)
                from sentence_transformers import SentenceTransformer
                builtins.print("SentenceTransformer imported", flush=True)

                builtins.print(
                    "EmbeddingModel.get_model: SentenceTransformer imported",
                    flush=True
                )

                builtins.print(
                    "EmbeddingModel.get_model: creating model",
                    flush=True
                )

                print("About to create SentenceTransformer", flush=True)
                cls._model = SentenceTransformer(
                    Settings.EMBEDDING_MODEL,
                    device="cpu",
                    cache_folder="/tmp/huggingface"
                    trust_remote_code=False,
                )
                print("SentenceTransformer created", flush=True)

                builtins.print(
                    "EmbeddingModel.get_model: model loaded",
                    flush=True
                )

                print("Embedding model loaded successfully.\n")

            except Exception as error:
                raise RuntimeError(
                    f"Failed to load embedding model: {error}"
                ) from error

        builtins.print(
            "EmbeddingModel.get_model: returning model",
            flush=True
        )

        return cls._model