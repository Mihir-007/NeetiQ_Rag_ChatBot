"""
Embedding model loader.

Loads the SentenceTransformer model only once
and reuses it throughout the application.
"""

from typing import Optional, Any
import logging
import os
import requests

from retrieval.config.settings import Settings

logging.basicConfig(level=logging.INFO)


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

                # -----------------------------
                # Torch Diagnostics
                # -----------------------------
                builtins.print("Importing torch...", flush=True)

                import torch

                builtins.print(
                    f"Torch version: {torch.__version__}",
                    flush=True
                )

                builtins.print(
                    f"CUDA available: {torch.cuda.is_available()}",
                    flush=True
                )

                builtins.print(
                    f"CUDA version: {torch.version.cuda}",
                    flush=True
                )

                builtins.print(
                    f"CUDA device count: {torch.cuda.device_count()}",
                    flush=True
                )

                # -----------------------------
                # Transformers Diagnostics
                # -----------------------------
                builtins.print(
                    "Importing transformers...",
                    flush=True
                )

                import transformers

                builtins.print(
                    f"Transformers version: {transformers.__version__}",
                    flush=True
                )

                # -----------------------------
                # Sentence Transformers Import
                # -----------------------------
                builtins.print(
                    "Importing sentence_transformers...",
                    flush=True
                )

                from sentence_transformers import SentenceTransformer

                builtins.print(
                    "SentenceTransformer imported",
                    flush=True
                )

                builtins.print(
                    "EmbeddingModel.get_model: SentenceTransformer imported",
                    flush=True
                )

                # -----------------------------
                # Hugging Face Diagnostics
                # -----------------------------
                from huggingface_hub.utils import logging as hf_logging

                hf_logging.set_verbosity_info()

                builtins.print(
                    f"HF_HOME = {os.getenv('HF_HOME')}",
                    flush=True
                )

                builtins.print(
                    f"TRANSFORMERS_CACHE = {os.getenv('TRANSFORMERS_CACHE')}",
                    flush=True
                )

                builtins.print(
                    "Checking connectivity to Hugging Face...",
                    flush=True
                )

                try:
                    response = requests.get(
                        "https://huggingface.co",
                        timeout=10,
                    )

                    builtins.print(
                        f"Hugging Face Status Code: {response.status_code}",
                        flush=True
                    )

                except Exception as e:
                    builtins.print(
                        f"Failed to reach Hugging Face: {e}",
                        flush=True
                    )

                # -----------------------------
                # Model Creation
                # -----------------------------
                builtins.print(
                    "EmbeddingModel.get_model: creating model",
                    flush=True
                )

                builtins.print(
                    "About to create SentenceTransformer",
                    flush=True
                )

                cls._model = SentenceTransformer(
                    Settings.EMBEDDING_MODEL,
                    device="cpu",
                    cache_folder="/tmp/huggingface",
                    trust_remote_code=False,
                )

                builtins.print(
                    "SentenceTransformer created",
                    flush=True
                )

                builtins.print(
                    "EmbeddingModel.get_model: model loaded",
                    flush=True
                )

                print("Embedding model loaded successfully.\n")

            except Exception as error:
                import traceback

                traceback.print_exc()

                raise RuntimeError(
                    f"Failed to load embedding model: {error}"
                ) from error

        builtins.print(
            "EmbeddingModel.get_model: returning model",
            flush=True
        )

        return cls._model