"""Local 384-dimensional embedding wrapper (FastEmbed) with test injection support."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import lru_cache

from fastembed import TextEmbedding

from shared.config import Settings, get_settings


@lru_cache(maxsize=2)
def _load_fastembed_model(model_name: str) -> TextEmbedding:
    """Cache ONNX FastEmbed model instance across requests."""
    return TextEmbedding(model_name=model_name)


class LocalEmbedder:
    """Compact 384-dim open-weight embedder recording model name and dimensionality."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        dim: int = 384,
        embed_fn: Callable[[Sequence[str]], list[list[float]]] | None = None,
    ) -> None:
        self.model_name = model_name
        self.dim = dim
        self._embed_fn = embed_fn

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> LocalEmbedder:
        cfg = settings or get_settings()
        return cls(model_name=cfg.embedding_model, dim=cfg.embedding_dim)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of chunk texts into normalized float32 vectors."""
        if not texts:
            return []
        if self._embed_fn is not None:
            return self._embed_fn(texts)
        model = _load_fastembed_model(self.model_name)
        vectors = [vec.tolist() for vec in model.embed(list(texts))]
        if vectors and len(vectors[0]) != self.dim:
            raise ValueError(
                f"Expected embedding dim {self.dim}, got {len(vectors[0])} for {self.model_name}"
            )
        return vectors

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        # BGE small works symmetrically or with short query instruction; keep direct embedding
        vecs = self.embed_documents([query])
        return vecs[0]
