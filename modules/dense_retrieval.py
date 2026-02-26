"""
═══════════════════════════════════════════════════════════════════════════════
 Module 1: Dense Retrieval (Base RAG)
═══════════════════════════════════════════════════════════════════════════════
 Mathematical Formulation:
   score(q, d) = cos(E(q), E(d)) = (E(q) · E(d)) / (‖E(q)‖ · ‖E(d)‖)

 Where E(·) is the embedding function (e.g., all-MiniLM-L6-v2).
 Retrieved set: R = top-k{d ∈ D : score(q, d)}

 Complexity: O(n·dim) for brute-force; O(log n) with FAISS/ANN index
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import time
import logging

from config import RAGConfig, Document, RetrievedDocument
from interfaces import BaseRetriever

logger = logging.getLogger(__name__)


class DenseRetriever(BaseRetriever):
    """
    Dense passage retrieval using sentence embeddings.
    Supports cosine similarity, dot product, and euclidean distance.
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self.documents: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None  # (n_docs, dim)
        self.embedding_model = None
        self._last_query_metrics: Dict[str, float] = {}
        self._is_indexed = False

    def _load_embedding_model(self):
        """Lazy-load the embedding model."""
        if self.embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("Loaded embedding model: all-MiniLM-L6-v2")
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. "
                    "Using pre-computed embeddings only."
                )

    def _compute_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for a single text."""
        self._load_embedding_model()
        if self.embedding_model is None:
            raise RuntimeError(
                "No embedding model available. Either install sentence-transformers "
                "or provide pre-computed embeddings."
            )
        return self.embedding_model.encode(text, normalize_embeddings=True)

    def _compute_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Compute embeddings for a batch of texts."""
        self._load_embedding_model()
        if self.embedding_model is None:
            raise RuntimeError("No embedding model available.")
        return self.embedding_model.encode(
            texts, normalize_embeddings=True, show_progress_bar=True, batch_size=32
        )

    def index(self, documents: List[Document]) -> None:
        """
        Index documents: compute or use pre-computed embeddings.

        Args:
            documents: List of Document objects. If doc.embedding is None,
                      embeddings will be computed automatically.
        """
        t_start = time.time()
        self.documents = documents

        # Check if embeddings are pre-computed
        has_embeddings = all(d.embedding is not None for d in documents)

        if has_embeddings:
            self.embeddings = np.array([d.embedding for d in documents], dtype=np.float32)
            # Normalize for cosine similarity
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            self.embeddings = self.embeddings / norms
            logger.info(f"Indexed {len(documents)} documents with pre-computed embeddings")
        else:
            texts = [d.content for d in documents]
            self.embeddings = self._compute_embeddings_batch(texts)
            # Store embeddings back into documents for downstream modules
            for i, doc in enumerate(self.documents):
                doc.embedding = self.embeddings[i].tolist()
            logger.info(f"Computed and indexed embeddings for {len(documents)} documents")

        self._is_indexed = True
        index_time = time.time() - t_start
        logger.info(f"Indexing completed in {index_time:.3f}s | Shape: {self.embeddings.shape}")

    def _compute_similarity(
        self, query_emb: np.ndarray, doc_embs: np.ndarray
    ) -> np.ndarray:
        """
        Compute similarity scores based on configured metric.

        Args:
            query_emb: (dim,) query embedding
            doc_embs: (n, dim) document embeddings

        Returns:
            (n,) similarity scores (higher = more similar)
        """
        if self.config.similarity_metric == "cosine":
            # Already normalized, so dot product = cosine similarity
            scores = doc_embs @ query_emb
        elif self.config.similarity_metric == "dot":
            scores = doc_embs @ query_emb
        elif self.config.similarity_metric == "euclidean":
            # Convert distance to similarity
            distances = np.linalg.norm(doc_embs - query_emb, axis=1)
            scores = 1.0 / (1.0 + distances)
        else:
            raise ValueError(f"Unknown metric: {self.config.similarity_metric}")
        return scores

    def retrieve(
        self, query: str, top_k: Optional[int] = None
    ) -> List[RetrievedDocument]:
        """
        Retrieve top-k most similar documents for a query.

        Args:
            query: Natural language query string
            top_k: Number of documents to retrieve (overrides config)

        Returns:
            List of RetrievedDocument sorted by descending similarity
        """
        if not self._is_indexed:
            raise RuntimeError("Must call index() before retrieve()")

        t_start = time.time()
        k = top_k or self.config.top_k

        # Compute query embedding
        query_emb = self._compute_embedding(query)

        # Compute similarities
        scores = self._compute_similarity(query_emb, self.embeddings)

        # Get top-k indices
        k = min(k, len(self.documents))
        top_indices = np.argsort(scores)[::-1][:k]

        # Build results
        results = []
        for rank, idx in enumerate(top_indices):
            results.append(RetrievedDocument(
                document=self.documents[idx],
                score=float(scores[idx]),
                rank=rank + 1,
            ))

        retrieval_time = time.time() - t_start

        # ── Log metrics for ablation ──
        all_scores = scores[top_indices]
        self._last_query_metrics = {
            "retrieval_time_ms": retrieval_time * 1000,
            "num_retrieved": len(results),
            "avg_similarity": float(np.mean(all_scores)),
            "max_similarity": float(np.max(all_scores)),
            "min_similarity": float(np.min(all_scores)),
            "score_std": float(np.std(all_scores)),
            "num_indexed": len(self.documents),
        }

        logger.info(
            f"Retrieved {k} docs in {retrieval_time*1000:.1f}ms | "
            f"Avg score: {self._last_query_metrics['avg_similarity']:.4f}"
        )

        return results

    def retrieve_with_embedding(
        self, query_embedding: np.ndarray, top_k: Optional[int] = None
    ) -> List[RetrievedDocument]:
        """Retrieve using a pre-computed query embedding (skips encoding)."""
        if not self._is_indexed:
            raise RuntimeError("Must call index() before retrieve()")

        k = top_k or self.config.top_k
        scores = self._compute_similarity(query_embedding, self.embeddings)
        k = min(k, len(self.documents))
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for rank, idx in enumerate(top_indices):
            results.append(RetrievedDocument(
                document=self.documents[idx],
                score=float(scores[idx]),
                rank=rank + 1,
            ))
        return results

    def get_metrics(self) -> Dict[str, float]:
        """Return metrics from the last retrieval call."""
        return self._last_query_metrics.copy()

    def get_all_embeddings(self) -> np.ndarray:
        """Return the full embedding matrix (needed by MMR teammate)."""
        if self.embeddings is None:
            raise RuntimeError("No embeddings indexed.")
        return self.embeddings.copy()

    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """Lookup a document by ID."""
        for doc in self.documents:
            if doc.doc_id == doc_id:
                return doc
        return None