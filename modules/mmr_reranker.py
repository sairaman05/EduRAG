"""
═══════════════════════════════════════════════════════════════════════════════
 MMR Reranker Module
═══════════════════════════════════════════════════════════════════════════════
 Implements Maximal Marginal Relevance for diversity-aware reranking.

 MMR(d) = λ · sim(query, d) − (1 − λ) · max_{s ∈ S} sim(d, s)

 Balances relevance (query similarity) and diversity (doc-to-doc dissimilarity).
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity

from interfaces import BaseReranker
from config import RAGConfig, RetrievedDocument


class MMRReranker(BaseReranker):
    """
    Maximal Marginal Relevance reranker.
    Uses document embeddings already computed by the dense retriever
    to avoid redundant encoding.
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self.lambda_mult = config.mmr_lambda
        self._last_metrics: Dict[str, float] = {}

    def rerank(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_n: int = 5
    ) -> List[RetrievedDocument]:
        """
        Rerank retrieved documents using MMR.

        Uses embeddings already stored in each document (from dense retrieval)
        so we don't need to load the embedding model again.
        """
        if not documents:
            return []

        top_n = min(top_n, len(documents))

        # ── Get embeddings from documents (already computed by dense retriever) ──
        doc_embeddings = []
        for doc in documents:
            if doc.document.embedding is not None:
                doc_embeddings.append(doc.document.embedding)
            else:
                # Fallback: can't do MMR without embeddings
                self._last_metrics = {"mmr_error": 1.0}
                return documents[:top_n]

        doc_embeddings = np.array(doc_embeddings, dtype=np.float32)

        # ── Use dense retrieval scores as query-doc similarity ──
        # These are already cosine similarities computed by the retriever
        query_doc_sim = np.array([doc.score for doc in documents], dtype=np.float32)

        # ── Compute doc-doc similarity matrix ──
        doc_doc_sim = cosine_similarity(doc_embeddings)

        # ── MMR Greedy Selection ──
        selected_indices = []
        candidate_indices = list(range(len(documents)))

        # First document: highest query similarity
        first_idx = int(np.argmax(query_doc_sim))
        selected_indices.append(first_idx)
        candidate_indices.remove(first_idx)

        mmr_scores_all = {}
        mmr_scores_all[first_idx] = float(query_doc_sim[first_idx])

        while len(selected_indices) < top_n and candidate_indices:
            best_idx = -1
            best_mmr = -float("inf")

            for idx in candidate_indices:
                relevance = query_doc_sim[idx]
                # Max similarity to any already-selected document
                max_sim_to_selected = max(
                    doc_doc_sim[idx][s] for s in selected_indices
                )
                mmr_score = (
                    self.lambda_mult * relevance
                    - (1 - self.lambda_mult) * max_sim_to_selected
                )

                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = idx

            if best_idx >= 0:
                selected_indices.append(best_idx)
                candidate_indices.remove(best_idx)
                mmr_scores_all[best_idx] = best_mmr

        # ── Build result with populated fields ──
        reranked = []
        for new_rank, idx in enumerate(selected_indices):
            doc = documents[idx]
            doc.mmr_score = mmr_scores_all.get(idx, 0.0)
            doc.rank = new_rank + 1
            # Diversity = how different this doc is from others in the selected set
            if len(selected_indices) > 1:
                others = [s for s in selected_indices if s != idx]
                doc.diversity_contribution = 1.0 - float(
                    np.mean([doc_doc_sim[idx][s] for s in others])
                )
            else:
                doc.diversity_contribution = 1.0
            reranked.append(doc)

        # ── Compute metrics ──
        # Average pairwise distance among selected docs
        if len(selected_indices) > 1:
            pairwise_sims = []
            for i, idx_i in enumerate(selected_indices):
                for idx_j in selected_indices[i + 1:]:
                    pairwise_sims.append(doc_doc_sim[idx_i][idx_j])
            avg_pairwise_sim = float(np.mean(pairwise_sims))
            diversity_score = 1.0 - avg_pairwise_sim
        else:
            diversity_score = 1.0
            avg_pairwise_sim = 0.0

        self._last_metrics = {
            "diversity_score": diversity_score,
            "avg_pairwise_similarity": avg_pairwise_sim,
            "mmr_lambda": self.lambda_mult,
            "num_reranked": len(reranked),
        }

        return reranked

    def get_metrics(self) -> Dict[str, float]:
        """Return MMR-specific metrics for logging."""
        return self._last_metrics.copy()