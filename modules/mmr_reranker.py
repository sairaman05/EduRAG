"""
═══════════════════════════════════════════════════════════════════════════════
 MMR Reranker Module
═══════════════════════════════════════════════════════════════════════════════
 Implements Maximal Marginal Relevance for diversity-aware reranking.
 Balances relevance (query similarity) and diversity (doc-to-doc dissimilarity).
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class MMRReranker:
    """
    Maximal Marginal Relevance reranker.

    MMR(d) = λ * sim(query, d) − (1 − λ) * max(sim(d, selected_docs))
    """

    def __init__(self, config):
        self.lambda_mult = getattr(config, "mmr_lambda", 0.5)
        self.top_n = getattr(config, "mmr_top_n", 5)

        # Use same embedding model as retriever ideally
        self.model = SentenceTransformer(
            getattr(config, "embedding_model", "all-MiniLM-L6-v2")
        )

    def rerank(self, query: str, documents: List, top_n: int = None) -> List:
        """
        Rerank retrieved documents using MMR.

        Args:
            query: User query
            documents: Retrieved documents
            top_n: Number of documents to return

        Returns:
            Reranked list of documents
        """

        if not documents:
            return []

        top_n = top_n or self.top_n
        top_n = min(top_n, len(documents))

        # ───────────────────────────────────────────────
        # Step 1: Embed query and documents
        # ───────────────────────────────────────────────
        doc_texts = [doc.document.content for doc in documents]
        query_embedding = self.model.encode([query])
        doc_embeddings = self.model.encode(doc_texts)

        # Compute similarity(query, doc)
        query_doc_sim = cosine_similarity(query_embedding, doc_embeddings)[0]

        # Compute similarity(doc_i, doc_j)
        doc_doc_sim = cosine_similarity(doc_embeddings)

        # ───────────────────────────────────────────────
        # Step 2: MMR Selection
        # ───────────────────────────────────────────────
        selected_indices = []
        candidate_indices = list(range(len(documents)))

        # Select first document (highest query similarity)
        first_idx = int(np.argmax(query_doc_sim))
        selected_indices.append(first_idx)
        candidate_indices.remove(first_idx)

        while len(selected_indices) < top_n and candidate_indices:
            mmr_scores = []

            for idx in candidate_indices:
                relevance = query_doc_sim[idx]

                diversity = max(
                    doc_doc_sim[idx][selected] for selected in selected_indices
                )

                mmr_score = (
                    self.lambda_mult * relevance
                    - (1 - self.lambda_mult) * diversity
                )

                mmr_scores.append((idx, mmr_score))

            # Select doc with highest MMR score
            next_selected = max(mmr_scores, key=lambda x: x[1])[0]

            selected_indices.append(next_selected)
            candidate_indices.remove(next_selected)

        # Return reranked docs
        return [documents[i] for i in selected_indices]