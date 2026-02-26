"""
═══════════════════════════════════════════════════════════════════════════════
 Placeholder Stubs for Teammate Modules
═══════════════════════════════════════════════════════════════════════════════
 These are NO-OP implementations that pass data through unchanged.
 Teammates replace these files with their actual implementations.

 The system runs end-to-end even without real implementations.
═══════════════════════════════════════════════════════════════════════════════
"""

from typing import List, Dict
from config import RetrievedDocument, Claim
from interfaces import BaseReranker, BaseCitationGenerator


class MMRRerankerStub(BaseReranker):
    """
    STUB: Replace with actual MMR implementation.
    File: modules/mmr_reranker.py

    Expected behavior:
        - Accept retrieved docs + query
        - Apply MMR diversification
        - Return reranked docs with mmr_score populated
        - Provide diversity metrics
    """

    def rerank(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_n: int = 5
    ) -> List[RetrievedDocument]:
        # Pass-through: return documents unchanged
        return documents[:top_n]

    def get_metrics(self) -> Dict[str, float]:
        return {"mmr_stub": True, "diversity_score": 0.0}


class CitationGeneratorStub(BaseCitationGenerator):
    """
    STUB: Replace with actual Citation implementation.
    File: modules/citation_generator.py

    Expected behavior:
        - Accept answer text, claims, and source documents
        - Map each claim to supporting document(s)
        - Return answer with inline citations [1], [2], etc.
        - Provide citation coverage/accuracy metrics
    """

    def generate_citations(
        self,
        answer: str,
        claims: List[Claim],
        documents: List[RetrievedDocument]
    ) -> Dict:
        return {
            "cited_answer": answer,
            "citations": [],
            "citation_stats": {
                "citation_stub": True,
                "coverage": 0.0,
                "accuracy": 0.0,
            },
        }

    def get_metrics(self) -> Dict[str, float]:
        return {"citation_stub": True}