"""
═══════════════════════════════════════════════════════════════════════════════
 Module Interfaces (Abstract Base Classes)
═══════════════════════════════════════════════════════════════════════════════
 Every module (Dense Retrieval, MMR, Citation, Hallucination Detection)
 MUST implement its respective interface for plug-and-play compatibility.
═══════════════════════════════════════════════════════════════════════════════
"""

from abc import ABC, abstractmethod
from typing import List, Dict
from config import RAGConfig, Document, RetrievedDocument, RAGResponse, Claim


class BaseRetriever(ABC):
    """Interface for retrieval modules (Dense, MMR, Hybrid)."""

    @abstractmethod
    def index(self, documents: List[Document]) -> None:
        """Index documents into the retrieval store."""
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedDocument]:
        """Retrieve top-k documents for a query."""
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """Return retrieval-specific metrics for logging."""
        pass


class BaseReranker(ABC):
    """
    Interface for reranking modules (MMR, Cross-Encoder, etc.).
    TEAMMATE: Implement this for the MMR module.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_n: int = 5
    ) -> List[RetrievedDocument]:
        """Rerank retrieved documents. Must populate mmr_score fields."""
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """Return reranking-specific metrics (diversity scores, etc.)."""
        pass


class BaseHallucinationDetector(ABC):
    """Interface for hallucination detection module."""

    @abstractmethod
    def extract_claims(self, text: str) -> List[Claim]:
        """Extract atomic claims from generated text."""
        pass

    @abstractmethod
    def verify_claims(
        self,
        claims: List[Claim],
        evidence: List[RetrievedDocument]
    ) -> List[Claim]:
        """Verify each claim against evidence. Must populate is_supported, support_score."""
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """Return hallucination metrics (rate, faithfulness, etc.)."""
        pass


class BaseCitationGenerator(ABC):
    """
    Interface for citation grounding module.
    TEAMMATE: Implement this for the Citation module.
    """

    @abstractmethod
    def generate_citations(
        self,
        answer: str,
        claims: List[Claim],
        documents: List[RetrievedDocument]
    ) -> Dict:
        """
        Generate citations mapping answer segments to source documents.
        Must return: {
            "cited_answer": str,          # answer with inline citations
            "citations": List[Dict],       # [{claim_id, doc_id, span, confidence}]
            "citation_stats": Dict          # {coverage, accuracy, precision, recall}
        }
        """
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """Return citation metrics."""
        pass