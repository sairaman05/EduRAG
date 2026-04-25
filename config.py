"""
═══════════════════════════════════════════════════════════════════════════════
 Education RAG System - Global Configuration & Shared Types
═══════════════════════════════════════════════════════════════════════════════
 This file defines the shared contract across all modules.
 ALL teammates must import types and configs from here.
═══════════════════════════════════════════════════════════════════════════════
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
import time
import uuid


@dataclass
class RAGConfig:
    """
    Master configuration for ablation experiments.
    Each flag toggles a module ON/OFF independently.
    """
    # ── Module Toggles ──
    use_mmr: bool = False
    use_citation: bool = False
    use_hallucination_detection: bool = False

    # ── Retrieval Settings ──
    top_k: int = 5
    embedding_dim: int = 384
    similarity_metric: str = "cosine"

    # ── MMR Settings ──
    mmr_lambda: float = 0.7
    mmr_top_n: int = 5

    # ── Hallucination Detection Settings ──
    hallucination_threshold: float = 0.5
    nli_model_name: str = "cross-encoder/nli-deberta-v3-small"
    claim_extraction_method: str = "sentence"

    # ── Citation Settings ──
    citation_min_similarity: float = 0.6

    # ── LLM Settings ──
    llm_model: str = "llama3"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 1024
    llm_provider: str = "ollama"

    # ── Evaluation ──
    log_metrics: bool = True
    random_seed: int = 42

    def get_variant_name(self) -> str:
        flags = []
        if self.use_mmr:
            flags.append("MMR")
        if self.use_citation:
            flags.append("Citation")
        if self.use_hallucination_detection:
            flags.append("HallucinationDetection")
        if not flags:
            return "Vanilla_RAG"
        if len(flags) == 3:
            return "Full_System"
        return "RAG+" + "+".join(flags)


@dataclass
class Document:
    doc_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = str(uuid.uuid4())[:8]


@dataclass
class RetrievedDocument:
    document: Document
    score: float
    rank: int
    mmr_score: Optional[float] = None
    diversity_contribution: Optional[float] = None
    citation_spans: Optional[List[Dict]] = None
    citation_id: Optional[str] = None


@dataclass
class Claim:
    claim_id: str
    text: str
    source_sentence: str
    is_supported: Optional[bool] = None
    support_score: Optional[float] = None
    supporting_doc_ids: List[str] = field(default_factory=list)
    evidence_text: Optional[str] = None


@dataclass
class RAGResponse:
    """
    Complete response object passed through the entire pipeline.
    Each module enriches this object — never replaces it.
    """
    query: str
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: float = field(default_factory=time.time)

    # ── Dense Retrieval output ──
    retrieved_docs: List[RetrievedDocument] = field(default_factory=list)

    # ── MMR output ──
    mmr_reranked_docs: Optional[List[RetrievedDocument]] = None

    # ── LLM Generation output ──
    raw_answer: str = ""
    final_answer: str = ""

    # ── Hallucination Detection output (only when module is active) ──
    claims: List[Claim] = field(default_factory=list)
    hallucination_flags: Dict[str, bool] = field(default_factory=dict)
    hallucination_stats: Dict[str, float] = field(default_factory=dict)

    # ── Citation output (only when module is active) ──
    citations: List[Dict] = field(default_factory=list)
    citation_stats: Dict[str, float] = field(default_factory=dict)

    # ═══════════════════════════════════════════════════════════════════════
    # Evaluation-Only Metrics — ALWAYS computed for uniform comparison
    # These do NOT modify the answer; they are read-only scoring passes
    # ═══════════════════════════════════════════════════════════════════════
    eval_claims: List[Claim] = field(default_factory=list)
    eval_hallucination_stats: Dict[str, float] = field(default_factory=dict)
    eval_citation_stats: Dict[str, float] = field(default_factory=dict)

    # ── Combined metrics ──
    metrics: Dict[str, float] = field(default_factory=dict)

    # ── Pipeline Metadata ──
    config_variant: str = ""
    module_timings: Dict[str, float] = field(default_factory=dict)

    def get_active_docs(self) -> List[RetrievedDocument]:
        if self.mmr_reranked_docs is not None:
            return self.mmr_reranked_docs
        return self.retrieved_docs

    def get_effective_hallucination_stats(self) -> Dict[str, float]:
        """
        Return hallucination stats from active module if it ran,
        otherwise from the evaluation-only pass. Always returns stats.
        """
        if self.hallucination_stats:
            return self.hallucination_stats
        return self.eval_hallucination_stats

    def get_effective_citation_stats(self) -> Dict[str, float]:
        """
        Return citation stats from active module if it ran,
        otherwise from the evaluation-only pass. Always returns stats.
        """
        if self.citation_stats:
            return self.citation_stats
        return self.eval_citation_stats

    def get_effective_claims(self) -> List:
        """
        Return claims from active module if it ran,
        otherwise from the evaluation-only pass.
        """
        if self.claims:
            return self.claims
        return self.eval_claims


ABLATION_VARIANTS = {
    "Vanilla RAG": RAGConfig(
        use_mmr=False, use_citation=False, use_hallucination_detection=False
    ),
    "RAG + MMR": RAGConfig(
        use_mmr=True, use_citation=False, use_hallucination_detection=False
    ),
    "RAG + Citation": RAGConfig(
        use_mmr=False, use_citation=True, use_hallucination_detection=False
    ),
    "RAG + Hallucination Detection": RAGConfig(
        use_mmr=False, use_citation=False, use_hallucination_detection=True
    ),
    "Full System": RAGConfig(
        use_mmr=True, use_citation=True, use_hallucination_detection=True
    ),
}