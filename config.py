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


# ─────────────────────────────────────────────────────────────────────────────
# System Configuration (Ablation Flags)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RAGConfig:
    """
    Master configuration for ablation experiments.
    Each flag toggles a module ON/OFF independently.

    Ablation Variants:
        Vanilla RAG:           use_mmr=False, use_citation=False, use_hallucination_detection=False
        RAG + MMR:             use_mmr=True,  use_citation=False, use_hallucination_detection=False
        RAG + Citation:        use_mmr=False, use_citation=True,  use_hallucination_detection=False
        RAG + Hallucination:   use_mmr=False, use_citation=False, use_hallucination_detection=True
        Full System:           use_mmr=True,  use_citation=True,  use_hallucination_detection=True
    """
    # ── Module Toggles ──
    use_mmr: bool = False
    use_citation: bool = False
    use_hallucination_detection: bool = False

    # ── Retrieval Settings ──
    top_k: int = 5
    embedding_dim: int = 384  # default for all-MiniLM-L6-v2
    similarity_metric: str = "cosine"  # "cosine" | "dot" | "euclidean"

    # ── MMR Settings (teammate's module) ──
    mmr_lambda: float = 0.7  # trade-off between relevance and diversity
    mmr_top_n: int = 5

    # ── Hallucination Detection Settings ──
    hallucination_threshold: float = 0.5  # claims below this are flagged
    nli_model_name: str = "cross-encoder/nli-deberta-v3-small"
    claim_extraction_method: str = "sentence"  # "sentence" | "clause"

    # ── Citation Settings (teammate's module) ──
    citation_min_similarity: float = 0.6

    # ── LLM Settings ──
    llm_model: str = "llama3"  # or any Ollama / API model
    llm_temperature: float = 0.3
    llm_max_tokens: int = 512
    llm_provider: str = "ollama"  # "ollama" | "openai" | "huggingface"

    # ── Evaluation ──
    log_metrics: bool = True
    random_seed: int = 42

    def get_variant_name(self) -> str:
        """Return human-readable ablation variant name."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Shared Data Types
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Document:
    """A single document/chunk in the knowledge base."""
    doc_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None  # populated by embedding module

    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = str(uuid.uuid4())[:8]


@dataclass
class RetrievedDocument:
    """A document returned by retrieval with relevance score."""
    document: Document
    score: float  # similarity score [0, 1]
    rank: int

    # ── Extended by MMR module (teammate fills these) ──
    mmr_score: Optional[float] = None
    diversity_contribution: Optional[float] = None

    # ── Extended by Citation module (teammate fills these) ──
    citation_spans: Optional[List[Dict]] = None  # [{start, end, text}]
    citation_id: Optional[str] = None


@dataclass
class Claim:
    """A single atomic claim extracted from the LLM answer."""
    claim_id: str
    text: str
    source_sentence: str  # original sentence it was extracted from

    # ── Hallucination detection results ──
    is_supported: Optional[bool] = None
    support_score: Optional[float] = None  # NLI entailment probability
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

    # ── MMR output (teammate) ──
    mmr_reranked_docs: Optional[List[RetrievedDocument]] = None

    # ── LLM Generation output ──
    raw_answer: str = ""
    final_answer: str = ""  # post-processed (after hallucination filtering)

    # ── Hallucination Detection output ──
    claims: List[Claim] = field(default_factory=list)
    hallucination_flags: Dict[str, bool] = field(default_factory=dict)
    hallucination_stats: Dict[str, float] = field(default_factory=dict)

    # ── Citation output (teammate) ──
    citations: List[Dict] = field(default_factory=list)
    citation_stats: Dict[str, float] = field(default_factory=dict)

    # ── Evaluation Metrics (logged per query) ──
    metrics: Dict[str, float] = field(default_factory=dict)

    # ── Pipeline Metadata ──
    config_variant: str = ""
    module_timings: Dict[str, float] = field(default_factory=dict)

    def get_active_docs(self) -> List[RetrievedDocument]:
        """Return MMR-reranked docs if available, else dense retrieval docs."""
        if self.mmr_reranked_docs is not None:
            return self.mmr_reranked_docs
        return self.retrieved_docs


# ─────────────────────────────────────────────────────────────────────────────
# Predefined Ablation Variants
# ─────────────────────────────────────────────────────────────────────────────
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