"""
═══════════════════════════════════════════════════════════════════════════════
 RAG Pipeline Orchestrator
═══════════════════════════════════════════════════════════════════════════════
 Orchestrates the full pipeline: Retrieve → [MMR] → Generate → [Hallucination] → [Citation]
 Each module is conditionally invoked based on RAGConfig flags.
═══════════════════════════════════════════════════════════════════════════════
"""

import time
import logging
from typing import List, Optional

from config import RAGConfig, Document, RAGResponse
from modules.dense_retrieval import DenseRetriever
from modules.hallucination_detector import HallucinationDetector
from modules.llm_generator import LLMGenerator
from modules.stubs import MMRRerankerStub, CitationGeneratorStub
from utils.metrics import MetricsLogger, QueryLog

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Modular RAG pipeline with ablation support.

    Usage:
        config = RAGConfig(use_hallucination_detection=True)
        pipeline = RAGPipeline(config)
        pipeline.index_documents(documents)
        response = pipeline.query("What is photosynthesis?")
    """

    def __init__(self, config: RAGConfig, metrics_logger: Optional[MetricsLogger] = None):
        self.config = config
        self.metrics_logger = metrics_logger or MetricsLogger()

        # ── Initialize core modules ──
        self.retriever = DenseRetriever(config)
        self.generator = LLMGenerator(config)

        # ── Initialize optional modules (stubs if not replaced) ──
        self.hallucination_detector = HallucinationDetector(config) if config.use_hallucination_detection else None

        # Teammates replace these with real implementations:
        self.mmr_reranker = None
        self.citation_generator = None

        self._try_load_teammate_modules()

    def _try_load_teammate_modules(self):
        """
        Attempt to load real teammate implementations.
        Falls back to stubs if not available.
        """
        # ── MMR Module ──
        if self.config.use_mmr:
            try:
                from modules.mmr_reranker import MMRReranker
                self.mmr_reranker = MMRReranker(self.config)
                logger.info("Loaded real MMR module")
            except ImportError:
                logger.info("MMR module not found, using stub")
                self.mmr_reranker = MMRRerankerStub()

        # ── Citation Module ──
        if self.config.use_citation:
            try:
                from modules.citation_generator import CitationGenerator
                self.citation_generator = CitationGenerator(self.config)
                logger.info("Loaded real Citation module")
            except ImportError:
                logger.info("Citation module not found, using stub")
                self.citation_generator = CitationGeneratorStub()

    def index_documents(self, documents: List[Document]) -> None:
        """Index documents into the retrieval store."""
        self.retriever.index(documents)

    def query(
        self,
        question: str,
        ground_truth: Optional[str] = None,
        relevant_doc_ids: Optional[List[str]] = None,
    ) -> RAGResponse:
        """
        Run the full RAG pipeline for a single query.

        Args:
            question: User's question
            ground_truth: Optional reference answer for evaluation
            relevant_doc_ids: Optional ground truth relevant doc IDs

        Returns:
            RAGResponse with all module outputs and metrics
        """
        t_total_start = time.time()
        response = RAGResponse(query=question, config_variant=self.config.get_variant_name())

        # ═══════════════════════════════════════════════════════
        # Stage 1: Dense Retrieval
        # ═══════════════════════════════════════════════════════
        t_start = time.time()
        response.retrieved_docs = self.retriever.retrieve(question, self.config.top_k)
        response.module_timings["retrieval"] = (time.time() - t_start) * 1000

        # ═══════════════════════════════════════════════════════
        # Stage 2: MMR Reranking (if enabled)
        # ═══════════════════════════════════════════════════════
        if self.config.use_mmr and self.mmr_reranker:
            t_start = time.time()
            response.mmr_reranked_docs = self.mmr_reranker.rerank(
                question, response.retrieved_docs, self.config.mmr_top_n
            )
            response.module_timings["mmr"] = (time.time() - t_start) * 1000

        # ═══════════════════════════════════════════════════════
        # Stage 3: LLM Answer Generation
        # ═══════════════════════════════════════════════════════
        active_docs = response.get_active_docs()
        t_start = time.time()
        response.raw_answer = self.generator.generate(question, active_docs)
        response.final_answer = response.raw_answer
        response.module_timings["generation"] = (time.time() - t_start) * 1000

        # ═══════════════════════════════════════════════════════
        # Stage 4: Hallucination Detection (if enabled)
        # ═══════════════════════════════════════════════════════
        if self.config.use_hallucination_detection and self.hallucination_detector:
            t_start = time.time()

            # Extract claims
            response.claims = self.hallucination_detector.extract_claims(response.raw_answer)

            # Verify claims against evidence
            response.claims = self.hallucination_detector.verify_claims(
                response.claims, active_docs
            )

            # Filter hallucinated content
            response.final_answer, removed = self.hallucination_detector.filter_hallucinated_claims(
                response.raw_answer, response.claims
            )

            # Record flags
            response.hallucination_flags = {
                c.claim_id: (not c.is_supported) for c in response.claims if c.is_supported is not None
            }
            response.hallucination_stats = self.hallucination_detector.get_metrics()
            response.module_timings["hallucination"] = (time.time() - t_start) * 1000

        # ═══════════════════════════════════════════════════════
        # Stage 5: Citation Generation (if enabled)
        # ═══════════════════════════════════════════════════════
        if self.config.use_citation and self.citation_generator:
            t_start = time.time()
            cit_result = self.citation_generator.generate_citations(
                response.final_answer, response.claims, active_docs
            )
            response.final_answer = cit_result.get("cited_answer", response.final_answer)
            response.citations = cit_result.get("citations", [])
            response.citation_stats = cit_result.get("citation_stats", {})
            response.module_timings["citation"] = (time.time() - t_start) * 1000

        # ═══════════════════════════════════════════════════════
        # Collect all metrics
        # ═══════════════════════════════════════════════════════
        total_time = (time.time() - t_total_start) * 1000
        response.module_timings["total"] = total_time

        # ── Merge all metrics into response.metrics ──
        response.metrics.update(self.retriever.get_metrics())
        response.metrics.update(self.generator.get_metrics())
        if self.hallucination_detector:
            response.metrics.update({
                f"hall_{k}": v for k, v in self.hallucination_detector.get_metrics().items()
            })

        # ── Log to metrics logger ──
        if self.config.log_metrics:
            query_log = QueryLog(
                run_id=response.run_id,
                query=question,
                variant=response.config_variant,
                retrieval_metrics=self.retriever.get_metrics(),
                generation_metrics=self.generator.get_metrics(),
                hallucination_metrics=(
                    self.hallucination_detector.get_metrics()
                    if self.hallucination_detector else {}
                ),
                citation_metrics=response.citation_stats,
                total_time_ms=total_time,
                module_timings=response.module_timings,
            )

            # Compute answer quality if ground truth provided
            if ground_truth:
                query_log.answer_quality = self.metrics_logger.compute_answer_quality(
                    response.final_answer, ground_truth
                )

            # Compute retrieval quality if relevant docs provided
            if relevant_doc_ids:
                retrieved_ids = [d.document.doc_id for d in response.retrieved_docs]
                retrieval_quality = self.metrics_logger.compute_retrieval_quality(
                    retrieved_ids, relevant_doc_ids, self.config.top_k
                )
                query_log.retrieval_metrics.update(retrieval_quality)

            self.metrics_logger.log_query(query_log)

        return response