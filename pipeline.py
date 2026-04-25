"""
═══════════════════════════════════════════════════════════════════════════════
 RAG Pipeline Orchestrator
═══════════════════════════════════════════════════════════════════════════════
 Orchestrates the full pipeline: Retrieve → [MMR] → Generate → [Hallucination] → [Citation]
 Each module is conditionally invoked based on RAGConfig flags.

 NEW: After the conditional pipeline, an evaluation-only pass ALWAYS runs
 hallucination scoring and citation scoring on the raw answer so that
 ALL variants have comparable metrics — without modifying the answer.
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

        # ── Always create an evaluation-only hallucination detector ──
        # This is used for the read-only scoring pass on ALL variants
        self._eval_hallucination_detector = HallucinationDetector(config)

        # Teammates replace these with real implementations:
        self.mmr_reranker = None
        self.citation_generator = None

        # Always create an evaluation-only citation generator
        self._eval_citation_generator = None

        self._try_load_teammate_modules()
        self._try_load_eval_citation_generator()

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

    def _try_load_eval_citation_generator(self):
        """
        Load citation generator for evaluation-only scoring pass.
        This runs on ALL variants regardless of flags.
        """
        try:
            from modules.citation_generator import CitationGenerator
            self._eval_citation_generator = CitationGenerator(self.config)
            logger.info("Loaded citation generator for evaluation scoring")
        except ImportError:
            logger.info("Citation module not available for evaluation scoring, using stub")
            self._eval_citation_generator = CitationGeneratorStub()

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
        # Stage 1: Dense Retrieval (ALWAYS ON)
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
        # Stage 3: LLM Answer Generation (ALWAYS ON)
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

            response.claims = self.hallucination_detector.extract_claims(response.raw_answer)
            response.claims = self.hallucination_detector.verify_claims(
                response.claims, active_docs
            )
            response.final_answer, removed = self.hallucination_detector.filter_hallucinated_claims(
                response.raw_answer, response.claims
            )
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
        # Stage 6: EVALUATION-ONLY SCORING PASS (ALWAYS RUNS)
        # ═══════════════════════════════════════════════════════
        # This computes hallucination + citation metrics on the RAW answer
        # for ALL variants, WITHOUT modifying the answer.
        # If the active module already ran, we still compute eval metrics
        # but the effective getters prefer the active module's results.
        self._run_evaluation_pass(response, active_docs)

        # ═══════════════════════════════════════════════════════
        # Collect all metrics
        # ═══════════════════════════════════════════════════════
        total_time = (time.time() - t_total_start) * 1000
        response.module_timings["total"] = total_time

        # ── Merge all metrics into response.metrics ──
        response.metrics.update(self.retriever.get_metrics())
        response.metrics.update(self.generator.get_metrics())

        # Add active hallucination metrics
        if self.hallucination_detector:
            response.metrics.update({
                f"hall_{k}": v for k, v in self.hallucination_detector.get_metrics().items()
            })

        # Always add evaluation-pass metrics (prefixed with eval_)
        for k, v in response.eval_hallucination_stats.items():
            response.metrics[f"eval_hall_{k}"] = v
        for k, v in response.eval_citation_stats.items():
            response.metrics[f"eval_cit_{k}"] = v

        # ── Log to metrics logger ──
        if self.config.log_metrics:
            # Use effective stats (active if available, else eval)
            eff_hall = response.get_effective_hallucination_stats()
            eff_cit = response.get_effective_citation_stats()

            query_log = QueryLog(
                run_id=response.run_id,
                query=question,
                variant=response.config_variant,
                retrieval_metrics=self.retriever.get_metrics(),
                generation_metrics=self.generator.get_metrics(),
                hallucination_metrics=eff_hall,
                citation_metrics=eff_cit,
                total_time_ms=total_time,
                module_timings=response.module_timings,
            )

            if ground_truth:
                query_log.answer_quality = self.metrics_logger.compute_answer_quality(
                    response.final_answer, ground_truth
                )

            if relevant_doc_ids:
                retrieved_ids = [d.document.doc_id for d in response.retrieved_docs]
                retrieval_quality = self.metrics_logger.compute_retrieval_quality(
                    retrieved_ids, relevant_doc_ids, self.config.top_k
                )
                query_log.retrieval_metrics.update(retrieval_quality)

            self.metrics_logger.log_query(query_log)

        return response

    def _run_evaluation_pass(self, response: RAGResponse, active_docs) -> None:
        """
        Evaluation-only scoring pass that ALWAYS runs.

        This computes hallucination and citation metrics on the raw answer
        without modifying it. The results are stored in eval_* fields
        on the response, providing a uniform comparison basis across
        all ablation variants.

        If the hallucination or citation module already ran as part of
        the active pipeline, the eval pass still runs independently
        on the raw answer for consistency.
        """
        t_start = time.time()

        # ── Evaluation Hallucination Scoring ──
        try:
            eval_claims = self._eval_hallucination_detector.extract_claims(response.raw_answer)
            eval_claims = self._eval_hallucination_detector.verify_claims(eval_claims, active_docs)
            response.eval_claims = eval_claims
            response.eval_hallucination_stats = self._eval_hallucination_detector.get_metrics()
        except Exception as e:
            logger.warning(f"Evaluation hallucination pass failed: {e}")
            response.eval_hallucination_stats = {
                "hallucination_rate": 0.0,
                "faithfulness_score": 0.0,
                "num_claims": 0,
                "num_supported": 0,
                "num_unsupported": 0,
            }

        # ── Evaluation Citation Scoring ──
        try:
            if self._eval_citation_generator:
                # Use eval_claims for citation scoring
                eval_cit_result = self._eval_citation_generator.generate_citations(
                    response.raw_answer, response.eval_claims, active_docs
                )
                response.eval_citation_stats = eval_cit_result.get("citation_stats", {})
        except Exception as e:
            logger.warning(f"Evaluation citation pass failed: {e}")
            response.eval_citation_stats = {
                "citation_coverage": 0.0,
                "citation_accuracy": 0.0,
                "citation_precision": 0.0,
                "citation_recall": 0.0,
            }

        response.module_timings["eval_scoring"] = (time.time() - t_start) * 1000