"""
═══════════════════════════════════════════════════════════════════════════════
 Module 4: Hallucination Detection
═══════════════════════════════════════════════════════════════════════════════
 Mathematical Formulation:
   Given answer A, extract claims C = {c₁, c₂, ..., cₘ}
   For each claim cᵢ and evidence set E = {e₁, e₂, ..., eₖ}:

     support(cᵢ) = max_{eⱼ ∈ E} P(entailment | eⱼ, cᵢ)

   A claim is supported if support(cᵢ) ≥ τ (threshold).

   Hallucination Rate = |{cᵢ : support(cᵢ) < τ}| / |C|
   Faithfulness Score = (1/|C|) Σᵢ support(cᵢ)

 NLI Model: Cross-encoder that classifies (premise, hypothesis) into
            {entailment, neutral, contradiction}

 Complexity: O(m × k) NLI inferences per query
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import re
import time
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import field

from config import RAGConfig, RetrievedDocument, Claim
from interfaces import BaseHallucinationDetector

logger = logging.getLogger(__name__)


class HallucinationDetector(BaseHallucinationDetector):
    """
    NLI-based hallucination detection with claim-level granularity.
    Uses a cross-encoder NLI model to check entailment between
    evidence passages and generated claims.
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self.nli_model = None
        self._last_metrics: Dict[str, float] = {}
        self._claim_counter = 0

    def _load_nli_model(self):
        """Lazy-load NLI cross-encoder model."""
        if self.nli_model is None:
            try:
                from sentence_transformers import CrossEncoder
                self.nli_model = CrossEncoder(
                    self.config.nli_model_name,
                    max_length=512
                )
                logger.info(f"Loaded NLI model: {self.config.nli_model_name}")
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. "
                    "Falling back to embedding-based similarity."
                )
            except Exception as e:
                logger.warning(f"Failed to load NLI model: {e}. Using fallback.")

    def extract_claims(self, text: str) -> List[Claim]:
        """
        Extract atomic claims from generated text.

        Strategy:
          - Split by sentences
          - Filter out short/non-factual sentences (questions, hedges)
          - Each remaining sentence = one claim

        For more advanced extraction, could use:
          - Constituency parsing for clause-level extraction
          - LLM-based decomposition

        Args:
            text: Generated answer text

        Returns:
            List of Claim objects
        """
        if not text.strip():
            return []

        # ── Sentence splitting ──
        # Handle common abbreviations to avoid false splits
        text_clean = text.replace("e.g.", "eg").replace("i.e.", "ie")
        text_clean = text_clean.replace("Dr.", "Dr").replace("Mr.", "Mr")
        text_clean = text_clean.replace("Mrs.", "Mrs").replace("Ms.", "Ms")
        text_clean = text_clean.replace("etc.", "etc")

        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text_clean.strip())

        claims = []
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            # ── Filter non-factual sentences ──
            # Skip very short fragments
            if len(sent.split()) < 4:
                continue

            # Skip questions
            if sent.endswith("?"):
                continue

            # Skip hedging / meta-commentary
            hedge_patterns = [
                r"^(I think|I believe|Perhaps|Maybe|It seems|In my opinion)",
                r"^(Note:|Disclaimer:|Warning:)",
                r"^(However|But|Although),?\s*(it is|this is)\s*(important|worth)",
            ]
            is_hedge = any(re.match(p, sent, re.IGNORECASE) for p in hedge_patterns)
            if is_hedge:
                continue

            self._claim_counter += 1
            claims.append(Claim(
                claim_id=f"claim_{self._claim_counter:04d}",
                text=sent,
                source_sentence=sent,
            ))

        logger.info(f"Extracted {len(claims)} claims from {len(sentences)} sentences")
        return claims

    def verify_claims(
        self,
        claims: List[Claim],
        evidence: List[RetrievedDocument]
    ) -> List[Claim]:
        """
        Verify each claim against retrieved evidence using NLI.

        For each claim:
          1. Pair with every evidence passage
          2. Run NLI inference → P(entailment), P(neutral), P(contradiction)
          3. Take max entailment score across all evidence
          4. Flag as supported/unsupported based on threshold

        Args:
            claims: List of extracted claims
            evidence: List of retrieved documents (evidence passages)

        Returns:
            Same claims list with verification fields populated
        """
        if not claims or not evidence:
            return claims

        t_start = time.time()
        evidence_texts = [doc.document.content for doc in evidence]
        evidence_ids = [doc.document.doc_id for doc in evidence]

        self._load_nli_model()

        if self.nli_model is not None:
            claims = self._verify_with_nli(claims, evidence_texts, evidence_ids)
        else:
            claims = self._verify_with_similarity(claims, evidence_texts, evidence_ids)

        # ── Compute aggregate metrics ──
        verify_time = time.time() - t_start
        supported_claims = [c for c in claims if c.is_supported]
        support_scores = [c.support_score for c in claims if c.support_score is not None]

        total = len(claims) if claims else 1
        self._last_metrics = {
            "hallucination_rate": 1.0 - (len(supported_claims) / total),
            "faithfulness_score": float(np.mean(support_scores)) if support_scores else 0.0,
            "num_claims": len(claims),
            "num_supported": len(supported_claims),
            "num_unsupported": total - len(supported_claims),
            "pct_unsupported_claims": ((total - len(supported_claims)) / total) * 100,
            "avg_support_score": float(np.mean(support_scores)) if support_scores else 0.0,
            "min_support_score": float(np.min(support_scores)) if support_scores else 0.0,
            "max_support_score": float(np.max(support_scores)) if support_scores else 0.0,
            "verification_time_ms": verify_time * 1000,
            "threshold_used": self.config.hallucination_threshold,
        }

        logger.info(
            f"Verified {len(claims)} claims in {verify_time*1000:.1f}ms | "
            f"Supported: {len(supported_claims)}/{len(claims)} | "
            f"Hallucination Rate: {self._last_metrics['hallucination_rate']:.2%}"
        )

        return claims

    def _verify_with_nli(
        self,
        claims: List[Claim],
        evidence_texts: List[str],
        evidence_ids: List[str]
    ) -> List[Claim]:
        """Verify claims using NLI cross-encoder model."""
        for claim in claims:
            best_score = 0.0
            best_evidence_id = None
            best_evidence_text = None

            # Create all (evidence, claim) pairs for this claim
            pairs = [(ev, claim.text) for ev in evidence_texts]

            if pairs:
                # NLI prediction: returns scores for [contradiction, neutral, entailment]
                scores = self.nli_model.predict(pairs)

                # Handle different output formats
                if isinstance(scores, np.ndarray) and scores.ndim == 2:
                    # Multi-class output: take entailment score (index 2 typically)
                    # For deberta-nli: labels are [contradiction, entailment, neutral]
                    # or [entailment, neutral, contradiction] depending on model
                    entailment_scores = self._extract_entailment_scores(scores)
                else:
                    # Single score output (some models)
                    entailment_scores = np.array(scores).flatten()

                best_idx = int(np.argmax(entailment_scores))
                best_score = float(entailment_scores[best_idx])
                best_evidence_id = evidence_ids[best_idx]
                best_evidence_text = evidence_texts[best_idx]

            claim.support_score = best_score
            claim.is_supported = best_score >= self.config.hallucination_threshold
            if best_evidence_id:
                claim.supporting_doc_ids = [best_evidence_id]
            if best_evidence_text:
                claim.evidence_text = best_evidence_text[:200]

        return claims

    def _extract_entailment_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Extract entailment probabilities from NLI output.
        Handles different label orderings across models.
        """
        # Apply softmax if raw logits
        from scipy.special import softmax as sp_softmax
        probs = sp_softmax(scores, axis=1)

        # Most DeBERTa NLI models: [contradiction, neutral, entailment]
        # We want the entailment column (index 2)
        if probs.shape[1] == 3:
            return probs[:, -1]  # entailment is last
        elif probs.shape[1] == 2:
            return probs[:, 1]   # binary: [not_entail, entail]
        else:
            return probs[:, 0]

    def _verify_with_similarity(
        self,
        claims: List[Claim],
        evidence_texts: List[str],
        evidence_ids: List[str]
    ) -> List[Claim]:
        """
        Fallback: verify claims using embedding cosine similarity.
        Less accurate than NLI but requires no additional model.
        """
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")

            claim_texts = [c.text for c in claims]
            claim_embs = model.encode(claim_texts, normalize_embeddings=True)
            evidence_embs = model.encode(evidence_texts, normalize_embeddings=True)

            # Cosine similarity matrix: (n_claims, n_evidence)
            sim_matrix = claim_embs @ evidence_embs.T

            for i, claim in enumerate(claims):
                best_idx = int(np.argmax(sim_matrix[i]))
                best_score = float(sim_matrix[i, best_idx])

                # Map similarity to support score (calibrate threshold accordingly)
                claim.support_score = best_score
                claim.is_supported = best_score >= self.config.hallucination_threshold
                claim.supporting_doc_ids = [evidence_ids[best_idx]]
                claim.evidence_text = evidence_texts[best_idx][:200]

        except ImportError:
            logger.error("No model available for hallucination detection.")
            # Mark all claims as unverified
            for claim in claims:
                claim.support_score = 0.0
                claim.is_supported = None

        return claims

    def filter_hallucinated_claims(
        self,
        answer: str,
        claims: List[Claim]
    ) -> Tuple[str, List[Claim]]:
        """
        Remove or flag hallucinated content from the answer.

        Strategy: Remove sentences that map to unsupported claims.
        Returns the filtered answer and the list of removed claims.

        Args:
            answer: Original generated answer
            claims: Verified claims

        Returns:
            (filtered_answer, removed_claims)
        """
        unsupported = [c for c in claims if c.is_supported is False]
        if not unsupported:
            return answer, []

        # Remove unsupported sentences
        filtered_sentences = []
        sentences = re.split(r'(?<=[.!?])\s+', answer.strip())

        unsupported_texts = {c.source_sentence.lower().strip() for c in unsupported}

        for sent in sentences:
            sent_clean = sent.strip().lower()
            # Check if this sentence matches any unsupported claim
            is_unsupported = any(
                self._fuzzy_match(sent_clean, ut) for ut in unsupported_texts
            )
            if not is_unsupported:
                filtered_sentences.append(sent)

        filtered_answer = " ".join(filtered_sentences)

        # If we removed everything, keep original with a warning prefix
        if not filtered_answer.strip():
            filtered_answer = (
                "[Warning: Low confidence in generated answer. "
                "Please verify with original sources.]\n\n" + answer
            )

        logger.info(
            f"Filtered {len(unsupported)} unsupported claims. "
            f"Original: {len(sentences)} sentences → Filtered: {len(filtered_sentences)}"
        )

        return filtered_answer, unsupported

    def _fuzzy_match(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """Simple fuzzy matching based on word overlap."""
        words1 = set(text1.split())
        words2 = set(text2.split())
        if not words1 or not words2:
            return False
        overlap = len(words1 & words2)
        return overlap / max(len(words1), len(words2)) >= threshold

    def get_metrics(self) -> Dict[str, float]:
        """Return hallucination detection metrics from last verification."""
        return self._last_metrics.copy()

    def get_detailed_report(self, claims: List[Claim]) -> Dict:
        """
        Generate a detailed hallucination report for the UI.

        Returns:
            Dict with structured report data for visualization
        """
        supported = [c for c in claims if c.is_supported is True]
        unsupported = [c for c in claims if c.is_supported is False]
        unverified = [c for c in claims if c.is_supported is None]

        return {
            "summary": {
                "total_claims": len(claims),
                "supported": len(supported),
                "unsupported": len(unsupported),
                "unverified": len(unverified),
                "hallucination_rate": len(unsupported) / max(len(claims), 1),
                "faithfulness_score": np.mean(
                    [c.support_score for c in claims if c.support_score is not None]
                ) if claims else 0.0,
            },
            "claims_detail": [
                {
                    "id": c.claim_id,
                    "text": c.text,
                    "is_supported": c.is_supported,
                    "support_score": c.support_score,
                    "evidence": c.evidence_text,
                    "supporting_docs": c.supporting_doc_ids,
                }
                for c in claims
            ],
        }