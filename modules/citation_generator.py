"""
═══════════════════════════════════════════════════════════════════════════════
 Citation Generator Module
═══════════════════════════════════════════════════════════════════════════════
 Maps answer segments to source documents and inserts inline citations.

 Two modes:
   A) Claims available (hallucination detection ON) → use claim-doc mappings
   B) No claims (hallucination OFF) → use semantic similarity to assign citations
═══════════════════════════════════════════════════════════════════════════════
"""

import re
import numpy as np
from typing import Dict, List

from interfaces import BaseCitationGenerator
from config import RAGConfig, Claim, RetrievedDocument


class CitationGenerator(BaseCitationGenerator):
    """
    Citation grounding: maps answer claims/sentences to source documents
    and inserts inline citation markers like [1], [2].
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self._last_metrics: Dict[str, float] = {}
        self._embedding_model = None

    def _get_embedding_model(self):
        """Lazy-load embedding model (only when needed for scenario B)."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedding_model

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = []
        for chunk in text.split("\n"):
            for s in re.split(r'(?<=[.!?])\s+', chunk):
                s = s.strip()
                if len(s.split()) >= 4:
                    sentences.append(s)
        return sentences

    def _build_doc_id_map(self, documents: List[RetrievedDocument]) -> Dict[str, int]:
        """Map doc_id → citation number [1], [2], etc."""
        seen = {}
        counter = 1
        for doc in documents:
            did = doc.document.doc_id
            if did not in seen:
                seen[did] = counter
                counter += 1
        return seen

    def generate_citations(
        self,
        answer: str,
        claims: List[Claim],
        documents: List[RetrievedDocument]
    ) -> Dict:
        """
        Generate citations mapping answer segments to source documents.
        """
        if not documents or not answer.strip():
            self._last_metrics = {
                "citation_coverage": 0.0,
                "citation_accuracy": 0.0,
                "citation_precision": 0.0,
                "citation_recall": 0.0,
            }
            return {
                "cited_answer": answer,
                "citations": [],
                "citation_stats": self._last_metrics,
            }

        doc_id_map = self._build_doc_id_map(documents)
        citations_list = []
        cited_answer = answer

        # ── SCENARIO A: Claims available (hallucination detection is ON) ──
        if claims and len(claims) > 0:
            cited_answer, citations_list = self._cite_from_claims(
                answer, claims, doc_id_map
            )
            total_items = len(claims)
            cited_items = len(set(c["claim_id"] for c in citations_list))

        # ── SCENARIO B: No claims (hallucination detection is OFF) ──
        else:
            cited_answer, citations_list = self._cite_from_similarity(
                answer, documents, doc_id_map
            )
            sentences = self._split_sentences(answer)
            total_items = max(len(sentences), 1)
            cited_items = len(set(c["claim_id"] for c in citations_list))

        # ── Compute metrics ──
        coverage = cited_items / max(total_items, 1)
        avg_confidence = (
            float(np.mean([c["confidence"] for c in citations_list]))
            if citations_list else 0.0
        )

        self._last_metrics = {
            "citation_coverage": coverage,
            "citation_accuracy": avg_confidence,
            "citation_precision": coverage,
            "citation_recall": coverage,
            "num_citations": len(citations_list),
            "avg_confidence": avg_confidence,
        }

        return {
            "cited_answer": cited_answer,
            "citations": citations_list,
            "citation_stats": self._last_metrics,
        }

    def _cite_from_claims(
        self,
        answer: str,
        claims: List[Claim],
        doc_id_map: Dict[str, int],
    ) -> tuple:
        """Insert citations using claim → supporting_doc_ids mapping."""
        citations_list = []
        cited_answer = answer

        for claim in claims:
            if not claim.is_supported or not claim.supporting_doc_ids:
                continue

            # Build citation tag like [1] or [1][3]
            cite_nums = []
            for doc_id in claim.supporting_doc_ids:
                if doc_id in doc_id_map:
                    cite_nums.append(doc_id_map[doc_id])

            if not cite_nums:
                continue

            cite_tag = "".join(f"[{n}]" for n in cite_nums)

            # Insert citation after the claim text in the answer
            if claim.text in cited_answer:
                cited_answer = cited_answer.replace(
                    claim.text, f"{claim.text} {cite_tag}", 1
                )

            for doc_id in claim.supporting_doc_ids:
                citations_list.append({
                    "claim_id": claim.claim_id,
                    "doc_id": doc_id,
                    "citation_num": doc_id_map.get(doc_id, 0),
                    "span": claim.text[:100],
                    "confidence": claim.support_score if claim.support_score else 0.0,
                })

        return cited_answer, citations_list

    def _cite_from_similarity(
        self,
        answer: str,
        documents: List[RetrievedDocument],
        doc_id_map: Dict[str, int],
    ) -> tuple:
        """Insert citations using semantic similarity (when no claims available)."""
        from sentence_transformers import util as st_util

        sentences = self._split_sentences(answer)
        if not sentences:
            return answer, []

        model = self._get_embedding_model()
        sent_embs = model.encode(sentences, convert_to_tensor=True)
        doc_texts = [d.document.content for d in documents]
        doc_embs = model.encode(doc_texts, convert_to_tensor=True)

        cosine_scores = st_util.cos_sim(sent_embs, doc_embs)

        citations_list = []
        cited_answer = answer

        for i, sentence in enumerate(sentences):
            scores = cosine_scores[i]
            best_score = 0.0
            cite_nums = []

            for j, score_val in enumerate(scores):
                s = score_val.item()
                if s >= self.config.citation_min_similarity:
                    doc_id = documents[j].document.doc_id
                    cite_nums.append(doc_id_map.get(doc_id, 0))
                    best_score = max(best_score, s)

                    citations_list.append({
                        "claim_id": f"sent_{i}",
                        "doc_id": doc_id,
                        "citation_num": doc_id_map.get(doc_id, 0),
                        "span": sentence[:100],
                        "confidence": s,
                    })

            if cite_nums:
                cite_tag = "".join(f"[{n}]" for n in sorted(set(cite_nums)))
                if sentence in cited_answer:
                    cited_answer = cited_answer.replace(
                        sentence, f"{sentence} {cite_tag}", 1
                    )

        return cited_answer, citations_list

    def get_metrics(self) -> Dict[str, float]:
        """Return citation-specific metrics for logging."""
        return self._last_metrics.copy()