"""
═══════════════════════════════════════════════════════════════════════════════
 Evaluation & Metrics Logger
═══════════════════════════════════════════════════════════════════════════════
 Collects, stores, and exports metrics for ablation study.
 Supports: Retrieval metrics, Answer quality, Hallucination, Citation metrics.
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import time
import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict
import logging
import csv
import os

logger = logging.getLogger(__name__)


@dataclass
class QueryLog:
    """Single query evaluation record."""
    run_id: str
    query: str
    variant: str
    timestamp: float = field(default_factory=time.time)

    # Retrieval
    retrieval_metrics: Dict[str, float] = field(default_factory=dict)

    # Generation
    generation_metrics: Dict[str, float] = field(default_factory=dict)

    # Hallucination
    hallucination_metrics: Dict[str, float] = field(default_factory=dict)

    # Citation
    citation_metrics: Dict[str, float] = field(default_factory=dict)

    # Answer quality (if ground truth available)
    answer_quality: Dict[str, float] = field(default_factory=dict)

    # Timing
    total_time_ms: float = 0.0
    module_timings: Dict[str, float] = field(default_factory=dict)


class MetricsLogger:
    """
    Central metrics collection and export for ablation experiments.
    """

    def __init__(self, experiment_name: str = "edu_rag_ablation"):
        self.experiment_name = experiment_name
        self.logs: List[QueryLog] = []

    def log_query(self, query_log: QueryLog) -> None:
        """Add a query evaluation record."""
        self.logs.append(query_log)
        logger.info(
            f"Logged query [{query_log.variant}]: {query_log.query[:50]}... | "
            f"Total: {query_log.total_time_ms:.1f}ms"
        )

    def get_variant_summary(self, variant: str) -> Dict[str, Any]:
        """
        Compute aggregate metrics for an ablation variant.
        Used for the comparison table in the UI.
        """
        variant_logs = [l for l in self.logs if l.variant == variant]
        if not variant_logs:
            return {}

        summary = {"variant": variant, "num_queries": len(variant_logs)}

        # ── Aggregate retrieval metrics ──
        retrieval_keys = set()
        for log in variant_logs:
            retrieval_keys.update(log.retrieval_metrics.keys())
        for key in retrieval_keys:
            values = [l.retrieval_metrics.get(key, 0) for l in variant_logs]
            summary[f"ret_{key}_mean"] = float(np.mean(values))
            summary[f"ret_{key}_std"] = float(np.std(values))

        # ── Aggregate hallucination metrics ──
        hall_keys = set()
        for log in variant_logs:
            hall_keys.update(log.hallucination_metrics.keys())
        for key in hall_keys:
            values = [l.hallucination_metrics.get(key, 0) for l in variant_logs]
            summary[f"hall_{key}_mean"] = float(np.mean(values))
            summary[f"hall_{key}_std"] = float(np.std(values))

        # ── Aggregate citation metrics ──
        cit_keys = set()
        for log in variant_logs:
            cit_keys.update(log.citation_metrics.keys())
        for key in cit_keys:
            values = [l.citation_metrics.get(key, 0) for l in variant_logs]
            summary[f"cit_{key}_mean"] = float(np.mean(values))
            summary[f"cit_{key}_std"] = float(np.std(values))

        # ── Timing ──
        times = [l.total_time_ms for l in variant_logs]
        summary["avg_time_ms"] = float(np.mean(times))
        summary["std_time_ms"] = float(np.std(times))

        return summary

    def get_comparison_table(self) -> List[Dict]:
        """
        Generate a comparison table across all variants.
        This is the main output for ablation study results.
        """
        variants = list(set(l.variant for l in self.logs))
        return [self.get_variant_summary(v) for v in sorted(variants)]

    def compute_answer_quality(
        self,
        predicted: str,
        reference: str,
    ) -> Dict[str, float]:
        """
        Compute answer quality metrics when ground truth is available.

        Metrics:
          - F1 Score (token-level)
          - Exact Match
          - ROUGE-L
        """
        metrics = {}

        # ── Token-level F1 ──
        pred_tokens = set(predicted.lower().split())
        ref_tokens = set(reference.lower().split())

        if not pred_tokens or not ref_tokens:
            metrics["f1_score"] = 0.0
            metrics["exact_match"] = 0.0
            metrics["rouge_l"] = 0.0
            return metrics

        common = pred_tokens & ref_tokens
        precision = len(common) / len(pred_tokens) if pred_tokens else 0
        recall = len(common) / len(ref_tokens) if ref_tokens else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        metrics["f1_score"] = f1
        metrics["token_precision"] = precision
        metrics["token_recall"] = recall

        # ── Exact Match ──
        metrics["exact_match"] = 1.0 if predicted.strip().lower() == reference.strip().lower() else 0.0

        # ── ROUGE-L (LCS-based) ──
        metrics["rouge_l"] = self._rouge_l(predicted, reference)

        return metrics

    def _rouge_l(self, predicted: str, reference: str) -> float:
        """Compute ROUGE-L F1 score using Longest Common Subsequence."""
        pred_words = predicted.lower().split()
        ref_words = reference.lower().split()

        if not pred_words or not ref_words:
            return 0.0

        # LCS dynamic programming
        m, n = len(pred_words), len(ref_words)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred_words[i-1] == ref_words[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        lcs_len = dp[m][n]
        precision = lcs_len / m if m > 0 else 0
        recall = lcs_len / n if n > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        return f1

    def compute_retrieval_quality(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int = 5,
    ) -> Dict[str, float]:
        """
        Compute retrieval quality metrics when ground truth relevance is available.

        Metrics:
          - Precision@k
          - Recall@k
          - MRR (Mean Reciprocal Rank)
          - nDCG@k
        """
        metrics = {}
        retrieved_k = retrieved_ids[:k]
        relevant_set = set(relevant_ids)

        # ── Precision@k ──
        hits = sum(1 for doc_id in retrieved_k if doc_id in relevant_set)
        metrics["precision_at_k"] = hits / k if k > 0 else 0

        # ── Recall@k ──
        metrics["recall_at_k"] = hits / len(relevant_set) if relevant_set else 0

        # ── MRR ──
        mrr = 0.0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_set:
                mrr = 1.0 / (i + 1)
                break
        metrics["mrr"] = mrr

        # ── nDCG@k ──
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_k):
            rel = 1.0 if doc_id in relevant_set else 0.0
            dcg += rel / np.log2(i + 2)  # i+2 because log2(1) = 0

        ideal_hits = min(len(relevant_set), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
        metrics["ndcg_at_k"] = dcg / idcg if idcg > 0 else 0

        return metrics

    def export_to_json(self, filepath: str) -> None:
        """Export all logs to JSON for analysis."""
        data = {
            "experiment": self.experiment_name,
            "num_queries": len(self.logs),
            "logs": [asdict(log) for log in self.logs],
            "comparison_table": self.get_comparison_table(),
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Exported {len(self.logs)} query logs to {filepath}")

    def export_to_csv(self, filepath: str) -> None:
        """Export flattened metrics to CSV for statistical analysis."""
        if not self.logs:
            return

        # Flatten all metrics into single rows
        rows = []
        for log in self.logs:
            row = {
                "run_id": log.run_id,
                "query": log.query[:100],
                "variant": log.variant,
                "total_time_ms": log.total_time_ms,
            }
            for k, v in log.retrieval_metrics.items():
                row[f"ret_{k}"] = v
            for k, v in log.hallucination_metrics.items():
                row[f"hall_{k}"] = v
            for k, v in log.citation_metrics.items():
                row[f"cit_{k}"] = v
            for k, v in log.answer_quality.items():
                row[f"aq_{k}"] = v
            rows.append(row)

        # Get all unique keys
        all_keys = set()
        for row in rows:
            all_keys.update(row.keys())

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
            writer.writeheader()
            writer.writerows(rows)

        logger.info(f"Exported {len(rows)} rows to {filepath}")