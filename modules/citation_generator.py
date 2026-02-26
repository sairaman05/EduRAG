import re
import numpy as np
from typing import Dict, List
from sentence_transformers import SentenceTransformer, util

from interfaces import BaseCitationGenerator
from config import RAGConfig, Claim, RetrievedDocument

class CitationGenerator(BaseCitationGenerator):
    """
    Citation grounding: maps answer claims to source documents
    and inserts inline citation markers.
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self._last_metrics = {}
        # We load this ONLY if used as a fallback if claims are empty
        self.embedding_model = None

    def generate_citations(
        self,
        answer: str,
        claims: List[Claim],
        documents: List[RetrievedDocument]
    ) -> Dict:
        """
        Input:
            - answer: the current final_answer string
            - claims: List[Claim] extracted from the answer
            - documents: List[RetrievedDocument] used as evidence

        Output: MUST return a dict with exactly these keys:
            {
                "cited_answer": str,
                "citations": List[Dict],
                "citation_stats": Dict[str, float]
            }
        """
        cited_answer = answer
        citations_list = []
        
        # SCENARIO A: Hallucination Detection is ON (claims are present)
        if claims and len(claims) > 0:
            cited_claims_set = set()
            for claim in claims:
                if claim.is_supported and claim.supporting_doc_ids:
                    # Format the citation string, e.g., "[bio_001][bio_002]"
                    cite_tags = "".join([f"[{doc_id}]" for doc_id in claim.supporting_doc_ids])
                    
                    # Inject tag into the string right after the claim text
                    if claim.text in cited_answer:
                        cited_answer = cited_answer.replace(claim.text, f"{claim.text} {cite_tags}")
                    
                    for doc_id in claim.supporting_doc_ids:
                        citations_list.append({
                            "claim_id": claim.claim_id,
                            "doc_id": doc_id,
                            "span": claim.text,
                            "confidence": claim.support_score if claim.support_score is not None else 1.0
                        })
                        cited_claims_set.add(claim.claim_id)
                        
        # SCENARIO B: Hallucination is OFF
        # Use Dense Semantic Similarity to find citations ourselves
        else:
            if documents:
                if self.embedding_model is None:
                    # Using the same model as dense_retrieval.py
                    self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                    
                # 1. Split answer into sentences by newlines and punctuation
                sentences = []
                for chunk in answer.split('\n'):
                    for s in re.split(r'(?<=[.!?])\s+', chunk):
                        if len(s.strip()) > 5:
                            sentences.append(s.strip())
                
                if sentences:
                    # 2. Embed sentences and documents
                    sentence_embeddings = self.embedding_model.encode(sentences, convert_to_tensor=True)
                    
                    valid_docs = documents
                    doc_contents = [d.document.content for d in valid_docs]
                    doc_tensor = self.embedding_model.encode(doc_contents, convert_to_tensor=True)
                    
                    # 3. Calculate Cosine Similarities
                    cosine_scores = util.cos_sim(sentence_embeddings, doc_tensor)
                    
                    modified_answer = answer
                    
                    for i, sentence in enumerate(sentences):
                        scores = cosine_scores[i]
                        cited_doc_ids = []
                        best_score = 0.0
                        for j, score in enumerate(scores):
                            s = score.item()
                            if s >= self.config.citation_min_similarity:
                                cited_doc_ids.append(valid_docs[j].document.doc_id)
                                best_score = max(best_score, s)
                                
                        if cited_doc_ids:
                            cite_tags = "".join([f"[{doc_id}]" for doc_id in cited_doc_ids])
                            if sentence in modified_answer:
                                modified_answer = modified_answer.replace(sentence, f"{sentence} {cite_tags}")
                            
                            for doc_id in cited_doc_ids:
                                citations_list.append({
                                    "claim_id": f"sent_{i}",
                                    "doc_id": doc_id,
                                    "span": sentence,
                                    "confidence": best_score
                                })
                                
                    cited_answer = modified_answer

        # Calculate final metrics for this query
        self._last_metrics = {
            "citation_coverage": 0.0,
            "citation_accuracy": 0.0,  # Proxy or requires ground truth
            "citation_precision": 0.0, # Proxy or requires ground truth
            "citation_recall": 0.0     # Proxy or requires ground truth
        }
        
        if claims and len(claims) > 0:
            cited_claims_count = len(set(c["claim_id"] for c in citations_list))
            self._last_metrics["citation_coverage"] = cited_claims_count / len(claims)
        elif not claims and citations_list:
            sentences = []
            for chunk in answer.split('\n'):
                for s in re.split(r'(?<=[.!?])\s+', chunk):
                    if len(s.strip()) > 5:
                        sentences.append(s.strip())
            
            if len(sentences) > 0:
               cited_sentences_count = len(set(c["claim_id"] for c in citations_list))
               self._last_metrics["citation_coverage"] = cited_sentences_count / len(sentences)

        return {
            "cited_answer": cited_answer,
            "citations": citations_list,
            "citation_stats": self._last_metrics
        }

    def get_metrics(self) -> Dict[str, float]:
        """Return citation-specific metrics for logging."""
        return self._last_metrics
