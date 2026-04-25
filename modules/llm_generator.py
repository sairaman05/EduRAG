"""
═══════════════════════════════════════════════════════════════════════════════
 Module 5: LLM Answer Generation
═══════════════════════════════════════════════════════════════════════════════
 Responsible for generating answers from retrieved context.
 Supports multiple LLM backends: Ollama (local), OpenAI API.
═══════════════════════════════════════════════════════════════════════════════
"""

import time
import logging
from typing import List, Dict, Optional

from config import RAGConfig, RetrievedDocument

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a helpful educational assistant. Answer the question based ONLY on the provided context. 
If the context does not contain enough information to answer, say so clearly.
Be factual, precise, and cite specific details from the context.
Do not make up information beyond what is provided.
Provide a complete, thorough answer. Do not cut off mid-sentence."""

CONTEXT_TEMPLATE = """Context:
{context}

Question: {question}

Answer based only on the above context. Provide a complete and detailed response:"""


class LLMGenerator:
    """
    LLM answer generation with support for multiple backends.
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self._last_metrics: Dict[str, float] = {}

    def _build_context(self, documents: List[RetrievedDocument]) -> str:
        """Build context string from retrieved documents."""
        context_parts = []
        for i, doc in enumerate(documents):
            source = doc.document.metadata.get("source", f"Document {i+1}")
            context_parts.append(
                f"[Source {i+1}: {source}] (relevance: {doc.score:.3f})\n"
                f"{doc.document.content}"
            )
        return "\n\n---\n\n".join(context_parts)

    def generate(
        self,
        query: str,
        documents: List[RetrievedDocument],
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate answer using the configured LLM.

        Args:
            query: User question
            documents: Retrieved context documents
            system_prompt: Optional override for system prompt

        Returns:
            Generated answer string
        """
        t_start = time.time()

        context = self._build_context(documents)
        prompt = CONTEXT_TEMPLATE.format(context=context, question=query)
        sys_prompt = system_prompt or SYSTEM_PROMPT

        # ── Route to appropriate backend ──
        if self.config.llm_provider == "ollama":
            answer = self._generate_ollama(sys_prompt, prompt)
        elif self.config.llm_provider == "openai":
            answer = self._generate_openai(sys_prompt, prompt)
        else:
            raise ValueError(f"Unknown LLM provider: {self.config.llm_provider}. Use 'ollama' or 'openai'.")

        gen_time = time.time() - t_start
        self._last_metrics = {
            "generation_time_ms": gen_time * 1000,
            "context_length_chars": len(context),
            "num_context_docs": len(documents),
            "answer_length_chars": len(answer),
            "answer_num_sentences": len([s for s in answer.split('.') if s.strip()]),
        }

        logger.info(f"Generated answer in {gen_time*1000:.1f}ms | {len(answer)} chars")
        return answer

    def _generate_ollama(self, system_prompt: str, prompt: str) -> str:
        """Generate using local Ollama."""
        try:
            import requests
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.config.llm_model,
                    "prompt": prompt,
                    "system": system_prompt,
                    "options": {
                        "temperature": self.config.llm_temperature,
                        "num_predict": self.config.llm_max_tokens,
                    },
                    "stream": False,
                },
                timeout=180,
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return self._generate_fallback(prompt)

    def _generate_openai(self, system_prompt: str, prompt: str) -> str:
        """Generate using OpenAI API."""
        try:
            import openai
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            return self._generate_fallback(prompt)

    def _generate_huggingface(self, system_prompt: str, prompt: str) -> str:
        """Generate using HuggingFace transformers pipeline."""
        try:
            from transformers import pipeline
            generator = pipeline(
                "text-generation",
                model=self.config.llm_model,
                max_new_tokens=self.config.llm_max_tokens,
                temperature=self.config.llm_temperature,
            )
            full_prompt = f"{system_prompt}\n\n{prompt}"
            result = generator(full_prompt, do_sample=True)
            return result[0]["generated_text"][len(full_prompt):].strip()
        except Exception as e:
            logger.error(f"HuggingFace generation failed: {e}")
            return self._generate_fallback(prompt)

    def _generate_fallback(self, prompt: str) -> str:
        """
        Extractive fallback when no LLM is available.
        Concatenates the most relevant passages as the "answer".
        """
        logger.warning("Using extractive fallback (no LLM available)")
        # Extract the context from the prompt
        if "Context:" in prompt and "Question:" in prompt:
            context = prompt.split("Context:")[1].split("Question:")[0].strip()
            # Take first 500 chars of context as a crude answer
            return f"[Extractive Answer] {context[:1500]}..."
        return "[No LLM available for generation. Please configure an LLM backend.]"

    def get_metrics(self) -> Dict[str, float]:
        return self._last_metrics.copy()