# 📚 EduRAG — Education-Focused Retrieval-Augmented Generation System

> A modular, research-grade RAG system designed for **ablation study experiments** comparing Vanilla RAG against enhanced variants with Hallucination Detection, MMR Diversification, and Citation Grounding.

---

## Table of Contents

1. [Research Objective](#1-research-objective)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Project Structure](#3-project-structure)
4. [Detailed Pipeline Flow](#4-detailed-pipeline-flow)
5. [What We Built (Our Contribution)](#5-what-we-built-our-contribution)
6. [Shared Contract — config.py and interfaces.py](#6-shared-contract--configpy-and-interfacespy)
7. [How to Run](#7-how-to-run)
8. [Streamlit UI — Four Modes](#8-streamlit-ui--four-modes)
9. [Teammate Integration Guide](#9-teammate-integration-guide)
10. [Ablation Study Design](#10-ablation-study-design)
11. [Evaluation Metrics](#11-evaluation-metrics)
12. [Technical Details](#12-technical-details)
13. [Dependencies](#13-dependencies)

---

## 1. Research Objective

We aim to build a **modular RAG pipeline** for educational question-answering that:

- Improves **factual correctness** of generated answers
- **Reduces hallucinations** using NLI-based claim verification
- Provides **citation-grounded** answers (teammate module)
- Increases **retrieval diversity** via MMR reranking (teammate module)
- Supports **component-wise ablation** where each module can be independently toggled ON/OFF

The final goal is to experimentally prove:

1. The **Full System outperforms Vanilla RAG** across all metrics
2. **Each module contributes measurable improvement** individually
3. Improvements are **statistically significant** (via multiple runs and std computation)

---

## 2. System Architecture Overview

The pipeline is a **linear chain** of 5 stages. Stages 2, 4, and 5 are conditionally executed based on boolean flags in the configuration.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER QUERY                                   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 1: Dense Retrieval  (ALWAYS ON)                               │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  • Encode query using all-MiniLM-L6-v2                         │  │
│  │  • Compute cosine similarity against all indexed documents     │  │
│  │  • Return top-k most similar documents                         │  │
│  │  • Output: List[RetrievedDocument] with similarity scores      │  │
│  └────────────────────────────────────────────────────────────────┘  │
│  File: modules/dense_retrieval.py                                    │
│  Metrics: avg_similarity, max/min_similarity, retrieval_time_ms      │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 2: MMR Reranking  (CONDITIONAL — use_mmr flag)                │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  • Takes retrieved docs from Stage 1                           │  │
│  │  • Applies Maximal Marginal Relevance diversification          │  │
│  │  • Balances relevance vs diversity via lambda parameter         │  │
│  │  • Output: Reranked List[RetrievedDocument] with mmr_scores    │  │
│  └────────────────────────────────────────────────────────────────┘  │
│  File: modules/mmr_reranker.py  (🔌 TEAMMATE MODULE)                │
│  Currently: modules/stubs.py (pass-through placeholder)              │
│  Metrics: diversity_score, mmr_lambda                                │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 3: LLM Answer Generation  (ALWAYS ON)                        │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  • Builds context string from active docs (MMR or Dense)       │  │
│  │  • Sends system prompt + context + question to LLM             │  │
│  │  • Supports: Ollama (local), OpenAI API, HuggingFace           │  │
│  │  • Fallback: extractive answer if no LLM available             │  │
│  │  • Output: raw_answer string                                   │  │
│  └────────────────────────────────────────────────────────────────┘  │
│  File: modules/llm_generator.py                                      │
│  Metrics: generation_time_ms, answer_length, num_context_docs        │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 4: Hallucination Detection  (CONDITIONAL — use_hallucination) │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Step 4a: Claim Extraction                                     │  │
│  │    • Split answer into sentences                               │  │
│  │    • Filter non-factual content (questions, hedges, short)     │  │
│  │    • Each remaining sentence = one Claim object                │  │
│  │                                                                │  │
│  │  Step 4b: Claim Verification (NLI-based)                       │  │
│  │    • For each claim, pair with every evidence passage           │  │
│  │    • Run NLI cross-encoder → P(entailment | evidence, claim)   │  │
│  │    • support_score = max entailment score across evidence       │  │
│  │    • is_supported = (support_score >= threshold)                │  │
│  │                                                                │  │
│  │  Step 4c: Hallucination Filtering                              │  │
│  │    • Remove sentences mapping to unsupported claims             │  │
│  │    • Output: final_answer (cleaned) + detailed claim report     │  │
│  └────────────────────────────────────────────────────────────────┘  │
│  File: modules/hallucination_detector.py                             │
│  Metrics: hallucination_rate, faithfulness_score,                    │
│           num_supported, pct_unsupported_claims, verification_time   │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 5: Citation Generation  (CONDITIONAL — use_citation flag)     │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  • Maps each claim to its supporting source document(s)        │  │
│  │  • Inserts inline citations [1], [2] into the answer           │  │
│  │  • Computes citation coverage, accuracy, precision, recall     │  │
│  │  • Output: cited_answer + citation metadata                    │  │
│  └────────────────────────────────────────────────────────────────┘  │
│  File: modules/citation_generator.py  (🔌 TEAMMATE MODULE)          │
│  Currently: modules/stubs.py (pass-through placeholder)              │
│  Metrics: citation_coverage, citation_accuracy, citation_precision   │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  METRICS COLLECTION & LOGGING                                        │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  • Merge metrics from all active modules                       │  │
│  │  • Compute answer quality if ground truth provided (F1, EM,    │  │
│  │    ROUGE-L, Token Precision, Token Recall)                     │  │
│  │  • Compute retrieval quality if relevant_doc_ids provided      │  │
│  │    (Precision@k, Recall@k, MRR, nDCG)                         │  │
│  │  • Log to MetricsLogger for aggregation and export             │  │
│  │  • Output: Complete RAGResponse object                         │  │
│  └────────────────────────────────────────────────────────────────┘  │
│  File: utils/metrics.py                                              │
│  Exports: JSON, CSV for statistical analysis                         │
└──────────────────────────────────────────────────────────────────────┘
```

### Key Design Principle: Enrichment, Not Replacement

The `RAGResponse` object flows through the entire pipeline. Each module **enriches** specific fields on the response — it never creates a new response or destroys previous module outputs. This means all intermediate data is preserved for evaluation and ablation.

```
RAGResponse created
    │
    ├── Stage 1 fills: retrieved_docs, retrieval metrics
    ├── Stage 2 fills: mmr_reranked_docs, mmr_score on each doc
    ├── Stage 3 fills: raw_answer, final_answer
    ├── Stage 4 fills: claims, hallucination_flags, hallucination_stats
    │                  (also modifies final_answer by removing unsupported sentences)
    ├── Stage 5 fills: citations, citation_stats
    │                  (also modifies final_answer by adding inline citations)
    └── Logger fills: metrics, module_timings
```

---

## 3. Project Structure

```
edu_rag/
│
├── app.py                              # 🖥️  Streamlit UI (4 modes)
├── config.py                           # 📋  Shared types, configs, ablation variants
├── interfaces.py                       # 📐  Abstract base classes (contracts)
├── pipeline.py                         # 🔗  Pipeline orchestrator
├── requirements.txt                    # 📦  Python dependencies
├── README.md                           # 📖  This file
│
├── modules/
│   ├── __init__.py
│   ├── dense_retrieval.py              # ✅  OUR MODULE — Dense cosine retrieval
│   ├── hallucination_detector.py       # ✅  OUR MODULE — NLI-based claim verification
│   ├── llm_generator.py               # ✅  OUR MODULE — Multi-backend LLM generation
│   ├── stubs.py                        # ⬜  Placeholder stubs for teammate modules
│   ├── mmr_reranker.py                 # 🔌  TEAMMATE — Drop-in MMR module
│   └── citation_generator.py           # 🔌  TEAMMATE — Drop-in Citation module
│
└── utils/
    ├── __init__.py
    └── metrics.py                      # ✅  OUR MODULE — Evaluation & logging utilities
```

### File Responsibilities

| File | Role | Who |
|------|------|-----|
| `config.py` | Defines `RAGConfig`, `Document`, `RetrievedDocument`, `Claim`, `RAGResponse`, `ABLATION_VARIANTS` | Shared (everyone imports from here) |
| `interfaces.py` | Abstract classes: `BaseRetriever`, `BaseReranker`, `BaseHallucinationDetector`, `BaseCitationGenerator` | Shared contract |
| `pipeline.py` | Orchestrates Stage 1→2→3→4→5, handles conditional module loading, metrics logging | Core (we built) |
| `modules/dense_retrieval.py` | Sentence-transformer embedding + cosine similarity retrieval | We built |
| `modules/hallucination_detector.py` | Claim extraction + NLI verification + hallucination filtering | We built |
| `modules/llm_generator.py` | Ollama/OpenAI/HuggingFace answer generation with context | We built |
| `modules/stubs.py` | No-op placeholders so system runs without teammate modules | We built |
| `utils/metrics.py` | F1, ROUGE-L, EM, Precision@k, Recall@k, MRR, nDCG, logging, export | We built |
| `app.py` | Streamlit UI with 4 modes: Single Query, Comparison, Ablation, Dashboard | We built |
| `modules/mmr_reranker.py` | MMR diversified reranking | **MMR Teammate** |
| `modules/citation_generator.py` | Citation grounding with inline references | **Citation Teammate** |

---

## 4. Detailed Pipeline Flow

### 4.1 What Happens When a User Asks a Question

Let's trace a complete query through the system with `use_hallucination_detection=True`:

```
User types: "What is photosynthesis?"
        │
        ▼
[1] pipeline.query("What is photosynthesis?") called
        │
        ▼
[2] RAGResponse object created with:
    - query = "What is photosynthesis?"
    - run_id = unique UUID
    - config_variant = "RAG+HallucinationDetection"
        │
        ▼
[3] STAGE 1 — Dense Retrieval
    │  dense_retrieval.retrieve("What is photosynthesis?", top_k=5)
    │    ├── Encode query → 384-dim vector using all-MiniLM-L6-v2
    │    ├── Compute cosine similarity with all 12 indexed document embeddings
    │    ├── Sort by similarity descending
    │    └── Return top-5 as RetrievedDocument objects:
    │         [bio_001 (0.87), bio_002 (0.82), bio_003 (0.78), chem_001 (0.31), ...]
    │
    ├── response.retrieved_docs = [top-5 results]
    ├── module_timings["retrieval"] = 45ms
        │
        ▼
[4] STAGE 2 — MMR (SKIPPED, use_mmr=False)
        │
        ▼
[5] STAGE 3 — LLM Generation
    │  active_docs = response.get_active_docs()  → uses retrieved_docs (no MMR)
    │  generator.generate("What is photosynthesis?", active_docs)
    │    ├── Build context string from 5 documents with source labels
    │    ├── Format prompt: System prompt + Context + Question
    │    ├── Send to Ollama (llama3) with temperature=0.3
    │    └── Return: "Photosynthesis is the process by which green plants use
    │                sunlight to synthesize nutrients from carbon dioxide and
    │                water. The light-dependent reactions occur in thylakoid
    │                membranes producing ATP and NADPH. The Calvin cycle then
    │                fixes carbon dioxide into organic molecules using RuBisCO."
    │
    ├── response.raw_answer = [generated text]
    ├── response.final_answer = [same as raw for now]
    ├── module_timings["generation"] = 2300ms
        │
        ▼
[6] STAGE 4 — Hallucination Detection
    │
    │  Step 4a: Claim Extraction
    │    hallucination_detector.extract_claims(raw_answer)
    │    ├── Split into sentences
    │    ├── Filter: len >= 4 words, not question, not hedge
    │    └── Result: 3 Claim objects:
    │         claim_0001: "Photosynthesis is the process by which green plants
    │                      use sunlight to synthesize nutrients from CO2 and water."
    │         claim_0002: "The light-dependent reactions occur in thylakoid
    │                      membranes producing ATP and NADPH."
    │         claim_0003: "The Calvin cycle then fixes carbon dioxide into
    │                      organic molecules using RuBisCO."
    │
    │  Step 4b: Claim Verification
    │    hallucination_detector.verify_claims(claims, active_docs)
    │    For each claim:
    │      ├── Pair with ALL 5 evidence passages
    │      ├── Run NLI cross-encoder on each (evidence, claim) pair
    │      ├── Get P(entailment) for each pair
    │      ├── support_score = max P(entailment) across evidence
    │      └── is_supported = (support_score >= 0.5)
    │
    │    Results:
    │      claim_0001: support=0.91, supported=True,  evidence=bio_001
    │      claim_0002: support=0.85, supported=True,  evidence=bio_002
    │      claim_0003: support=0.88, supported=True,  evidence=bio_003
    │
    │  Step 4c: Filtering
    │    hallucination_detector.filter_hallucinated_claims(raw_answer, claims)
    │    ├── No unsupported claims found
    │    └── final_answer unchanged
    │
    │  Metrics computed:
    │    hallucination_rate = 0.0 (0/3 unsupported)
    │    faithfulness_score = 0.88 (mean of support scores)
    │
    ├── response.claims = [3 verified claims]
    ├── response.hallucination_stats = {rate: 0.0, faithfulness: 0.88, ...}
    ├── module_timings["hallucination"] = 850ms
        │
        ▼
[7] STAGE 5 — Citation (SKIPPED, use_citation=False)
        │
        ▼
[8] METRICS COLLECTION
    │  Merge all module metrics into response.metrics
    │  Log QueryLog to MetricsLogger
    │  If ground_truth provided → compute F1, EM, ROUGE-L
    │  module_timings["total"] = 3195ms
        │
        ▼
[9] Return RAGResponse to UI
    │  UI renders:
    │    ├── Final answer in styled box
    │    ├── Retrieved documents with scores
    │    ├── Claim-level analysis (supported/unsupported cards)
    │    ├── Metric cards (Faithfulness, Hallucination Rate, etc.)
    │    └── Pipeline timing breakdown
```

### 4.2 How the Pipeline Auto-Detects Teammate Modules

This is the critical mechanism that makes integration seamless:

```python
# In pipeline.py → _try_load_teammate_modules()

# When use_mmr=True, the pipeline tries:
try:
    from modules.mmr_reranker import MMRReranker    # ← Look for real module
    self.mmr_reranker = MMRReranker(self.config)     # ← Use it!
except ImportError:
    self.mmr_reranker = MMRRerankerStub()             # ← Fallback to pass-through

# Same for citation:
try:
    from modules.citation_generator import CitationGenerator
    self.citation_generator = CitationGenerator(self.config)
except ImportError:
    self.citation_generator = CitationGeneratorStub()
```

**What this means:**
- **Before teammate delivers:** System runs fine with stubs (pass-through, no errors)
- **After teammate drops in their file:** System automatically uses the real module
- **No other files need modification** — just add the `.py` file to `modules/`

---

## 5. What We Built (Our Contribution)

### 5.1 Dense Retrieval Module (`modules/dense_retrieval.py`)

**Purpose:** Base retrieval engine that all variants build upon.

**Mathematical Formulation:**
```
score(q, d) = cos(E(q), E(d)) = (E(q) · E(d)) / (‖E(q)‖ · ‖E(d)‖)
Retrieved set: R = top-k{d ∈ D : score(q, d)}
```

**Key Implementation Details:**
- Uses `all-MiniLM-L6-v2` sentence transformer (384-dim embeddings)
- Supports pre-computed embeddings (skip encoding) or auto-computation
- Three similarity metrics: cosine (default), dot product, euclidean
- L2-normalizes all embeddings at index time for efficient cosine via dot product
- Provides `get_all_embeddings()` method that the MMR teammate needs to compute inter-document similarity
- Complexity: O(n × d) per query for brute-force search

**Metrics Logged:** `retrieval_time_ms`, `avg_similarity`, `max_similarity`, `min_similarity`, `score_std`, `num_indexed`

### 5.2 Hallucination Detection Module (`modules/hallucination_detector.py`)

**Purpose:** Detect and remove unsupported claims from LLM-generated answers.

**Mathematical Formulation:**
```
Given answer A, extract claims C = {c₁, c₂, ..., cₘ}
For each claim cᵢ and evidence set E = {e₁, ..., eₖ}:

    support(cᵢ) = max_{eⱼ ∈ E} P(entailment | eⱼ, cᵢ)

Supported if: support(cᵢ) ≥ τ   (threshold, default 0.5)

Hallucination Rate = |{cᵢ : support(cᵢ) < τ}| / |C|
Faithfulness Score = (1/|C|) Σᵢ support(cᵢ)
```

**Three-Step Process:**

1. **Claim Extraction** — Sentence splitting with filtering (removes questions, hedges, short fragments). Each sentence becomes one `Claim` object.

2. **Claim Verification** — For each claim, creates (evidence, claim) pairs with ALL retrieved documents. Runs NLI cross-encoder (`cross-encoder/nli-deberta-v3-small`) to get entailment probability. Takes the max across evidence as the support score. Fallback to embedding cosine similarity if NLI model is unavailable.

3. **Hallucination Filtering** — Removes sentences from the answer that map to unsupported claims using fuzzy word-overlap matching. If all sentences are removed, returns original with a warning prefix.

**Metrics Logged:** `hallucination_rate`, `faithfulness_score`, `num_claims`, `num_supported`, `num_unsupported`, `pct_unsupported_claims`, `avg/min/max_support_score`, `verification_time_ms`

### 5.3 LLM Answer Generator (`modules/llm_generator.py`)

**Purpose:** Generate natural language answers from retrieved context.

**Supports Three Backends:**
- **Ollama** (default) — Local LLM, no API key needed
- **OpenAI** — Via API with configurable model
- **HuggingFace** — Via transformers pipeline
- **Extractive Fallback** — If no LLM available, returns concatenated context

**System Prompt Design:** Instructs the LLM to answer ONLY from provided context, be factual, and not make up information.

**Metrics Logged:** `generation_time_ms`, `context_length_chars`, `num_context_docs`, `answer_length_chars`, `answer_num_sentences`

### 5.4 Evaluation & Metrics (`utils/metrics.py`)

**Purpose:** Comprehensive evaluation for research-quality ablation analysis.

**Answer Quality Metrics (when ground truth provided):**
- **F1 Score** — Token-level precision × recall harmonic mean
- **Exact Match** — Binary: normalized predicted == reference
- **ROUGE-L** — LCS-based F1 between predicted and reference words
- **Token Precision & Recall** — Breakdown of F1

**Retrieval Quality Metrics (when relevant doc IDs provided):**
- **Precision@k** — Fraction of top-k that are relevant
- **Recall@k** — Fraction of relevant docs found in top-k
- **MRR** — Reciprocal rank of first relevant document
- **nDCG@k** — Normalized Discounted Cumulative Gain

**Export Formats:**
- `export_to_json()` — Full logs with comparison table
- `export_to_csv()` — Flattened rows for statistical analysis (R, Python, Excel)

### 5.5 Streamlit UI (`app.py`)

**Purpose:** Interactive web interface for querying, comparing, and running ablation experiments. Four modes described in Section 8.

### 5.6 Pipeline Orchestrator (`pipeline.py`)

**Purpose:** Connects all modules in sequence, handles conditional execution, timing, and logging.

**Key Design Decisions:**
- Auto-detects teammate modules via `try/except ImportError`
- Each module receives and enriches the same `RAGResponse` object
- All timings measured per-module for performance analysis
- Supports optional `ground_truth` and `relevant_doc_ids` for evaluation

### 5.7 Stubs (`modules/stubs.py`)

**Purpose:** No-op implementations so the system runs end-to-end without teammate modules.

- `MMRRerankerStub` — Returns documents unchanged (pass-through)
- `CitationGeneratorStub` — Returns answer unchanged (no citations)

---

## 6. Shared Contract — config.py and interfaces.py

These two files are the **bridge between our code and teammate code**. Every module on the team imports from here.

### 6.1 Shared Data Types (`config.py`)

| Type | Purpose | Key Fields |
|------|---------|------------|
| `RAGConfig` | Master configuration with ablation flags | `use_mmr`, `use_citation`, `use_hallucination_detection`, `top_k`, `mmr_lambda`, `hallucination_threshold`, `llm_provider`, etc. |
| `Document` | A single chunk in the knowledge base | `doc_id`, `content`, `metadata`, `embedding` |
| `RetrievedDocument` | Document returned by retrieval | `document`, `score`, `rank`, `mmr_score` (for MMR teammate), `citation_spans` (for Citation teammate) |
| `Claim` | An atomic claim extracted from the answer | `claim_id`, `text`, `is_supported`, `support_score`, `supporting_doc_ids`, `evidence_text` |
| `RAGResponse` | Complete pipeline output — enriched by each module | `retrieved_docs`, `mmr_reranked_docs`, `raw_answer`, `final_answer`, `claims`, `hallucination_stats`, `citations`, `citation_stats`, `metrics`, `module_timings` |
| `ABLATION_VARIANTS` | Pre-configured experiment variants | Dictionary of 5 `RAGConfig` objects |

### 6.2 Abstract Interfaces (`interfaces.py`)

| Interface | Who Implements | Methods |
|-----------|---------------|---------|
| `BaseRetriever` | Dense Retrieval (us) | `index()`, `retrieve()`, `get_metrics()` |
| `BaseReranker` | **MMR Teammate** | `rerank()`, `get_metrics()` |
| `BaseHallucinationDetector` | Hallucination Detection (us) | `extract_claims()`, `verify_claims()`, `get_metrics()` |
| `BaseCitationGenerator` | **Citation Teammate** | `generate_citations()`, `get_metrics()` |

---

## 7. How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

**Required:** `streamlit`, `numpy`, `pandas`, `sentence-transformers`, `scipy`, `scikit-learn`

**Optional (for LLM generation):**
- **Ollama** (recommended): Install from https://ollama.ai, then `ollama pull llama3`
- **OpenAI**: `pip install openai` + set `OPENAI_API_KEY`
- If no LLM is configured, the system uses extractive fallback (concatenates retrieved context)

### Start the Application

```bash
cd edu_rag
streamlit run app.py
```

### Quick Demo

1. Click **"📥 Load Sample Documents"** in the sidebar (loads 12 educational passages)
2. Select **"Comparison Mode"** from the sidebar
3. Type a question: *"What is photosynthesis?"*
4. See Vanilla RAG vs Hallucination-Free RAG side by side

---

## 8. Streamlit UI — Four Modes

### Mode 1: Single Query

Ask one question using either Vanilla RAG or Hallucination-Free RAG. Shows:
- Generated answer in styled box
- Retrieved documents with similarity scores
- Claim-level hallucination analysis (if enabled): each claim shown as supported (green) or unsupported (red) with support scores
- Raw vs filtered answer comparison
- Pipeline metrics and timing breakdown

### Mode 2: Comparison Mode

Runs the **same query** through both variants **side-by-side**. Highlights:
- Left column: Vanilla RAG answer + metrics
- Right column: Hallucination-Free RAG answer + claim analysis
- Optional ground truth input → computes F1, EM, ROUGE-L for both variants
- Comparison table showing metric differences

### Mode 3: Ablation Study

Batch experiment runner for systematic evaluation:
- Select which ablation variants to compare
- Enter multiple queries (manual or predefined set)
- Set number of runs per query (for statistical significance)
- Progress bar during execution
- Results table with per-query metrics
- Summary statistics grouped by variant (mean ± std)
- Bar chart visualization of hallucination rates
- Download results as CSV or JSON

### Mode 4: Metrics Dashboard

Accumulated view across all queries run in the session:
- Total queries logged and variants tested
- Cross-variant comparison table (aggregated means and standard deviations)
- Query timeline with per-query metrics
- Export all logs to JSON or CSV

---

## 9. Teammate Integration Guide

### 🔌 For the MMR Teammate

**What to do:** Create the file `modules/mmr_reranker.py`

**What to implement:** A class called `MMRReranker` that implements `BaseReranker`

**Step-by-step:**

1. Create `modules/mmr_reranker.py`
2. Import the required types:

```python
from interfaces import BaseReranker
from config import RAGConfig, RetrievedDocument
```

3. Implement the class:

```python
class MMRReranker(BaseReranker):
    """
    Maximal Marginal Relevance reranking for retrieval diversity.

    MMR(dᵢ) = λ · sim(q, dᵢ) - (1-λ) · max_{dⱼ ∈ S} sim(dᵢ, dⱼ)
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self.lambda_param = config.mmr_lambda  # 0.7 default
        self._last_metrics = {}

    def rerank(self, query, documents, top_n=5):
        """
        Input:
            - query: the user's question (string)
            - documents: List[RetrievedDocument] from dense retrieval
              (each has .document.embedding and .score already filled)
            - top_n: how many to return after reranking

        Output:
            - List[RetrievedDocument] reranked by MMR
            - MUST populate .mmr_score on each returned document
            - MUST populate .diversity_contribution on each returned document

        Available config parameters:
            - self.config.mmr_lambda (0.7 default, set in sidebar)
            - self.config.mmr_top_n (5 default)

        Notes:
            - Document embeddings are in doc.document.embedding (list of floats)
            - Similarity scores from dense retrieval are in doc.score
        """
        # YOUR MMR IMPLEMENTATION HERE
        pass

    def get_metrics(self):
        """Return: {"diversity_score": ..., "redundancy_reduction": ..., "mmr_lambda": ...}"""
        return self._last_metrics
```

4. **That's it.** No other files need editing. The pipeline auto-detects your module.

**How to test:** Set `use_mmr=True` in the sidebar or use "RAG + MMR" in Ablation Study mode.

---

### 🔌 For the Citation Teammate

**What to do:** Create the file `modules/citation_generator.py`

**What to implement:** A class called `CitationGenerator` that implements `BaseCitationGenerator`

**Step-by-step:**

1. Create `modules/citation_generator.py`
2. Import the required types:

```python
from interfaces import BaseCitationGenerator
from config import RAGConfig, RetrievedDocument, Claim
```

3. Implement the class:

```python
class CitationGenerator(BaseCitationGenerator):
    """
    Citation grounding: maps answer claims to source documents
    and inserts inline citation markers.
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self._last_metrics = {}

    def generate_citations(self, answer, claims, documents):
        """
        Input:
            - answer: the current final_answer string
              (may have been filtered by hallucination detector already)
            - claims: List[Claim] extracted from the answer
              (each has .text, .is_supported, .support_score, .supporting_doc_ids)
              NOTE: claims list may be EMPTY if hallucination detection is off
            - documents: List[RetrievedDocument] used as evidence
              (each has .document.doc_id, .document.content, .document.metadata)

        Output: MUST return a dict with exactly these keys:
            {
                "cited_answer": str,
                    # Answer with inline citations, e.g.,
                    # "Photosynthesis uses sunlight [1] to produce energy [2]."

                "citations": List[Dict],
                    # Each: {"claim_id": str, "doc_id": str,
                    #         "span": str, "confidence": float}

                "citation_stats": Dict[str, float],
                    # Must include: "coverage", "accuracy",
                    #                "precision", "recall"
            }

        Available config:
            - self.config.citation_min_similarity (0.6 default)
        """
        # YOUR CITATION IMPLEMENTATION HERE
        pass

    def get_metrics(self):
        """Return citation-specific metrics for logging."""
        return self._last_metrics
```

4. **That's it.** The pipeline auto-detects your module.

**How to test:** Set `use_citation=True` in the sidebar or use "RAG + Citation" in Ablation Study mode.

---

### Integration Checklist for Teammates

- [ ] File is named exactly `mmr_reranker.py` or `citation_generator.py`
- [ ] File is placed in the `modules/` directory
- [ ] Class is named exactly `MMRReranker` or `CitationGenerator`
- [ ] Class implements the correct interface from `interfaces.py`
- [ ] All imports use types from `config.py` (not custom types)
- [ ] `get_metrics()` returns a `Dict[str, float]` (for logging)
- [ ] Module does NOT modify `config.py`, `pipeline.py`, or `app.py`
- [ ] Module works independently (no dependency on the other teammate's module)

---

## 10. Ablation Study Design

### Experiment Variants

| # | Variant Name | MMR | Citation | Hallucination | What It Tests |
|---|-------------|-----|----------|---------------|---------------|
| 1 | Vanilla RAG | ❌ | ❌ | ❌ | Baseline performance |
| 2 | RAG + MMR | ✅ | ❌ | ❌ | Does diversity improve answer quality? |
| 3 | RAG + Citation | ❌ | ✅ | ❌ | Does citation grounding improve trust? |
| 4 | RAG + Hallucination | ❌ | ❌ | ✅ | Does claim verification reduce errors? |
| 5 | Full System | ✅ | ✅ | ✅ | Combined performance |

### How Ablation Flags Work in Code

```python
# config.py — toggle any combination
config = RAGConfig(
    use_mmr = True,                      # Stage 2: ON/OFF
    use_citation = True,                 # Stage 5: ON/OFF
    use_hallucination_detection = True,  # Stage 4: ON/OFF
)

# pipeline.py — conditional execution
if self.config.use_mmr and self.mmr_reranker:
    response.mmr_reranked_docs = self.mmr_reranker.rerank(...)

if self.config.use_hallucination_detection and self.hallucination_detector:
    response.claims = self.hallucination_detector.extract_claims(...)
    response.claims = self.hallucination_detector.verify_claims(...)

if self.config.use_citation and self.citation_generator:
    cit_result = self.citation_generator.generate_citations(...)
```

---

## 11. Evaluation Metrics

### Metrics Collected Per Query

| Category | Metric | When Computed | Formula |
|----------|--------|---------------|---------|
| **Retrieval** | Avg Similarity | Always | Mean cosine similarity of top-k |
| | Precision@k | If relevant_doc_ids provided | \|relevant ∩ retrieved\| / k |
| | Recall@k | If relevant_doc_ids provided | \|relevant ∩ retrieved\| / \|relevant\| |
| | MRR | If relevant_doc_ids provided | 1/rank of first relevant doc |
| | nDCG@k | If relevant_doc_ids provided | DCG / IDCG |
| **Hallucination** | Hallucination Rate | If hallucination ON | unsupported / total claims |
| | Faithfulness Score | If hallucination ON | mean(support_scores) |
| | % Unsupported Claims | If hallucination ON | (unsupported / total) × 100 |
| **Citation** | Citation Coverage | If citation ON | % claims with citations |
| | Citation Accuracy | If citation ON | % correct citations |
| **Answer Quality** | F1 Score | If ground truth provided | 2PR / (P+R) on token sets |
| | Exact Match | If ground truth provided | Normalized string equality |
| | ROUGE-L | If ground truth provided | LCS-based F1 |
| **Performance** | Total Time (ms) | Always | End-to-end latency |
| | Per-Module Timing | Always | Time per stage |

---

## 12. Technical Details

### Embedding Model

- **Model:** `all-MiniLM-L6-v2` from sentence-transformers
- **Dimension:** 384
- **Normalization:** L2-normalized at index time
- **Why this model:** Good balance of quality and speed; widely used in RAG research

### NLI Model (Hallucination Detection)

- **Model:** `cross-encoder/nli-deberta-v3-small`
- **Input:** (evidence_passage, claim) pairs
- **Output:** Logits for [contradiction, neutral, entailment]
- **We use:** Softmax → entailment probability as support score
- **Fallback:** If NLI model unavailable, uses cosine similarity between claim and evidence embeddings

### LLM Generation

- **Default:** Ollama with llama3 (local, no API key)
- **System Prompt:** Constrains LLM to answer ONLY from provided context
- **Temperature:** 0.3 (low for factual consistency)
- **Fallback:** If no LLM available, extracts first 500 chars of context

### Computational Complexity

| Module | Complexity per Query | Dominant Factor |
|--------|---------------------|-----------------|
| Dense Retrieval | O(n × d) | n = num docs, d = embedding dim |
| MMR Reranking | O(k² × d) | k = top-k, pairwise similarity |
| LLM Generation | O(context_len) | Depends on LLM backend |
| Hallucination Detection | O(m × k) | m = num claims, k = num evidence docs |
| Citation Generation | O(m × k) | m = num claims, k = num docs |

---

## 13. Dependencies

### Required

```
streamlit>=1.30.0              # Web UI framework
numpy>=1.24.0                  # Numerical computation
pandas>=2.0.0                  # Data manipulation and display
sentence-transformers>=2.2.0   # Embedding models + NLI cross-encoder
scipy>=1.10.0                  # Softmax for NLI score processing
scikit-learn>=1.3.0            # Utility functions
requests>=2.31.0               # HTTP client for Ollama API
```

### Optional

```
openai                         # For OpenAI LLM backend
transformers                   # For HuggingFace LLM backend
torch                          # Required by sentence-transformers
```

### LLM Backend Setup

**Ollama (Recommended for local use):**
```bash
# Install: https://ollama.ai
ollama pull llama3
# The app connects to localhost:11434 automatically
```

**OpenAI:**
```bash
pip install openai
export OPENAI_API_KEY="sk-..."
# Change LLM Provider to "openai" in sidebar
```

---

## License

This project is developed as part of an academic research project for educational purposes.