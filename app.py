"""
═══════════════════════════════════════════════════════════════════════════════
 EduRAG — Streamlit UI
═══════════════════════════════════════════════════════════════════════════════
 Three tabs:
   1. Ask — Query your documents with any RAG variant
   2. Compare — Side-by-side Vanilla vs Full System
   3. Ablation — Run your questions through selected variants, see metrics
═══════════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import numpy as np
import pandas as pd
import time
import json
import os
import sys
import io
import logging
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RAGConfig, Document, RAGResponse, ABLATION_VARIANTS
from pipeline import RAGPipeline
from utils.metrics import MetricsLogger, QueryLog

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="EduRAG", page_icon="📚", layout="wide")

st.markdown("""
<style>
    .metric-box {
        background: #1a1f2e; border: 1px solid #2d3748; border-radius: 10px;
        padding: 16px; margin: 6px 0;
    }
    .metric-box h4 { color: #8b9dc3; font-size: 0.8rem; margin: 0; }
    .metric-box .val { color: #e2e8f0; font-size: 1.5rem; font-weight: 700; }
    .supported { background: #0a2e1a; border-left: 4px solid #22c55e;
        padding: 10px 14px; border-radius: 6px; margin: 6px 0; }
    .unsupported { background: #2e0a0a; border-left: 4px solid #ef4444;
        padding: 10px 14px; border-radius: 6px; margin: 6px 0; }
    .answer-box { background: #1a1f2e; border: 1px solid #2d3748;
        border-radius: 10px; padding: 18px; margin: 10px 0; line-height: 1.7; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────────────────────────────────────
if "documents" not in st.session_state:
    st.session_state.documents = []
if "metrics_logger" not in st.session_state:
    st.session_state.metrics_logger = MetricsLogger("edu_rag")
if "parse_errors" not in st.session_state:
    st.session_state.parse_errors = []


# ─────────────────────────────────────────────────────────────────────────────
# Document Parsing
# ─────────────────────────────────────────────────────────────────────────────
def chunk_text(text: str, filename: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """Split text into overlapping chunks and return Document objects."""
    docs = []
    text = text.strip()
    if not text:
        return docs

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    full_text = "\n".join(paragraphs)

    if not full_text:
        return docs

    start = 0
    i = 0
    while start < len(full_text):
        end = min(start + chunk_size, len(full_text))
        chunk = full_text[start:end].strip()
        if chunk:
            docs.append(Document(
                doc_id=f"{filename}_{i:03d}",
                content=chunk,
                metadata={"source": filename, "chunk": i},
            ))
            i += 1
        start += chunk_size - overlap
    return docs


def parse_pdf(file_obj, filename: str) -> list:
    """Parse PDF file. Tries PyPDF2 first, then PyMuPDF."""
    # Reset file pointer
    file_obj.seek(0)
    raw_bytes = file_obj.read()

    # ── Try PyPDF2 ──
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(raw_bytes))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text and text.strip():
                pages.append(text.strip())
        if pages:
            full_text = "\n\n".join(pages)
            logger.info(f"PyPDF2: extracted {len(pages)} pages, {len(full_text)} chars from {filename}")
            return chunk_text(full_text, filename)
    except ImportError:
        logger.info("PyPDF2 not installed, trying PyMuPDF...")
    except Exception as e:
        logger.warning(f"PyPDF2 failed on {filename}: {e}")

    # ── Try PyMuPDF (fitz) ──
    try:
        import fitz
        doc = fitz.open(stream=raw_bytes, filetype="pdf")
        pages = []
        for page in doc:
            text = page.get_text()
            if text and text.strip():
                pages.append(text.strip())
        doc.close()
        if pages:
            full_text = "\n\n".join(pages)
            logger.info(f"PyMuPDF: extracted {len(pages)} pages, {len(full_text)} chars from {filename}")
            return chunk_text(full_text, filename)
    except ImportError:
        logger.info("PyMuPDF not installed either")
    except Exception as e:
        logger.warning(f"PyMuPDF failed on {filename}: {e}")

    # ── Try pdfplumber ──
    try:
        import pdfplumber
        pdf = pdfplumber.open(io.BytesIO(raw_bytes))
        pages = []
        for page in pdf.pages:
            text = page.extract_text()
            if text and text.strip():
                pages.append(text.strip())
        pdf.close()
        if pages:
            full_text = "\n\n".join(pages)
            logger.info(f"pdfplumber: extracted {len(pages)} pages, {len(full_text)} chars from {filename}")
            return chunk_text(full_text, filename)
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"pdfplumber failed on {filename}: {e}")

    # All failed
    return []


def parse_docx(file_obj, filename: str) -> list:
    """Parse DOCX/DOC file."""
    file_obj.seek(0)

    try:
        from docx import Document as DocxDocument
        doc_file = DocxDocument(file_obj)
        paragraphs = [p.text.strip() for p in doc_file.paragraphs if p.text.strip()]
        if paragraphs:
            full_text = "\n".join(paragraphs)
            logger.info(f"python-docx: extracted {len(paragraphs)} paragraphs from {filename}")
            return chunk_text(full_text, filename)
    except ImportError:
        logger.warning("python-docx not installed")
    except Exception as e:
        logger.warning(f"python-docx failed on {filename}: {e}")

    return []


def parse_uploaded_files(uploaded_files) -> tuple:
    """
    Parse all uploaded files into Document objects.
    Returns (docs_list, errors_list).
    """
    docs = []
    errors = []

    for f in uploaded_files:
        name = f.name
        try:
            if name.lower().endswith(".pdf"):
                result = parse_pdf(f, name)
                if result:
                    docs.extend(result)
                else:
                    errors.append(f"❌ **{name}**: No text extracted. Install `pip install PyPDF2` or check if the PDF has selectable text (scanned PDFs need OCR).")

            elif name.lower().endswith((".docx", ".doc")):
                result = parse_docx(f, name)
                if result:
                    docs.extend(result)
                else:
                    errors.append(f"❌ **{name}**: No text extracted. Install `pip install python-docx`.")

            elif name.lower().endswith(".txt"):
                content = f.read().decode("utf-8", errors="ignore")
                result = chunk_text(content, name)
                docs.extend(result)

            elif name.lower().endswith(".csv"):
                f.seek(0)
                df = pd.read_csv(f)
                if "content" in df.columns:
                    for i, row in df.iterrows():
                        docs.append(Document(
                            doc_id=row.get("doc_id", f"csv_{i:04d}"),
                            content=str(row["content"]),
                            metadata={
                                "source": row.get("source", name),
                                "subject": row.get("subject", ""),
                                "topic": row.get("topic", ""),
                            },
                        ))
                else:
                    errors.append(f"⚠️ **{name}**: CSV must have a 'content' column. Found columns: {list(df.columns)}")

            elif name.lower().endswith(".json"):
                f.seek(0)
                data = json.loads(f.read())
                if isinstance(data, list):
                    for i, item in enumerate(data):
                        if isinstance(item, dict) and "content" in item:
                            docs.append(Document(
                                doc_id=item.get("doc_id", f"json_{i:04d}"),
                                content=item["content"],
                                metadata=item.get("metadata", {"source": name}),
                            ))
                else:
                    errors.append(f"⚠️ **{name}**: JSON must be a list of objects with 'content' key.")

            else:
                errors.append(f"⚠️ **{name}**: Unsupported file type.")

        except Exception as e:
            errors.append(f"❌ **{name}**: {str(e)}")
            logger.error(f"Error parsing {name}: {traceback.format_exc()}")

    return docs, errors


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Helper
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_cached_pipeline(_docs_hash: str, config_dict: dict):
    """Cache pipeline to avoid re-indexing on every interaction."""
    config = RAGConfig(**config_dict)
    pipeline = RAGPipeline(config)
    return pipeline


def build_pipeline(config: RAGConfig) -> RAGPipeline:
    """Build pipeline and index documents."""
    pipeline = RAGPipeline(config, st.session_state.metrics_logger)
    pipeline.index_documents(st.session_state.documents)
    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Display Helpers
# ─────────────────────────────────────────────────────────────────────────────
def metric_card(label, value, sub=""):
    st.markdown(f"""
    <div class="metric-box">
        <h4>{label}</h4>
        <div class="val">{value}</div>
        <div style="color:#64748b;font-size:0.7rem;">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def show_claims(claims):
    for i, c in enumerate(claims):
        css = "supported" if c.is_supported else "unsupported"
        icon = "✅" if c.is_supported else "❌"
        score = f"{c.support_score:.3f}" if c.support_score is not None else "N/A"
        st.markdown(f"""
        <div class="{css}">
            <strong>{icon} Claim {i+1}</strong> &nbsp;
            <span style="color:#94a3b8;font-size:0.8rem;">Score: {score}</span>
            <div style="color:#e2e8f0;margin-top:4px;">{c.text}</div>
        </div>
        """, unsafe_allow_html=True)


def show_retrieved_docs(docs):
    for d in docs:
        src = d.document.metadata.get("source", d.document.doc_id)
        st.markdown(f"**{src}** — similarity: `{d.score:.4f}`")
        st.caption(d.document.content[:300])
        st.markdown("---")


def show_citations(citations, documents):
    if not citations:
        return
    st.markdown("**References:**")
    shown = set()
    for c in citations:
        num = c.get("citation_num", 0)
        doc_id = c.get("doc_id", "")
        if num not in shown:
            source = doc_id
            for doc in documents:
                if doc.document.doc_id == doc_id:
                    source = doc.document.metadata.get("source", doc_id)
                    break
            st.markdown(f"[{num}] {source} (`{doc_id}`)")
            shown.add(num)


def get_sidebar_config() -> dict:
    """Return common config kwargs from sidebar settings."""
    return dict(
        top_k=top_k,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_temperature=temperature,
        hallucination_threshold=hall_threshold,
        mmr_lambda=mmr_lambda,
    )


def apply_sidebar_config(config: RAGConfig):
    """Apply sidebar settings to a config object."""
    config.top_k = top_k
    config.llm_provider = llm_provider
    config.llm_model = llm_model
    config.llm_temperature = temperature
    config.hallucination_threshold = hall_threshold
    config.mmr_lambda = mmr_lambda
    return config


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📚 EduRAG")
    st.markdown("---")

    # ── Upload Documents ──
    st.markdown("### 📄 Upload Documents")
    uploaded = st.file_uploader(
        "Upload your educational content",
        type=["txt", "csv", "json", "pdf", "docx", "doc"],
        accept_multiple_files=True,
        help="Supports: PDF, DOCX, TXT, CSV, JSON",
    )

    if uploaded:
        # Parse files on every upload change
        with st.spinner("Parsing documents..."):
            new_docs, parse_errors = parse_uploaded_files(uploaded)

        if new_docs:
            st.session_state.documents = new_docs
            st.session_state.parse_errors = parse_errors
            st.success(f"✅ Loaded **{len(new_docs)}** chunks from {len(uploaded)} file(s)")
        else:
            st.session_state.documents = []
            st.session_state.parse_errors = parse_errors

        # Show parse errors
        for err in parse_errors:
            st.error(err)
    else:
        # File uploader cleared
        if st.session_state.documents:
            st.session_state.documents = []

    num_docs = len(st.session_state.documents)
    if num_docs > 0:
        st.info(f"📊 **{num_docs}** document chunks ready")
        with st.expander("Preview documents"):
            for doc in st.session_state.documents[:8]:
                st.caption(f"**{doc.doc_id}**: {doc.content[:120]}...")
            if num_docs > 8:
                st.caption(f"... and {num_docs - 8} more")

    st.markdown("---")

    # ── LLM Settings ──
    st.markdown("### ⚙️ LLM Settings")
    llm_provider = st.selectbox("Provider", ["ollama", "openai", "huggingface"])
    llm_model = st.text_input("Model", value="llama3")
    top_k = st.slider("Top-K retrieval", 1, 20, 5)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)

    st.markdown("---")
    st.markdown("### 🎯 Thresholds")
    hall_threshold = st.slider("Hallucination threshold", 0.0, 1.0, 0.5, 0.05)
    mmr_lambda = st.slider("MMR lambda", 0.0, 1.0, 0.7, 0.05,
                           help="Higher = more relevance, Lower = more diversity")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT — Title + Tabs (always visible)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# 📚 EduRAG — Education RAG System")

# Always show tabs — show a message inside each tab if no documents
has_docs = len(st.session_state.documents) > 0

tab_ask, tab_compare, tab_ablation = st.tabs(["🔍 Ask", "⚔️ Compare", "🧪 Ablation Study"])

NO_DOCS_MSG = """
**No documents loaded yet.** Upload files using the sidebar to get started.

Supported: **PDF**, **DOCX**, **TXT**, **CSV**, **JSON**
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1: Ask
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_ask:
    if not has_docs:
        st.warning(NO_DOCS_MSG)
    else:
        st.markdown("Choose a RAG variant and ask your question.")

        variant = st.radio(
            "RAG Variant",
            ["Vanilla RAG", "RAG + MMR", "RAG + Hallucination Detection",
             "RAG + Citation", "Full System"],
            horizontal=True,
            key="ask_variant",
        )

        query = st.text_input("Your question", placeholder="Type your question here...", key="ask_q")

        if query:
            config = apply_sidebar_config(ABLATION_VARIANTS[variant])

            with st.spinner(f"Running {variant}..."):
                try:
                    pipeline = build_pipeline(config)
                    response = pipeline.query(query)
                except Exception as e:
                    st.error(f"Pipeline error: {e}")
                    st.stop()

            # ── Answer ──
            st.markdown(f'<div class="answer-box">{response.final_answer}</div>',
                        unsafe_allow_html=True)

            # ── Metrics Row ──
            cols = st.columns(4)
            with cols[0]:
                metric_card("Retrieval", f"{response.metrics.get('avg_similarity', 0):.4f}",
                            "avg cosine sim")
            with cols[1]:
                metric_card("Total Time", f"{response.module_timings.get('total', 0):.0f}ms",
                            "end-to-end")
            with cols[2]:
                hr = response.hallucination_stats.get("hallucination_rate")
                metric_card("Halluc. Rate",
                            f"{hr:.1%}" if hr is not None else "—",
                            "lower is better" if hr is not None else "detection OFF")
            with cols[3]:
                cc = response.citation_stats.get("citation_coverage")
                metric_card("Citation Cov.",
                            f"{cc:.1%}" if cc is not None else "—",
                            "higher is better" if cc is not None else "citation OFF")

            # ── Hallucination Claims ──
            if response.claims:
                with st.expander("🔬 Claim-Level Analysis", expanded=True):
                    show_claims(response.claims)

            # ── Citations ──
            if response.citations:
                with st.expander("📎 Citations", expanded=True):
                    show_citations(response.citations, response.get_active_docs())

            # ── Retrieved Documents ──
            with st.expander("📄 Retrieved Documents"):
                show_retrieved_docs(response.get_active_docs())

            # ── Raw Metrics ──
            with st.expander("📊 Full Metrics"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Module Timings:**")
                    st.json(response.module_timings)
                with col2:
                    st.markdown("**All Metrics:**")
                    st.json(response.metrics)
                if response.hallucination_stats:
                    st.markdown("**Hallucination Stats:**")
                    st.json(response.hallucination_stats)
                if response.citation_stats:
                    st.markdown("**Citation Stats:**")
                    st.json(response.citation_stats)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2: Compare
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_compare:
    if not has_docs:
        st.warning(NO_DOCS_MSG)
    else:
        st.markdown("Same question through **Vanilla RAG** vs **Full System** side-by-side.")

        cmp_query = st.text_input("Your question", placeholder="Type your question here...",
                                  key="cmp_q")

        ground_truth = st.text_area(
            "Ground truth answer (optional — enables F1, ROUGE-L, Exact Match)",
            placeholder="Paste the correct answer here if you have one...",
            height=80, key="cmp_gt",
        )

        if cmp_query:
            gt = ground_truth.strip() if ground_truth.strip() else None

            config_vanilla = RAGConfig(
                use_mmr=False, use_citation=False, use_hallucination_detection=False,
                **get_sidebar_config(),
            )
            config_full = RAGConfig(
                use_mmr=True, use_citation=True, use_hallucination_detection=True,
                **get_sidebar_config(),
            )

            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown("### 📦 Vanilla RAG")
                with st.spinner("Running..."):
                    pipe_v = build_pipeline(config_vanilla)
                    resp_v = pipe_v.query(cmp_query, ground_truth=gt)
                st.markdown(f'<div class="answer-box">{resp_v.final_answer}</div>',
                            unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    metric_card("Avg Similarity",
                                f"{resp_v.metrics.get('avg_similarity', 0):.4f}")
                with c2:
                    metric_card("Time", f"{resp_v.module_timings.get('total', 0):.0f}ms")

            with col_right:
                st.markdown("### 🛡️ Full System")
                st.caption("MMR + Hallucination Detection + Citation")
                with st.spinner("Running..."):
                    pipe_f = build_pipeline(config_full)
                    resp_f = pipe_f.query(cmp_query, ground_truth=gt)
                st.markdown(f'<div class="answer-box">{resp_f.final_answer}</div>',
                            unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)
                with c1:
                    metric_card("Faithfulness",
                                f"{resp_f.hallucination_stats.get('faithfulness_score', 0):.1%}")
                with c2:
                    metric_card("Halluc. Rate",
                                f"{resp_f.hallucination_stats.get('hallucination_rate', 0):.1%}")
                with c3:
                    metric_card("Time", f"{resp_f.module_timings.get('total', 0):.0f}ms")

                if resp_f.claims:
                    with st.expander("🔬 Claims"):
                        show_claims(resp_f.claims)
                if resp_f.citations:
                    with st.expander("📎 Citations"):
                        show_citations(resp_f.citations, resp_f.get_active_docs())

            # ── Answer Quality (if ground truth) ──
            if gt:
                st.markdown("---")
                st.markdown("### 📊 Answer Quality Comparison")
                ml = st.session_state.metrics_logger
                aq_v = ml.compute_answer_quality(resp_v.final_answer, gt)
                aq_f = ml.compute_answer_quality(resp_f.final_answer, gt)

                comp_df = pd.DataFrame({
                    "Metric": ["F1 Score", "Exact Match", "ROUGE-L"],
                    "Vanilla RAG": [f"{aq_v.get('f1_score',0):.4f}",
                                    f"{aq_v.get('exact_match',0):.0f}",
                                    f"{aq_v.get('rouge_l',0):.4f}"],
                    "Full System": [f"{aq_f.get('f1_score',0):.4f}",
                                    f"{aq_f.get('exact_match',0):.0f}",
                                    f"{aq_f.get('rouge_l',0):.4f}"],
                })
                st.dataframe(comp_df, use_container_width=True, hide_index=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3: Ablation Study
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_ablation:
    if not has_docs:
        st.warning(NO_DOCS_MSG)
    else:
        st.markdown("Enter your questions, pick variants, and run the experiment.")

        # ── Step 1: Questions ──
        st.markdown("### Step 1: Enter your questions (one per line)")
        questions_text = st.text_area(
            "Questions",
            placeholder="Type each question on a new line...",
            height=150,
            key="abl_questions",
        )
        questions = [q.strip() for q in questions_text.strip().split("\n") if q.strip()]

        if questions:
            st.success(f"📝 {len(questions)} question(s)")

        # ── Step 2: Variants ──
        st.markdown("### Step 2: Select variants to compare")
        selected_variants = st.multiselect(
            "Variants",
            list(ABLATION_VARIANTS.keys()),
            default=["Vanilla RAG", "Full System"],
            key="abl_variants",
        )

        # ── Step 3: Run ──
        st.markdown("### Step 3: Run")
        num_runs = st.slider("Runs per question", 1, 5, 1, key="abl_runs")

        ready = len(questions) > 0 and len(selected_variants) > 0

        if not ready:
            if not questions:
                st.info("☝️ Enter at least one question above.")
            if not selected_variants:
                st.info("☝️ Select at least one variant.")

        if st.button("🚀 Run Ablation Study", type="primary", disabled=not ready,
                     use_container_width=True):

            st.session_state.metrics_logger = MetricsLogger("ablation")

            total_steps = len(selected_variants) * len(questions) * num_runs
            progress = st.progress(0)
            status = st.empty()
            step = 0
            results = []

            for variant_name in selected_variants:
                config = apply_sidebar_config(ABLATION_VARIANTS[variant_name])
                pipeline = build_pipeline(config)

                for q in questions:
                    for run in range(num_runs):
                        step += 1
                        progress.progress(step / total_steps)
                        status.text(f"{variant_name} | {q[:50]}... | Run {run+1}/{num_runs}")

                        try:
                            resp = pipeline.query(q)
                        except Exception as e:
                            logger.error(f"Query failed: {e}")
                            continue

                        row = {
                            "Variant": variant_name,
                            "Question": q,
                            "Run": run + 1,
                            "Avg Similarity": round(resp.metrics.get("avg_similarity", 0), 4),
                            "Total Time (ms)": round(resp.module_timings.get("total", 0), 1),
                            "Answer Length": len(resp.final_answer),
                        }

                        if resp.hallucination_stats:
                            row["Halluc. Rate"] = round(
                                resp.hallucination_stats.get("hallucination_rate", 0), 4)
                            row["Faithfulness"] = round(
                                resp.hallucination_stats.get("faithfulness_score", 0), 4)
                            row["Claims"] = int(
                                resp.hallucination_stats.get("num_claims", 0))
                            row["Supported"] = int(
                                resp.hallucination_stats.get("num_supported", 0))
                        else:
                            row["Halluc. Rate"] = None
                            row["Faithfulness"] = None
                            row["Claims"] = None
                            row["Supported"] = None

                        if resp.citation_stats:
                            row["Cit. Coverage"] = round(
                                resp.citation_stats.get("citation_coverage", 0), 4)
                            row["Cit. Confidence"] = round(
                                resp.citation_stats.get("avg_confidence", 0), 4)
                        else:
                            row["Cit. Coverage"] = None
                            row["Cit. Confidence"] = None

                        row["MMR Active"] = resp.mmr_reranked_docs is not None
                        results.append(row)

            progress.progress(1.0)
            status.text("✅ Done!")

            if not results:
                st.error("No results generated. Check your LLM settings.")
            else:
                st.markdown("---")
                df = pd.DataFrame(results)

                # ── Summary ──
                st.markdown("### 📊 Summary by Variant")
                num_cols = ["Avg Similarity", "Total Time (ms)", "Answer Length",
                            "Halluc. Rate", "Faithfulness", "Cit. Coverage"]
                existing = [c for c in num_cols if c in df.columns]
                summary = df.groupby("Variant")[existing].agg(["mean", "std"]).round(4)
                st.dataframe(summary, use_container_width=True)

                # ── Charts ──
                hr_data = df.dropna(subset=["Halluc. Rate"])
                if not hr_data.empty:
                    st.markdown("### Hallucination Rate by Variant")
                    st.bar_chart(hr_data.groupby("Variant")["Halluc. Rate"].mean())

                f_data = df.dropna(subset=["Faithfulness"])
                if not f_data.empty:
                    st.markdown("### Faithfulness Score by Variant")
                    st.bar_chart(f_data.groupby("Variant")["Faithfulness"].mean())

                # ── Full Table ──
                st.markdown("### Full Results")
                st.dataframe(df, use_container_width=True, hide_index=True)

                # ── Export ──
                st.markdown("### 💾 Export")
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button("📥 CSV", df.to_csv(index=False),
                                       "ablation_results.csv", "text/csv",
                                       use_container_width=True)
                with col2:
                    st.download_button("📥 JSON",
                                       json.dumps(results, indent=2, default=str),
                                       "ablation_results.json", "application/json",
                                       use_container_width=True)