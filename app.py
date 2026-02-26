"""
═══════════════════════════════════════════════════════════════════════════════
 Education RAG System — Streamlit UI
═══════════════════════════════════════════════════════════════════════════════
 Features:
   1. Upload & index educational documents (PDF, TXT, CSV)
   2. Query with Vanilla RAG or Hallucination-Free RAG
   3. Side-by-side comparison mode
   4. Detailed metrics dashboard
   5. Ablation study runner
   6. Export results for research
═══════════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import numpy as np
import pandas as pd
import time
import json
import os
import sys
import logging
from typing import List, Dict, Optional

# ── Setup path ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RAGConfig, Document, RAGResponse, ABLATION_VARIANTS
from pipeline import RAGPipeline
from modules.hallucination_detector import HallucinationDetector
from utils.metrics import MetricsLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduRAG — Education-Focused RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom Styling
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main theme */
    .stApp { background-color: #0e1117; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #0f1318 100%);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); border-color: #4a9eff; }
    .metric-card h3 { color: #8b9dc3; font-size: 0.85rem; margin-bottom: 4px; }
    .metric-card .value { color: #e2e8f0; font-size: 1.8rem; font-weight: 700; }
    .metric-card .subtext { color: #64748b; font-size: 0.75rem; }

    /* Claim cards */
    .claim-supported {
        background: #0a2e1a; border-left: 4px solid #22c55e;
        padding: 12px 16px; border-radius: 8px; margin: 8px 0;
    }
    .claim-unsupported {
        background: #2e0a0a; border-left: 4px solid #ef4444;
        padding: 12px 16px; border-radius: 8px; margin: 8px 0;
    }
    .claim-score { font-size: 0.8rem; color: #94a3b8; }

    /* Answer box */
    .answer-box {
        background: #1a1f2e; border: 1px solid #2d3748;
        border-radius: 12px; padding: 20px; margin: 12px 0;
        line-height: 1.7;
    }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #4a9eff22, transparent);
        border-left: 3px solid #4a9eff;
        padding: 8px 16px; margin: 16px 0 12px 0;
        border-radius: 0 8px 8px 0;
        font-weight: 600; color: #e2e8f0;
    }

    /* Hide streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Comparison mode */
    .vs-badge {
        background: #4a9eff; color: white; padding: 4px 12px;
        border-radius: 20px; font-size: 0.75rem; font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────────────────────────────────────
def init_session_state():
    defaults = {
        "documents": [],
        "pipelines": {},
        "metrics_logger": MetricsLogger("edu_rag_ablation"),
        "query_history": [],
        "indexed": False,
        "sample_loaded": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────
def load_sample_documents() -> List[Document]:
    """Load built-in sample educational documents for demo."""
    samples = [
        Document(
            doc_id="bio_001",
            content="Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize nutrients from carbon dioxide and water. Photosynthesis in plants generally involves the green pigment chlorophyll and generates oxygen as a by-product. The process primarily takes place in the leaves of plants.",
            metadata={"source": "Biology Textbook Ch.5", "subject": "Biology", "topic": "Photosynthesis"},
        ),
        Document(
            doc_id="bio_002",
            content="The light-dependent reactions of photosynthesis occur in the thylakoid membranes. These reactions use light energy to produce ATP and NADPH, which are then used in the Calvin cycle. Water molecules are split during this process, releasing oxygen as a byproduct through photolysis.",
            metadata={"source": "Biology Textbook Ch.5", "subject": "Biology", "topic": "Light Reactions"},
        ),
        Document(
            doc_id="bio_003",
            content="The Calvin cycle, also known as the light-independent reactions, takes place in the stroma of chloroplasts. During this cycle, carbon dioxide is fixed into organic molecules using the ATP and NADPH produced during the light-dependent reactions. The enzyme RuBisCO catalyzes the first step of carbon fixation.",
            metadata={"source": "Biology Textbook Ch.5", "subject": "Biology", "topic": "Calvin Cycle"},
        ),
        Document(
            doc_id="phy_001",
            content="Newton's First Law of Motion states that an object at rest stays at rest, and an object in motion stays in motion with the same speed and direction, unless acted upon by an unbalanced force. This is also known as the Law of Inertia. Inertia is the tendency of an object to resist changes in its state of motion.",
            metadata={"source": "Physics Textbook Ch.3", "subject": "Physics", "topic": "Newton's Laws"},
        ),
        Document(
            doc_id="phy_002",
            content="Newton's Second Law of Motion states that the acceleration of an object is directly proportional to the net force acting on it and inversely proportional to its mass. This is expressed mathematically as F = ma, where F is force, m is mass, and a is acceleration. This law quantifies the relationship between force and motion.",
            metadata={"source": "Physics Textbook Ch.3", "subject": "Physics", "topic": "Newton's Laws"},
        ),
        Document(
            doc_id="phy_003",
            content="Newton's Third Law of Motion states that for every action, there is an equal and opposite reaction. When one body exerts a force on a second body, the second body simultaneously exerts a force equal in magnitude and opposite in direction on the first body. These force pairs are called action-reaction pairs.",
            metadata={"source": "Physics Textbook Ch.3", "subject": "Physics", "topic": "Newton's Laws"},
        ),
        Document(
            doc_id="chem_001",
            content="The periodic table organizes elements by increasing atomic number and groups them by similar chemical properties. Elements in the same group (vertical column) have the same number of valence electrons, which determines their chemical behavior. The table has 18 groups and 7 periods.",
            metadata={"source": "Chemistry Textbook Ch.2", "subject": "Chemistry", "topic": "Periodic Table"},
        ),
        Document(
            doc_id="chem_002",
            content="Chemical bonds form when atoms share or transfer electrons. Covalent bonds involve sharing of electron pairs between atoms, while ionic bonds result from the electrostatic attraction between positively and negatively charged ions. The type of bond depends on the electronegativity difference between the atoms involved.",
            metadata={"source": "Chemistry Textbook Ch.4", "subject": "Chemistry", "topic": "Chemical Bonding"},
        ),
        Document(
            doc_id="math_001",
            content="The Pythagorean theorem states that in a right triangle, the square of the length of the hypotenuse equals the sum of the squares of the other two sides. This is expressed as a² + b² = c², where c is the hypotenuse. This theorem is fundamental to Euclidean geometry and has numerous practical applications.",
            metadata={"source": "Mathematics Textbook Ch.7", "subject": "Mathematics", "topic": "Geometry"},
        ),
        Document(
            doc_id="math_002",
            content="Calculus is the mathematical study of continuous change. Differential calculus concerns instantaneous rates of change and slopes of curves, while integral calculus concerns accumulation of quantities and areas under curves. The fundamental theorem of calculus links these two branches.",
            metadata={"source": "Mathematics Textbook Ch.12", "subject": "Mathematics", "topic": "Calculus"},
        ),
        Document(
            doc_id="hist_001",
            content="The Industrial Revolution began in Britain in the late 18th century and transformed manufacturing processes from hand production to machine manufacturing. Key innovations included the steam engine, spinning jenny, and power loom. This revolution led to urbanization, changes in labor patterns, and significant economic growth.",
            metadata={"source": "History Textbook Ch.9", "subject": "History", "topic": "Industrial Revolution"},
        ),
        Document(
            doc_id="cs_001",
            content="Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. Supervised learning uses labeled training data, while unsupervised learning finds hidden patterns in unlabeled data. Deep learning uses neural networks with multiple layers.",
            metadata={"source": "CS Textbook Ch.15", "subject": "Computer Science", "topic": "Machine Learning"},
        ),
    ]
    return samples


def parse_uploaded_file(uploaded_file) -> List[Document]:
    """Parse uploaded file into Document objects."""
    docs = []
    filename = uploaded_file.name

    if filename.endswith(".txt"):
        content = uploaded_file.read().decode("utf-8")
        # Split into chunks of ~500 chars
        chunks = [content[i:i+500] for i in range(0, len(content), 450)]
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                docs.append(Document(
                    doc_id=f"{filename}_{i:03d}",
                    content=chunk.strip(),
                    metadata={"source": filename, "chunk": i},
                ))

    elif filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        # Assume columns: content (required), optional: doc_id, source, subject
        if "content" in df.columns:
            for i, row in df.iterrows():
                docs.append(Document(
                    doc_id=row.get("doc_id", f"csv_{i:04d}"),
                    content=str(row["content"]),
                    metadata={
                        "source": row.get("source", filename),
                        "subject": row.get("subject", ""),
                        "topic": row.get("topic", ""),
                    },
                ))
        else:
            st.warning("CSV must have a 'content' column.")

    elif filename.endswith(".json"):
        data = json.loads(uploaded_file.read())
        if isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict) and "content" in item:
                    docs.append(Document(
                        doc_id=item.get("doc_id", f"json_{i:04d}"),
                        content=item["content"],
                        metadata=item.get("metadata", {"source": filename}),
                    ))

    return docs


def get_or_create_pipeline(variant_name: str, config: RAGConfig) -> RAGPipeline:
    """Get cached pipeline or create new one."""
    if variant_name not in st.session_state.pipelines:
        pipeline = RAGPipeline(config, st.session_state.metrics_logger)
        if st.session_state.documents:
            pipeline.index_documents(st.session_state.documents)
        st.session_state.pipelines[variant_name] = pipeline
    return st.session_state.pipelines[variant_name]


def render_metric_card(label: str, value: str, subtext: str = "", delta: str = ""):
    """Render a styled metric card."""
    delta_html = f'<div style="color: {"#22c55e" if "+" in delta or "↑" in delta else "#ef4444"}; font-size: 0.8rem;">{delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="metric-card">
        <h3>{label}</h3>
        <div class="value">{value}</div>
        {delta_html}
        <div class="subtext">{subtext}</div>
    </div>
    """, unsafe_allow_html=True)


def render_claim_card(claim, idx: int):
    """Render a single claim with support status."""
    css_class = "claim-supported" if claim.is_supported else "claim-unsupported"
    icon = "✅" if claim.is_supported else "❌"
    score = f"{claim.support_score:.3f}" if claim.support_score is not None else "N/A"

    st.markdown(f"""
    <div class="{css_class}">
        <strong>{icon} Claim {idx + 1}</strong>
        <div style="margin: 6px 0; color: #e2e8f0;">{claim.text}</div>
        <div class="claim-score">
            Support Score: {score} | Threshold: {st.session_state.get("hall_threshold", 0.5)} |
            Evidence: {claim.evidence_text[:100] + '...' if claim.evidence_text else 'None'}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📚 EduRAG System")
    st.markdown("*Research-Grade RAG with Ablation*")
    st.markdown("---")

    # ── Mode Selection ──
    mode = st.radio(
        "🔧 Mode",
        ["Single Query", "Comparison Mode", "Ablation Study", "Metrics Dashboard"],
        index=0,
    )

    st.markdown("---")

    # ── Document Management ──
    st.markdown("### 📄 Knowledge Base")

    if st.button("📥 Load Sample Documents", use_container_width=True):
        st.session_state.documents = load_sample_documents()
        st.session_state.pipelines = {}  # Reset pipelines
        st.session_state.indexed = False
        st.session_state.sample_loaded = True
        st.success(f"Loaded {len(st.session_state.documents)} sample documents")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["txt", "csv", "json"],
        accept_multiple_files=True,
        help="Upload educational content. CSV must have 'content' column.",
    )

    if uploaded_files:
        new_docs = []
        for f in uploaded_files:
            new_docs.extend(parse_uploaded_file(f))
        if new_docs:
            st.session_state.documents.extend(new_docs)
            st.session_state.pipelines = {}
            st.session_state.indexed = False
            st.success(f"Added {len(new_docs)} document chunks")

    # Show document count
    num_docs = len(st.session_state.documents)
    st.info(f"📊 {num_docs} documents in knowledge base")

    if num_docs > 0:
        with st.expander("Preview Documents"):
            for doc in st.session_state.documents[:5]:
                st.markdown(f"**{doc.doc_id}** ({doc.metadata.get('subject', 'N/A')})")
                st.caption(doc.content[:150] + "...")

    st.markdown("---")

    # ── Configuration ──
    st.markdown("### ⚙️ Configuration")

    llm_provider = st.selectbox("LLM Provider", ["ollama", "openai", "huggingface"], index=0)
    llm_model = st.text_input("Model Name", value="llama3")
    top_k = st.slider("Top-K Retrieval", 1, 20, 5)
    temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.3, 0.1)

    st.markdown("---")
    st.markdown("### 🎯 Hallucination Settings")
    hall_threshold = st.slider("Support Threshold", 0.0, 1.0, 0.5, 0.05)
    st.session_state["hall_threshold"] = hall_threshold

    st.markdown("---")

    # ── Module Toggle Status ──
    st.markdown("### 🔌 Module Status")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("✅ Dense Retrieval")
        st.markdown("✅ LLM Generator")
    with col2:
        # Check if real modules are available
        try:
            from modules.mmr_reranker import MMRReranker
            st.markdown("✅ MMR Module")
        except ImportError:
            st.markdown("⬜ MMR (stub)")

        try:
            from modules.citation_generator import CitationGenerator
            st.markdown("✅ Citation Module")
        except ImportError:
            st.markdown("⬜ Citation (stub)")

    st.markdown("✅ Hallucination Detector")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODE 1: Single Query
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if mode == "Single Query":
    st.markdown("# 🔍 Single Query Mode")
    st.markdown("Ask a question using either Vanilla RAG or Hallucination-Free RAG.")

    if not st.session_state.documents:
        st.warning("⚠️ Please load documents first (use sidebar).")
        st.stop()

    # ── Variant Selection ──
    variant = st.radio(
        "Select RAG Variant",
        ["Vanilla RAG", "RAG + Hallucination Detection"],
        horizontal=True,
    )

    # ── Query Input ──
    query = st.text_input(
        "💬 Ask a question",
        placeholder="e.g., What is photosynthesis and how does it work?",
    )

    # ── Sample Questions ──
    st.markdown("**Quick questions:**")
    sample_qs = [
        "What is photosynthesis?",
        "Explain Newton's three laws of motion",
        "What is machine learning?",
        "Describe the periodic table",
        "What was the Industrial Revolution?",
    ]
    cols = st.columns(len(sample_qs))
    for i, q in enumerate(sample_qs):
        if cols[i].button(q[:25] + "...", key=f"sq_{i}"):
            query = q

    if query:
        # Build config based on variant
        use_hall = variant == "RAG + Hallucination Detection"
        config = RAGConfig(
            use_mmr=False,
            use_citation=False,
            use_hallucination_detection=use_hall,
            top_k=top_k,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_temperature=temperature,
            hallucination_threshold=hall_threshold,
        )

        pipeline = get_or_create_pipeline(variant, config)

        # Index if needed
        if not st.session_state.indexed:
            with st.spinner("📦 Indexing documents..."):
                pipeline.index_documents(st.session_state.documents)
                st.session_state.indexed = True

        # Run query
        with st.spinner(f"🔄 Running {variant}..."):
            response = pipeline.query(query)

        # ── Display Answer ──
        st.markdown('<div class="section-header">💡 Answer</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="answer-box">{response.final_answer}</div>', unsafe_allow_html=True)

        # ── Retrieved Documents ──
        with st.expander("📄 Retrieved Documents", expanded=False):
            for doc in response.retrieved_docs:
                st.markdown(f"""
                **{doc.document.doc_id}** — Score: `{doc.score:.4f}` | 
                Subject: {doc.document.metadata.get('subject', 'N/A')} |
                Topic: {doc.document.metadata.get('topic', 'N/A')}
                """)
                st.caption(doc.document.content)
                st.markdown("---")

        # ── Hallucination Analysis (if enabled) ──
        if use_hall and response.claims:
            st.markdown('<div class="section-header">🔬 Hallucination Analysis</div>', unsafe_allow_html=True)

            detector = HallucinationDetector(config)
            report = detector.get_detailed_report(response.claims)

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                render_metric_card(
                    "Faithfulness",
                    f"{report['summary']['faithfulness_score']:.1%}",
                    "Higher is better"
                )
            with col2:
                render_metric_card(
                    "Hallucination Rate",
                    f"{report['summary']['hallucination_rate']:.1%}",
                    "Lower is better"
                )
            with col3:
                render_metric_card(
                    "Claims Supported",
                    f"{report['summary']['supported']}/{report['summary']['total_claims']}",
                    "Verified claims"
                )
            with col4:
                render_metric_card(
                    "Verification Time",
                    f"{response.module_timings.get('hallucination', 0):.0f}ms",
                    "Processing time"
                )

            # Individual claims
            st.markdown("**Claim-Level Analysis:**")
            for i, claim in enumerate(response.claims):
                render_claim_card(claim, i)

            # Show raw vs filtered
            if response.raw_answer != response.final_answer:
                with st.expander("📝 Raw vs Filtered Answer"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Raw Answer (before filtering):**")
                        st.markdown(response.raw_answer)
                    with col2:
                        st.markdown("**Filtered Answer (after removing hallucinations):**")
                        st.markdown(response.final_answer)

        # ── Pipeline Metrics ──
        with st.expander("📊 Pipeline Metrics", expanded=False):
            st.json(response.metrics)
            st.markdown("**Module Timings:**")
            timing_df = pd.DataFrame([response.module_timings]).T
            timing_df.columns = ["Time (ms)"]
            st.dataframe(timing_df)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODE 2: Comparison Mode
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif mode == "Comparison Mode":
    st.markdown("# ⚔️ Comparison Mode")
    st.markdown("Run the same query through Vanilla RAG and Hallucination-Free RAG side-by-side.")

    if not st.session_state.documents:
        st.warning("⚠️ Please load documents first (use sidebar).")
        st.stop()

    query = st.text_input(
        "💬 Ask a question",
        placeholder="e.g., What is photosynthesis and how does it work?",
    )

    # Optional ground truth for evaluation
    ground_truth = st.text_area(
        "📋 Ground Truth Answer (optional, for metric computation)",
        placeholder="Paste a reference answer here to compute F1, EM, ROUGE-L...",
        height=80,
    )

    if query:
        # ── Build two pipelines ──
        config_vanilla = RAGConfig(
            use_hallucination_detection=False,
            top_k=top_k, llm_provider=llm_provider, llm_model=llm_model,
            llm_temperature=temperature,
        )
        config_hallfree = RAGConfig(
            use_hallucination_detection=True,
            top_k=top_k, llm_provider=llm_provider, llm_model=llm_model,
            llm_temperature=temperature, hallucination_threshold=hall_threshold,
        )

        pipe_vanilla = get_or_create_pipeline("Vanilla RAG", config_vanilla)
        pipe_hallfree = get_or_create_pipeline("RAG + Hallucination Detection", config_hallfree)

        if not st.session_state.indexed:
            with st.spinner("📦 Indexing documents..."):
                pipe_vanilla.index_documents(st.session_state.documents)
                pipe_hallfree.index_documents(st.session_state.documents)
                st.session_state.indexed = True

        # ── Run both ──
        gt = ground_truth.strip() if ground_truth.strip() else None
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("### 📦 Vanilla RAG")
            with st.spinner("Running..."):
                resp_vanilla = pipe_vanilla.query(query, ground_truth=gt)
            st.markdown(f'<div class="answer-box">{resp_vanilla.final_answer}</div>', unsafe_allow_html=True)

            # Metrics
            st.markdown("**Metrics:**")
            m = resp_vanilla.metrics
            render_metric_card("Avg Similarity", f"{m.get('avg_similarity', 0):.4f}")
            render_metric_card("Total Time", f"{resp_vanilla.module_timings.get('total', 0):.0f}ms")

        with col_right:
            st.markdown("### 🛡️ Hallucination-Free RAG")
            with st.spinner("Running..."):
                resp_hallfree = pipe_hallfree.query(query, ground_truth=gt)
            st.markdown(f'<div class="answer-box">{resp_hallfree.final_answer}</div>', unsafe_allow_html=True)

            # Hallucination metrics
            st.markdown("**Metrics:**")
            h = resp_hallfree.hallucination_stats
            render_metric_card(
                "Faithfulness",
                f"{h.get('faithfulness_score', 0):.1%}",
                f"Hallucination Rate: {h.get('hallucination_rate', 0):.1%}"
            )
            render_metric_card(
                "Claims",
                f"{h.get('num_supported', 0)}/{h.get('num_claims', 0)} supported"
            )
            render_metric_card("Total Time", f"{resp_hallfree.module_timings.get('total', 0):.0f}ms")

        # ── Answer Quality Comparison (if ground truth provided) ──
        if gt:
            st.markdown("---")
            st.markdown('<div class="section-header">📊 Answer Quality Comparison</div>', unsafe_allow_html=True)

            ml = st.session_state.metrics_logger
            aq_vanilla = ml.compute_answer_quality(resp_vanilla.final_answer, gt)
            aq_hallfree = ml.compute_answer_quality(resp_hallfree.final_answer, gt)

            comp_df = pd.DataFrame({
                "Metric": ["F1 Score", "Exact Match", "ROUGE-L", "Token Precision", "Token Recall"],
                "Vanilla RAG": [
                    f"{aq_vanilla.get('f1_score', 0):.4f}",
                    f"{aq_vanilla.get('exact_match', 0):.0f}",
                    f"{aq_vanilla.get('rouge_l', 0):.4f}",
                    f"{aq_vanilla.get('token_precision', 0):.4f}",
                    f"{aq_vanilla.get('token_recall', 0):.4f}",
                ],
                "Hall-Free RAG": [
                    f"{aq_hallfree.get('f1_score', 0):.4f}",
                    f"{aq_hallfree.get('exact_match', 0):.0f}",
                    f"{aq_hallfree.get('rouge_l', 0):.4f}",
                    f"{aq_hallfree.get('token_precision', 0):.4f}",
                    f"{aq_hallfree.get('token_recall', 0):.4f}",
                ],
            })
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # ── Claim Detail for HallFree ──
        if resp_hallfree.claims:
            with st.expander("🔬 Claim-Level Analysis (Hallucination-Free RAG)"):
                for i, claim in enumerate(resp_hallfree.claims):
                    render_claim_card(claim, i)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODE 3: Ablation Study
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif mode == "Ablation Study":
    st.markdown("# 🧪 Ablation Study Runner")
    st.markdown("""
    Run systematic experiments comparing all RAG variants.
    Results are logged for statistical analysis.
    """)

    if not st.session_state.documents:
        st.warning("⚠️ Please load documents first (use sidebar).")
        st.stop()

    # ── Variant Selection ──
    st.markdown("### Select Variants to Compare")
    selected_variants = st.multiselect(
        "Ablation variants",
        list(ABLATION_VARIANTS.keys()),
        default=["Vanilla RAG", "RAG + Hallucination Detection"],
    )

    # ── Query Set ──
    st.markdown("### Query Set")
    query_input_method = st.radio("Input method", ["Manual", "Predefined Set"], horizontal=True)

    if query_input_method == "Manual":
        queries_text = st.text_area(
            "Enter queries (one per line)",
            value="What is photosynthesis?\nExplain Newton's laws of motion\nWhat is machine learning?",
            height=150,
        )
        queries = [q.strip() for q in queries_text.strip().split("\n") if q.strip()]
    else:
        queries = [
            "What is photosynthesis and how does it work?",
            "Explain Newton's three laws of motion.",
            "What is the periodic table?",
            "How do chemical bonds form?",
            "What is the Pythagorean theorem?",
            "Describe the Industrial Revolution.",
            "What is machine learning?",
            "What is calculus used for?",
        ]
        st.info(f"Using {len(queries)} predefined educational queries")

    num_runs = st.slider("Runs per query (for statistical significance)", 1, 5, 1)

    if st.button("🚀 Run Ablation Study", type="primary", use_container_width=True):
        # Reset metrics logger for fresh experiment
        st.session_state.metrics_logger = MetricsLogger("ablation_study")

        progress = st.progress(0)
        status = st.empty()
        total_steps = len(selected_variants) * len(queries) * num_runs
        step = 0

        results_table = []

        for variant_name in selected_variants:
            config = ABLATION_VARIANTS[variant_name]
            # Override with sidebar settings
            config.top_k = top_k
            config.llm_provider = llm_provider
            config.llm_model = llm_model
            config.llm_temperature = temperature
            config.hallucination_threshold = hall_threshold

            pipeline = RAGPipeline(config, st.session_state.metrics_logger)
            pipeline.index_documents(st.session_state.documents)

            for query in queries:
                for run in range(num_runs):
                    step += 1
                    progress.progress(step / total_steps)
                    status.text(f"Running: {variant_name} | Query: {query[:40]}... | Run {run+1}/{num_runs}")

                    response = pipeline.query(query)

                    results_table.append({
                        "Variant": variant_name,
                        "Query": query[:50],
                        "Run": run + 1,
                        "Avg Similarity": response.metrics.get("avg_similarity", 0),
                        "Hallucination Rate": response.hallucination_stats.get("hallucination_rate", None),
                        "Faithfulness": response.hallucination_stats.get("faithfulness_score", None),
                        "Num Claims": response.hallucination_stats.get("num_claims", None),
                        "Total Time (ms)": response.module_timings.get("total", 0),
                        "Answer Length": len(response.final_answer),
                    })

        progress.progress(1.0)
        status.text("✅ Ablation study complete!")

        # ── Results Table ──
        st.markdown("---")
        st.markdown("### 📊 Results")

        df = pd.DataFrame(results_table)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # ── Summary Statistics ──
        st.markdown("### 📈 Summary by Variant")
        summary = df.groupby("Variant").agg({
            "Avg Similarity": ["mean", "std"],
            "Hallucination Rate": ["mean", "std"],
            "Faithfulness": ["mean", "std"],
            "Total Time (ms)": ["mean", "std"],
            "Answer Length": ["mean"],
        }).round(4)
        st.dataframe(summary, use_container_width=True)

        # ── Bar Chart ──
        if "Hallucination Rate" in df.columns and df["Hallucination Rate"].notna().any():
            st.markdown("### Hallucination Rate by Variant")
            chart_data = df.groupby("Variant")["Hallucination Rate"].mean()
            st.bar_chart(chart_data)

        # ── Export ──
        st.markdown("### 💾 Export Results")
        col1, col2 = st.columns(2)
        with col1:
            csv_data = df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv_data,
                "ablation_results.csv",
                "text/csv",
                use_container_width=True,
            )
        with col2:
            json_data = json.dumps(results_table, indent=2, default=str)
            st.download_button(
                "📥 Download JSON",
                json_data,
                "ablation_results.json",
                "application/json",
                use_container_width=True,
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODE 4: Metrics Dashboard
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif mode == "Metrics Dashboard":
    st.markdown("# 📊 Metrics Dashboard")
    st.markdown("View accumulated metrics from all queries across sessions.")

    ml = st.session_state.metrics_logger

    if not ml.logs:
        st.info("No query logs yet. Run some queries first!")
        st.stop()

    # ── Overview ──
    st.markdown(f"**Total queries logged:** {len(ml.logs)}")
    variants_seen = list(set(l.variant for l in ml.logs))
    st.markdown(f"**Variants tested:** {', '.join(variants_seen)}")

    # ── Per-Variant Summary ──
    st.markdown("---")
    st.markdown("### Variant Comparison Table")

    comparison = ml.get_comparison_table()
    if comparison:
        comp_df = pd.DataFrame(comparison)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # ── Timeline ──
    st.markdown("### Query Timeline")
    timeline_data = []
    for log in ml.logs:
        timeline_data.append({
            "Query": log.query[:40],
            "Variant": log.variant,
            "Time (ms)": log.total_time_ms,
            "Hallucination Rate": log.hallucination_metrics.get("hallucination_rate", None),
        })
    st.dataframe(pd.DataFrame(timeline_data), use_container_width=True, hide_index=True)

    # ── Export All ──
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Export All Logs (JSON)", use_container_width=True):
            export_path = "/tmp/edu_rag_all_logs.json"
            ml.export_to_json(export_path)
            with open(export_path) as f:
                st.download_button(
                    "Download JSON",
                    f.read(),
                    "edu_rag_all_logs.json",
                    "application/json",
                )
    with col2:
        if st.button("📥 Export All Logs (CSV)", use_container_width=True):
            export_path = "/tmp/edu_rag_all_logs.csv"
            ml.export_to_csv(export_path)
            with open(export_path) as f:
                st.download_button(
                    "Download CSV",
                    f.read(),
                    "edu_rag_all_logs.csv",
                    "text/csv",
                )