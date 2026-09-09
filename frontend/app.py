"""
Streamlit frontend for the Mini AI Assistant.
Provides a chat-based interface for business analytics and RAG-powered insights.

Usage:
    streamlit run frontend/app.py
"""

import streamlit as st
import requests
import json

# ── Configuration ─────────────────────────────────────────
API_URL = "http://localhost:5000"

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="Mini AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 {
        color: white;
        font-size: 2.2rem;
        margin-bottom: 0.2rem;
    }
    .main-header p {
        color: rgba(255,255,255,0.85);
        font-size: 1rem;
        margin-top: 0;
    }

    /* Response type badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    .badge-analytics {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
    }
    .badge-rag {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
    }
    .badge-hybrid {
        background: linear-gradient(135deg, #f093fb, #f5576c);
        color: white;
    }

    /* Metric card */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.75rem 0;
        color: white;
    }
    .metric-card .label {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.6);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38ef7d;
        margin-top: 0.25rem;
    }

    /* Sidebar styling */
    .sidebar-section {
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        padding: 0.75rem;
        margin-bottom: 0.75rem;
    }

    /* Status indicator */
    .status-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .status-online { background: #38ef7d; }
    .status-offline { background: #f5576c; }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🤖 AI-Powered Retail Assistant</h1>
    <p>Conversational Business Intelligence</p>
</div>
""", unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Controls")

    # Health check
    try:
        resp = requests.get(f"{API_URL}/health", timeout=3)
        if resp.status_code == 200:
            st.markdown(
                '<span class="status-dot status-online"></span> **API Online**',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-dot status-offline"></span> **API Offline**',
                unsafe_allow_html=True,
            )
    except requests.ConnectionError:
        st.markdown(
            '<span class="status-dot status-offline"></span> **API Offline**',
            unsafe_allow_html=True,
        )
        st.caption("Start the Flask API with: `python backend/main.py`")

    st.divider()

    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # Example questions
    st.markdown("### 💡 Example Questions")

    example_questions = [
        "What's the total revenue?",
        "Top 3 products by sales?",
        "Which country has highest revenue?",
        "Describe customer purchasing behavior.",
        "What trends do you observe?",
        "Why is the UK leading?",
    ]

    for eq in example_questions:
        if st.button(eq, key=f"eq_{eq}", use_container_width=True):
            st.session_state.pending_question = eq
            st.rerun()

    st.divider()

    # Architecture info
    st.markdown("### 📐 Architecture")
    st.caption(
        "• **Analytics** → Pandas (local)\n"
        "• **Embeddings** → Sentence Transformers (local)\n"
        "• **Search** → FAISS (local)\n"
        "• **LLM** → Hugging Face (free tier)"
    )


# ── Helper Functions ──────────────────────────────────────

def get_badge_html(q_type: str) -> str:
    """Return styled HTML badge for the response type."""
    labels = {
        "analytics": ("📊 Analytics", "badge-analytics"),
        "rag": ("🔍 RAG + LLM", "badge-rag"),
        "hybrid": ("⚡ Hybrid", "badge-hybrid"),
    }
    label, css = labels.get(q_type, ("❓ Unknown", "badge-rag"))
    return f'<span class="badge {css}">{label}</span>'


def format_response(result: dict) -> str:
    """Format the API response for display."""
    q_type = result.get("type", "unknown")
    answer = result.get("answer", "No answer available.")
    data = result.get("data")

    parts = []

    # Badge
    parts.append(get_badge_html(q_type))

    # Metric cards for analytics data
    if data and q_type in ("analytics", "hybrid"):
        if isinstance(data, dict):
            if "metric" in data and data["metric"] == "total_revenue":
                parts.append(
                    f'<div class="metric-card">'
                    f'<div class="label">Total Revenue</div>'
                    f'<div class="value">£{data["value"]:,.2f}</div>'
                    f'</div>'
                )
            elif "country" in data:
                parts.append(
                    f'<div class="metric-card">'
                    f'<div class="label">{data["country"]}</div>'
                    f'<div class="value">£{data["revenue"]:,.2f}</div>'
                    f'</div>'
                )
        elif isinstance(data, list):
            # Product table
            table = "| Rank | Product | Revenue |\n|------|---------|--------|\n"
            for p in data:
                table += f"| {p['rank']} | {p['product']} | £{p['revenue']:,.2f} |\n"
            parts.append(table)

    # Text answer (for RAG/hybrid explanation or analytics text)
    if q_type == "rag":
        parts.append(f"\n{answer}")
    elif q_type == "hybrid":
        # Extract explanation part (after the metric)
        if "Explanation:" in answer:
            explanation = answer.split("Explanation:", 1)[1].strip()
            parts.append(f"\n**Explanation:**\n{explanation}")
    elif q_type == "analytics" and not data:
        parts.append(f"\n{answer}")

    return "\n".join(parts)


def ask_question(question: str) -> dict | None:
    """Send a question to the Flask API."""
    try:
        resp = requests.post(
            f"{API_URL}/ask",
            json={"question": question},
            timeout=60,
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            return {"type": "error", "answer": f"API Error {resp.status_code}: {resp.text}"}
    except requests.ConnectionError:
        return {
            "type": "error",
            "answer": "Cannot connect to the API. Make sure the Flask server is running:\n\n`python backend/main.py`",
        }
    except requests.Timeout:
        return {"type": "error", "answer": "Request timed out. The server may be overloaded."}


# ── Chat Display ──────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(msg["content"], unsafe_allow_html=True)
        else:
            st.markdown(msg["content"])


# ── Handle pending question from sidebar ──────────────────
if "pending_question" in st.session_state:
    question = st.session_state.pending_question
    del st.session_state.pending_question

    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = ask_question(question)
            if result:
                formatted = format_response(result)
                st.markdown(formatted, unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": formatted})
            else:
                error_msg = "❌ Failed to get a response."
                st.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})


# ── Chat Input ────────────────────────────────────────────
if prompt := st.chat_input("Ask about the Online Retail dataset..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = ask_question(prompt)
            if result:
                formatted = format_response(result)
                st.markdown(formatted, unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": formatted})
            else:
                error_msg = "❌ Failed to get a response."
                st.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
