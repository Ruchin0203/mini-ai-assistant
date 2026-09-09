"""
Flask REST API for the Mini AI Assistant.
Endpoints: GET /health, POST /ask
"""

import sys
import os

# Ensure project root is on the path so 'backend' package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
from flask_cors import CORS

from backend.config import FLASK_PORT
from backend.data_loader import load_data
from backend.analytics import get_total_revenue, get_top_products, get_top_country, get_country_revenue
from backend.router import classify_question, detect_analytics_intent, extract_top_n
from backend.rag import ask_rag, retrieve, generate_answer

app = Flask(__name__)
CORS(app)


# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════

def _format_currency(value: float) -> str:
    """Format a number as GBP currency string."""
    return f"£{value:,.2f}"


def _handle_analytics(question: str, df) -> dict:
    """Process an analytics question and return a structured response."""
    intent = detect_analytics_intent(question)

    if intent == "total_revenue":
        data = get_total_revenue(df)
        return {
            "question": question,
            "type": "analytics",
            "answer": f"The total revenue is {_format_currency(data['value'])}.",
            "data": data,
        }

    elif intent == "top_products":
        top_n = extract_top_n(question)
        data = get_top_products(df, top_n=top_n)
        lines = [f"{p['rank']}. {p['product']} — {_format_currency(p['revenue'])}" for p in data]
        answer = f"Top {top_n} products by revenue:\n" + "\n".join(lines)
        return {
            "question": question,
            "type": "analytics",
            "answer": answer,
            "data": data,
        }

    elif intent == "top_country":
        data = get_top_country(df)
        return {
            "question": question,
            "type": "analytics",
            "answer": (
                f"The country with the highest revenue is {data['country']} "
                f"with {_format_currency(data['revenue'])}."
            ),
            "data": data,
        }

    else:
        # Fallback: return total revenue for generic analytics questions
        data = get_total_revenue(df)
        return {
            "question": question,
            "type": "analytics",
            "answer": f"The total revenue is {_format_currency(data['value'])}.",
            "data": data,
        }


def _handle_hybrid(question: str, df) -> dict:
    """
    Process a hybrid question:
    Step 1: Pandas calculates the deterministic metric.
    Step 2: FAISS retrieves relevant transaction records.
    Step 3: LLM explains patterns from retrieved context.
    Step 4: Combine metric + explanation.
    """
    intent = detect_analytics_intent(question)

    # Step 1: Get deterministic data
    if intent == "country_revenue":
        data = get_country_revenue(df, "United Kingdom")
    elif intent == "top_country":
        data = get_top_country(df)
    elif intent == "top_products":
        top_n = extract_top_n(question)
        data = get_top_products(df, top_n=top_n)
    elif intent == "total_revenue":
        data = get_total_revenue(df)
    else:
        data = get_top_country(df)

    # Step 2 & 3: RAG retrieval + LLM explanation
    try:
        retrieved_docs = retrieve(question)
        context = "\n".join(retrieved_docs)
        explanation = generate_answer(question, context)
    except FileNotFoundError:
        explanation = (
            "FAISS index not found. Run 'python scripts/build_index.py' first "
            "to enable RAG-based explanations."
        )
        retrieved_docs = []

    # Step 4: Combine
    if isinstance(data, dict) and "country" in data:
        metric_text = f"{data['country']} Revenue: {_format_currency(data['revenue'])}"
    elif isinstance(data, dict) and "value" in data:
        metric_text = f"Total Revenue: {_format_currency(data['value'])}"
    elif isinstance(data, list):
        lines = [f"{p['rank']}. {p['product']} — {_format_currency(p['revenue'])}" for p in data]
        metric_text = "Top Products:\n" + "\n".join(lines)
    else:
        metric_text = str(data)

    answer = f"{metric_text}\n\nExplanation:\n{explanation}"

    return {
        "question": question,
        "type": "hybrid",
        "answer": answer,
        "data": data,
        "retrieved_documents": len(retrieved_docs),
    }


# ═══════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"})


@app.route("/ask", methods=["POST"])
def ask():
    """
    Main question-answering endpoint.

    Request JSON:
        {"question": "What is total revenue?"}

    Returns structured JSON with question, type, answer, and optional data.
    """
    # ── Validate request ──────────────────────────────────
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    body = request.get_json(silent=True)
    if body is None:
        return jsonify({"error": "Invalid JSON"}), 400

    question = body.get("question", "").strip()
    if not question:
        return jsonify({"error": "Missing or empty 'question' field"}), 400

    try:
        # ── Load data ─────────────────────────────────────
        df = load_data()

        # ── Route question ────────────────────────────────
        q_type = classify_question(question)

        if q_type == "analytics":
            result = _handle_analytics(question, df)

        elif q_type == "hybrid":
            result = _handle_hybrid(question, df)

        else:  # "rag"
            result = ask_rag(question)

        return jsonify(result)

    except FileNotFoundError as e:
        return jsonify({
            "error": "Dataset or index not found",
            "detail": str(e),
        }), 404

    except Exception as e:
        return jsonify({
            "error": "Internal server error",
            "detail": "An unexpected error occurred. Please check the server logs.",
        }), 500


# ═══════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Mini AI Assistant — Flask API")
    print("=" * 60)

    # Pre-load dataset on startup
    try:
        df = load_data()
        print(f"  Dataset loaded: {len(df):,} transactions")
        print(f"  Total revenue: £{df['Revenue'].sum():,.2f}")
    except FileNotFoundError:
        print("  WARNING: Dataset not found. Download it first.")
        print("  Run: python scripts/download_data.py")

    print(f"\n  Starting server on port {FLASK_PORT}...")
    print(f"  Health: http://localhost:{FLASK_PORT}/health")
    print(f"  Ask:    POST http://localhost:{FLASK_PORT}/ask")
    print("=" * 60)

    app.run(host="0.0.0.0", port=FLASK_PORT, debug=False)
