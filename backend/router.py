"""
Question router: classifies user questions into analytics, rag, or hybrid.
Uses keyword matching — no LLM calls.
"""

import re


# ── Keyword patterns ──────────────────────────────────────

_ANALYTICS_PATTERNS = [
    r"\btotal\s+revenue\b",
    r"\btotal\s+sales\b",
    r"\boverall\s+revenue\b",
    r"\btop\s+\d*\s*products?\b",
    r"\bbest\s+selling\b",
    r"\btop\s+\d*\s*selling\b",
    r"\bwhich\s+country\b.*\b(highest|most|top|leading)\b",
    r"\b(highest|most|top|leading)\s+country\b",
    r"\bhow\s+much\s+revenue\b",
    r"\btop\s+\d*\s*countr",
    r"\bproducts?\s+by\s+(sales|revenue)\b",
    r"\bcountry.*highest\s+revenue\b",
]

_HYBRID_PATTERNS = [
    r"\bwhy\b.*\b(uk|united\s+kingdom|leading|highest|top|most)\b",
    r"\bwhy\s+is\b",
    r"\bwhy\s+does\b",
    r"\bexplain\b.*\b(revenue|sales|country|product)\b",
    r"\breason\b.*\b(revenue|sales|country|product)\b",
]

_RAG_PATTERNS = [
    r"\bdescribe\b",
    r"\btrend",
    r"\bpattern",
    r"\bbehavior\b",
    r"\bbehaviour\b",
    r"\bobserv",
    r"\binsight",
    r"\bsummar",
    r"\banalys[ei]s\b",
    r"\bwhat\s+do\s+you\s+(see|observe|notice)\b",
    r"\btell\s+me\s+about\b",
]


def classify_question(question: str) -> str:
    """
    Classify a user question into one of: 'analytics', 'rag', or 'hybrid'.

    Args:
        question: The user's natural language question.

    Returns:
        One of 'analytics', 'rag', or 'hybrid'.
    """
    q = question.lower().strip()

    # Check analytics first (deterministic queries take priority)
    for pattern in _ANALYTICS_PATTERNS:
        if re.search(pattern, q):
            return "analytics"

    # Check hybrid (requires both analytics data AND LLM explanation)
    for pattern in _HYBRID_PATTERNS:
        if re.search(pattern, q):
            return "hybrid"

    # Check RAG (qualitative / exploratory questions)
    for pattern in _RAG_PATTERNS:
        if re.search(pattern, q):
            return "rag"

    # Default to RAG for unclassified questions
    return "rag"


def detect_analytics_intent(question: str) -> str | None:
    """
    For analytics and hybrid questions, determine which specific analytics
    function to call.

    Returns:
        One of 'total_revenue', 'top_products', 'top_country', 'country_revenue',
        or None if no specific intent is detected.
    """
    q = question.lower().strip()

    # Total revenue
    if re.search(r"\btotal\s+(revenue|sales)\b", q) or re.search(r"\boverall\s+revenue\b", q):
        return "total_revenue"

    if re.search(r"\bhow\s+much\s+revenue\b", q):
        return "total_revenue"

    # Top products
    if re.search(r"\btop\s+\d*\s*products?\b", q) or re.search(r"\bbest\s+selling\b", q):
        return "top_products"

    if re.search(r"\bproducts?\s+by\s+(sales|revenue)\b", q):
        return "top_products"

    # Top country
    if re.search(r"\b(which|what)\s+country\b.*\b(highest|most|top|leading)\b", q):
        return "top_country"

    if re.search(r"\b(highest|most|top|leading)\s+country\b", q):
        return "top_country"

    if re.search(r"\btop\s+\d*\s*countr", q):
        return "top_country"

    # Specific country (for hybrid)
    if re.search(r"\b(uk|united\s+kingdom)\b", q):
        return "country_revenue"

    return None


def extract_top_n(question: str) -> int:
    """
    Extract the 'N' from questions like 'Top 5 products'.
    Defaults to 3 if no number is found.
    """
    match = re.search(r"\btop\s+(\d+)", question.lower())
    if match:
        return int(match.group(1))
    return 3
