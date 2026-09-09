"""
Unit and integration tests for the Flask API.
Uses mocks for the LLM and data loading — no external API calls needed.
"""

import pytest
import json
import sys
import os
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd


# ── Test Fixtures ─────────────────────────────────────────

@pytest.fixture
def mock_df():
    """Create a mock DataFrame for testing."""
    data = {
        "InvoiceNo": ["536365", "536365", "536366", "536367", "536368"],
        "StockCode": ["85123A", "71053", "84406B", "84029G", "84029E"],
        "Description": [
            "WHITE HANGING HEART T-LIGHT HOLDER",
            "WHITE METAL LANTERN",
            "CREAM CUPID HEARTS COAT HANGER",
            "KNITTED UNION FLAG HOT WATER BOTTLE",
            "RED WOOLLY HOTTIE WHITE HEART",
        ],
        "Quantity": [6, 6, 8, 6, 6],
        "UnitPrice": [2.55, 3.39, 2.75, 3.39, 3.39],
        "CustomerID": [17850, 17850, 17850, 13047, 13047],
        "Country": [
            "United Kingdom",
            "United Kingdom",
            "France",
            "United Kingdom",
            "France",
        ],
    }
    df = pd.DataFrame(data)
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    return df


@pytest.fixture
def client(mock_df):
    """Create a Flask test client with mocked data loading."""
    with patch("backend.main.load_data", return_value=mock_df):
        from backend.main import app
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client


# ── Health Endpoint Tests ─────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok(self, client):
        response = client.get("/health")
        data = response.get_json()
        assert data["status"] == "ok"


# ── Ask Endpoint Tests ────────────────────────────────────

class TestAskEndpoint:
    def test_missing_json(self, client):
        response = client.post("/ask", data="not json", content_type="text/plain")
        assert response.status_code == 400

    def test_empty_question(self, client):
        response = client.post("/ask", json={"question": ""})
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    def test_missing_question_field(self, client):
        response = client.post("/ask", json={"query": "test"})
        assert response.status_code == 400

    def test_total_revenue(self, client):
        response = client.post("/ask", json={"question": "What is total revenue?"})
        assert response.status_code == 200
        data = response.get_json()
        assert data["type"] == "analytics"
        assert "data" in data
        assert data["data"]["metric"] == "total_revenue"
        assert data["data"]["currency"] == "GBP"
        assert isinstance(data["data"]["value"], float)

    def test_top_products(self, client):
        response = client.post("/ask", json={"question": "Top 3 products by sales?"})
        assert response.status_code == 200
        data = response.get_json()
        assert data["type"] == "analytics"
        assert isinstance(data["data"], list)
        assert len(data["data"]) == 3

    def test_top_country(self, client):
        response = client.post("/ask", json={"question": "Which country has highest revenue?"})
        assert response.status_code == 200
        data = response.get_json()
        assert data["type"] == "analytics"
        assert data["data"]["country"] == "United Kingdom"

    @patch("backend.main.ask_rag")
    def test_rag_question(self, mock_rag, client):
        mock_rag.return_value = {
            "question": "What trends do you observe?",
            "type": "rag",
            "answer": "Based on the retrieved transactions, there is a trend...",
            "retrieved_documents": 5,
        }
        response = client.post("/ask", json={"question": "What trends do you observe?"})
        assert response.status_code == 200
        data = response.get_json()
        assert data["type"] == "rag"
        assert "answer" in data

    @patch("backend.main.retrieve")
    @patch("backend.main.generate_answer")
    def test_hybrid_question(self, mock_gen, mock_retrieve, client):
        mock_retrieve.return_value = ["doc1", "doc2"]
        mock_gen.return_value = "The UK leads because of high transaction volume."

        response = client.post("/ask", json={"question": "Why is the UK leading?"})
        assert response.status_code == 200
        data = response.get_json()
        assert data["type"] == "hybrid"
        assert "data" in data
        assert "answer" in data


# ── Router Tests ──────────────────────────────────────────

class TestRouter:
    def test_analytics_total_revenue(self):
        from backend.router import classify_question
        assert classify_question("What is total revenue?") == "analytics"

    def test_analytics_top_products(self):
        from backend.router import classify_question
        assert classify_question("Top 3 products by sales?") == "analytics"

    def test_analytics_top_country(self):
        from backend.router import classify_question
        assert classify_question("Which country has highest revenue?") == "analytics"

    def test_rag_trends(self):
        from backend.router import classify_question
        assert classify_question("What trends do you observe?") == "rag"

    def test_rag_behavior(self):
        from backend.router import classify_question
        assert classify_question("Describe customer purchasing behavior.") == "rag"

    def test_hybrid_why_uk(self):
        from backend.router import classify_question
        assert classify_question("Why is the UK leading?") == "hybrid"

    def test_extract_top_n_default(self):
        from backend.router import extract_top_n
        assert extract_top_n("Top products") == 3

    def test_extract_top_n_custom(self):
        from backend.router import extract_top_n
        assert extract_top_n("Top 5 products") == 5
