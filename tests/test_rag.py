"""
Unit tests for the RAG module.
Tests document creation, sampling, and prompt/generation fallbacks without external API calls.
"""

import pytest
import pandas as pd
import sys
import os
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.rag import create_documents, sample_data_for_rag, generate_answer, ask_rag


@pytest.fixture
def sample_df():
    data = {
        "InvoiceNo": ["536365", "536366"],
        "StockCode": ["85123A", "71053"],
        "Description": ["WHITE HEART T-LIGHT HOLDER", "WHITE METAL LANTERN"],
        "Quantity": [6, 2],
        "UnitPrice": [2.55, 3.39],
        "CustomerID": [17850, 17850],
        "Country": ["United Kingdom", "United Kingdom"],
    }
    df = pd.DataFrame(data)
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    return df


class TestDocumentCreation:
    def test_create_documents_count(self, sample_df):
        docs = create_documents(sample_df)
        assert len(docs) == 2

    def test_document_content(self, sample_df):
        docs = create_documents(sample_df)
        assert "Customer 17850" in docs[0]
        assert "United Kingdom" in docs[0]
        assert "WHITE HEART T-LIGHT HOLDER" in docs[0]
        assert "£15.30" in docs[0]


class TestSampling:
    def test_sample_smaller_than_size(self, sample_df):
        sampled = sample_data_for_rag(sample_df, sample_size=10)
        assert len(sampled) == 2

    def test_sample_exact_size(self, sample_df):
        sampled = sample_data_for_rag(sample_df, sample_size=1)
        assert len(sampled) == 1


class TestLLMGenerationFallback:
    def test_generate_without_token(self):
        with patch("backend.rag.HF_TOKEN", ""):
            ans = generate_answer("test question", "context")
            assert "Please configure HF_TOKEN" in ans or "unavailable" in ans.lower()

    def test_generate_handles_exception_gracefully(self):
        with patch("backend.rag.HF_TOKEN", "mock_token"):
            with patch("backend.rag._get_llm", side_effect=Exception("API Error")):
                ans = generate_answer("test question", "context")
                assert "currently unavailable" in ans
