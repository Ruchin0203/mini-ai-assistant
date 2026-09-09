"""
Unit tests for the analytics module.
Uses mock DataFrames — no external data or API required.
"""

import pytest
import pandas as pd
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.analytics import get_total_revenue, get_top_products, get_top_country, get_country_revenue


# ── Test Fixtures ─────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Create a mock DataFrame resembling the Online Retail dataset."""
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


# ── Tests ─────────────────────────────────────────────────

class TestGetTotalRevenue:
    def test_returns_dict(self, sample_df):
        result = get_total_revenue(sample_df)
        assert isinstance(result, dict)

    def test_has_required_keys(self, sample_df):
        result = get_total_revenue(sample_df)
        assert "metric" in result
        assert "value" in result
        assert "currency" in result

    def test_metric_name(self, sample_df):
        result = get_total_revenue(sample_df)
        assert result["metric"] == "total_revenue"

    def test_currency_is_gbp(self, sample_df):
        result = get_total_revenue(sample_df)
        assert result["currency"] == "GBP"

    def test_correct_value(self, sample_df):
        result = get_total_revenue(sample_df)
        expected = sum([
            6 * 2.55,  # 15.30
            6 * 3.39,  # 20.34
            8 * 2.75,  # 22.00
            6 * 3.39,  # 20.34
            6 * 3.39,  # 20.34
        ])
        assert result["value"] == round(expected, 2)

    def test_value_is_float(self, sample_df):
        result = get_total_revenue(sample_df)
        assert isinstance(result["value"], float)


class TestGetTopProducts:
    def test_returns_list(self, sample_df):
        result = get_top_products(sample_df, top_n=3)
        assert isinstance(result, list)

    def test_correct_length(self, sample_df):
        result = get_top_products(sample_df, top_n=3)
        assert len(result) == 3

    def test_has_required_keys(self, sample_df):
        result = get_top_products(sample_df, top_n=1)
        assert "rank" in result[0]
        assert "product" in result[0]
        assert "revenue" in result[0]

    def test_rank_ordering(self, sample_df):
        result = get_top_products(sample_df, top_n=3)
        for i, item in enumerate(result, start=1):
            assert item["rank"] == i

    def test_descending_revenue(self, sample_df):
        result = get_top_products(sample_df, top_n=3)
        revenues = [item["revenue"] for item in result]
        assert revenues == sorted(revenues, reverse=True)

    def test_top_1(self, sample_df):
        result = get_top_products(sample_df, top_n=1)
        assert len(result) == 1
        assert result[0]["rank"] == 1

    def test_custom_top_n(self, sample_df):
        result = get_top_products(sample_df, top_n=2)
        assert len(result) == 2


class TestGetTopCountry:
    def test_returns_dict(self, sample_df):
        result = get_top_country(sample_df)
        assert isinstance(result, dict)

    def test_has_required_keys(self, sample_df):
        result = get_top_country(sample_df)
        assert "country" in result
        assert "revenue" in result

    def test_correct_top_country(self, sample_df):
        result = get_top_country(sample_df)
        # UK: 15.30 + 20.34 + 20.34 = 55.98
        # France: 22.00 + 20.34 = 42.34
        assert result["country"] == "United Kingdom"

    def test_revenue_is_float(self, sample_df):
        result = get_top_country(sample_df)
        assert isinstance(result["revenue"], float)


class TestGetCountryRevenue:
    def test_existing_country(self, sample_df):
        result = get_country_revenue(sample_df, "France")
        assert result["country"] == "France"
        assert result["revenue"] > 0

    def test_case_insensitive(self, sample_df):
        result = get_country_revenue(sample_df, "france")
        assert result["country"] == "France"

    def test_nonexistent_country(self, sample_df):
        result = get_country_revenue(sample_df, "Mars")
        assert result["revenue"] == 0.0

    def test_uk_revenue(self, sample_df):
        result = get_country_revenue(sample_df, "United Kingdom")
        expected = 6 * 2.55 + 6 * 3.39 + 6 * 3.39  # 55.98
        assert result["revenue"] == round(expected, 2)
