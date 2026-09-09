"""
Deterministic analytics functions using Pandas.
These functions NEVER call the LLM — they operate purely on the local dataset.
"""

import pandas as pd


def get_total_revenue(df: pd.DataFrame) -> dict:
    """
    Calculate total revenue from the entire cleaned dataset.

    Returns:
        dict with metric name, computed value, and currency.
    """
    total = float(df["Revenue"].sum())
    return {
        "metric": "total_revenue",
        "value": round(total, 2),
        "currency": "GBP",
    }


def get_top_products(df: pd.DataFrame, top_n: int = 3) -> list[dict]:
    """
    Get the top N products by total revenue.

    Args:
        df: Cleaned DataFrame with 'Description' and 'Revenue' columns.
        top_n: Number of top products to return.

    Returns:
        List of dicts with rank, product name, and revenue.
    """
    product_revenue = (
        df.groupby("Description")["Revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )

    results = []
    for rank, (product, revenue) in enumerate(product_revenue.items(), start=1):
        results.append({
            "rank": rank,
            "product": str(product),
            "revenue": round(float(revenue), 2),
        })

    return results


def get_top_country(df: pd.DataFrame) -> dict:
    """
    Get the country with the highest total revenue.

    Returns:
        dict with country name and revenue.
    """
    country_revenue = (
        df.groupby("Country")["Revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    top_country = country_revenue.index[0]
    top_revenue = float(country_revenue.iloc[0])

    return {
        "country": str(top_country),
        "revenue": round(top_revenue, 2),
    }


def get_country_revenue(df: pd.DataFrame, country: str) -> dict:
    """
    Get the revenue for a specific country.
    Used by hybrid routing to provide deterministic data alongside RAG explanations.

    Args:
        df: Cleaned DataFrame.
        country: Country name to look up.

    Returns:
        dict with country name and revenue.
    """
    country_data = df[df["Country"].str.lower() == country.lower()]

    if country_data.empty:
        return {"country": country, "revenue": 0.0}

    revenue = float(country_data["Revenue"].sum())
    # Use the actual casing from the dataset
    actual_name = country_data["Country"].iloc[0]

    return {
        "country": str(actual_name),
        "revenue": round(revenue, 2),
    }
