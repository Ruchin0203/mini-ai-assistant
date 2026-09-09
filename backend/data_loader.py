"""
Data loader for the UCI Online Retail dataset.
Loads, cleans, and caches the dataset in memory.
"""

import pandas as pd
from backend.config import DATA_PATH

# ── In-memory cache ───────────────────────────────────────
_cached_df: pd.DataFrame | None = None


def load_data(data_path: str | None = None, force_reload: bool = False) -> pd.DataFrame:
    """
    Load and clean the Online Retail dataset.

    Args:
        data_path: Path to the Excel file. Defaults to DATA_PATH from config.
        force_reload: If True, bypass the in-memory cache.

    Returns:
        Cleaned DataFrame with a computed 'Revenue' column.
    """
    global _cached_df

    if _cached_df is not None and not force_reload:
        return _cached_df

    path = data_path or DATA_PATH

    # ── Read Excel ────────────────────────────────────────
    df = pd.read_excel(path, engine="openpyxl")

    # ── Basic cleaning ────────────────────────────────────
    # Drop rows where Description is missing (can't identify the product)
    df = df.dropna(subset=["Description"])

    # Drop rows where CustomerID is missing (optional but standard)
    # We keep them to preserve valid transaction data
    # df = df.dropna(subset=["CustomerID"])

    # Ensure numeric types
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")

    # Drop rows where Quantity or UnitPrice became NaN after coercion
    df = df.dropna(subset=["Quantity", "UnitPrice"])

    # Filter to valid transactions (positive quantity and price)
    # This removes cancellations (negative quantities) and free items
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    # ── Compute Revenue ───────────────────────────────────
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    # Reset index after filtering
    df = df.reset_index(drop=True)

    # ── Cache the result ──────────────────────────────────
    _cached_df = df

    return df
