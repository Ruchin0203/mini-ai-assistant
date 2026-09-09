"""
Configuration module for Mini AI Assistant.
Loads settings from .env file and provides defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env from project root ──────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ── Paths ─────────────────────────────────────────────────
DATA_PATH = os.getenv("DATA_PATH", "data/Online Retail.xlsx")
VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "vectorstore/")

# Make paths absolute relative to project root
if not os.path.isabs(DATA_PATH):
    DATA_PATH = str(PROJECT_ROOT / DATA_PATH)
if not os.path.isabs(VECTORSTORE_PATH):
    VECTORSTORE_PATH = str(PROJECT_ROOT / VECTORSTORE_PATH)

# ── Hugging Face ──────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

# ── Embedding Model (local) ──────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── RAG Settings ──────────────────────────────────────────
TOP_K = int(os.getenv("TOP_K", "5"))
RAG_SAMPLE_SIZE = int(os.getenv("RAG_SAMPLE_SIZE", "5000"))

# ── Flask ─────────────────────────────────────────────────
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
