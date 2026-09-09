"""
Build the FAISS index from the Online Retail dataset using LangChain.

Usage:
    python scripts/build_index.py

This script:
1. Loads and cleans the dataset.
2. Samples representative transactions (configurable via RAG_SAMPLE_SIZE).
3. Creates human-readable text documents from transactions.
4. Encodes documents using LangChain HuggingFaceEmbeddings (local).
5. Builds and saves a LangChain FAISS vector store to disk.
"""

import sys
import os
import time

# Add project root to path so we can import backend modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import VECTORSTORE_PATH, RAG_SAMPLE_SIZE, EMBEDDING_MODEL
from backend.data_loader import load_data
from backend.rag import create_documents, sample_data_for_rag, build_faiss_index


def main():
    print("=" * 60)
    print("  Mini AI Assistant — Build FAISS Index (LangChain)")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1/4] Loading dataset...")
    start = time.time()
    df = load_data()
    print(f"  Loaded {len(df):,} transactions in {time.time() - start:.1f}s")

    # Step 2: Sample
    print(f"\n[2/4] Sampling {RAG_SAMPLE_SIZE:,} representative transactions...")
    sample_df = sample_data_for_rag(df, RAG_SAMPLE_SIZE)
    print(f"  Sampled {len(sample_df):,} transactions")

    # Step 3: Create documents
    print("\n[3/4] Creating text documents...")
    start = time.time()
    documents = create_documents(sample_df)
    print(f"  Created {len(documents):,} documents in {time.time() - start:.1f}s")

    # Step 4: Build LangChain FAISS index
    print(f"\n[4/4] Building LangChain FAISS index using {EMBEDDING_MODEL}...")
    start = time.time()
    vectorstore = build_faiss_index(documents, VECTORSTORE_PATH)
    print(f"  Index built in {time.time() - start:.1f}s")

    print("\n" + "=" * 60)
    print("  FAISS index built successfully!")
    print(f"  Location: {VECTORSTORE_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
