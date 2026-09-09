"""
RAG module using LangChain for document creation, FAISS retrieval, and LLM explanation.
Uses LangChain's HuggingFaceEmbeddings (local) and HuggingFaceEndpoint (free LLM).
"""

import os
import json
import pandas as pd

from backend.config import (
    HF_TOKEN,
    HF_MODEL,
    EMBEDDING_MODEL,
    VECTORSTORE_PATH,
    TOP_K,
    RAG_SAMPLE_SIZE,
)

# ── Module-level caches ───────────────────────────────────
_vectorstore = None
_embeddings = None


# ═══════════════════════════════════════════════════════════
# Document Creation
# ═══════════════════════════════════════════════════════════

def create_documents(df: pd.DataFrame) -> list[str]:
    """
    Convert transaction rows into readable text documents for RAG.

    Args:
        df: Cleaned DataFrame (should be a sample for RAG, not the full dataset).

    Returns:
        List of human-readable document strings.
    """
    documents = []

    for _, row in df.iterrows():
        customer_id = row.get("CustomerID", "Unknown")
        if pd.isna(customer_id):
            customer_id = "Unknown"
        else:
            customer_id = str(int(customer_id))

        country = str(row.get("Country", "Unknown"))
        description = str(row.get("Description", "Unknown"))
        quantity = int(row.get("Quantity", 0))
        unit_price = float(row.get("UnitPrice", 0.0))
        revenue = float(row.get("Revenue", 0.0))

        doc = (
            f"Customer {customer_id} from {country} purchased {quantity} units "
            f"of {description}, at a unit price of £{unit_price:.2f}, "
            f"generating £{revenue:.2f} in revenue."
        )
        documents.append(doc)

    return documents


def sample_data_for_rag(df: pd.DataFrame, sample_size: int | None = None) -> pd.DataFrame:
    """
    Create a representative sample of the dataset for RAG indexing.

    Args:
        df: Full cleaned DataFrame.
        sample_size: Number of rows to sample. Defaults to RAG_SAMPLE_SIZE from config.

    Returns:
        Sampled DataFrame.
    """
    size = sample_size or RAG_SAMPLE_SIZE

    if len(df) <= size:
        return df

    return df.sample(n=size, random_state=42).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════
# LangChain Embeddings (Local)
# ═══════════════════════════════════════════════════════════

def get_embeddings():
    """Get the LangChain HuggingFace embeddings model (cached, runs locally)."""
    global _embeddings
    if _embeddings is None:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


# ═══════════════════════════════════════════════════════════
# FAISS Vector Store (via LangChain)
# ═══════════════════════════════════════════════════════════

def build_faiss_index(documents: list[str], save_path: str | None = None):
    """
    Build a LangChain FAISS vector store from documents and save to disk.

    Args:
        documents: List of text documents to embed.
        save_path: Directory to save the index. Defaults to VECTORSTORE_PATH.

    Returns:
        LangChain FAISS vector store.
    """
    from langchain_community.vectorstores import FAISS

    path = save_path or VECTORSTORE_PATH
    embeddings = get_embeddings()

    print(f"  Encoding {len(documents)} documents with LangChain FAISS...")
    vectorstore = FAISS.from_texts(documents, embeddings)

    # Save to disk
    os.makedirs(path, exist_ok=True)
    vectorstore.save_local(path)

    # Save metadata
    metadata = {
        "num_documents": len(documents),
        "embedding_model": EMBEDDING_MODEL,
    }
    with open(os.path.join(path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Index saved to {path} ({len(documents)} documents)")

    return vectorstore


def load_faiss_index(load_path: str | None = None):
    """
    Load a pre-built LangChain FAISS vector store from disk.

    Returns:
        LangChain FAISS vector store.
    """
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    from langchain_community.vectorstores import FAISS

    path = load_path or VECTORSTORE_PATH
    index_file = os.path.join(path, "index.faiss")

    if not os.path.exists(index_file):
        raise FileNotFoundError(
            f"FAISS index not found at '{path}'. "
            f"Run 'python scripts/build_index.py' first."
        )

    embeddings = get_embeddings()
    _vectorstore = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

    return _vectorstore


# ═══════════════════════════════════════════════════════════
# Retrieval
# ═══════════════════════════════════════════════════════════

def retrieve(query: str, top_k: int | None = None) -> list[str]:
    """
    Retrieve the most relevant documents for a query using LangChain FAISS.

    Args:
        query: User question.
        top_k: Number of documents to retrieve. Defaults to TOP_K from config.

    Returns:
        List of relevant document strings.
    """
    k = top_k or TOP_K
    vectorstore = load_faiss_index()

    results = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in results]


# ═══════════════════════════════════════════════════════════
# LLM Generation (via LangChain + HuggingFace)
# ═══════════════════════════════════════════════════════════

# Strict RAG prompt — the LLM must not invent data
_SYSTEM_PROMPT = """You are a retail business analysis assistant.
Answer the user's question using ONLY the provided retrieved transaction context.
Do not invent facts.
Do not invent numerical values.
Do not calculate global metrics yourself.
If the retrieved context is insufficient, say so.
Explain observed purchasing behavior and patterns clearly."""


from typing import Optional, List, Any
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from huggingface_hub import InferenceClient


class HuggingFaceRouterLLM(LLM):
    """LangChain LLM implementation using Hugging Face's serverless InferenceClient."""
    # model_name: str = HF_MODEL or "Qwen/Qwen3.8-27B"
    model_name: str = "Qwen/Qwen3.8-27B"
    max_tokens: int = 512
    temperature: float = 0.3

    @property
    def _llm_type(self) -> str:
        return "huggingface_router"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        client = InferenceClient(token=HF_TOKEN)
        # Prioritize configured model, followed by verified serverless free models
        models_to_try = [self.model_name]
        for fallback in [
            "meta-llama/Llama-3.1-8B-Instruct",
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            "meta-llama/Llama-3.3-70B-Instruct",
        ]:
            if fallback not in models_to_try:
                models_to_try.append(fallback)

        last_error = None
        for model in models_to_try:
            try:
                response = client.chat_completion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                content = response.choices[0].message.content
                if content:
                    return content.strip()
            except Exception as e:
                last_error = e
                continue

        if last_error:
            raise last_error
        return ""


def _get_llm():
    """
    Get the LangChain LLM instance for Hugging Face inference.
    """
    return HuggingFaceRouterLLM(
        model_name=HF_MODEL or "meta-llama/Llama-3.1-8B-Instruct",
        max_tokens=512,
        temperature=0.3,
    )


def generate_answer(question: str, context: str) -> str:
    """
    Generate an LLM answer using LangChain + Hugging Face.

    Args:
        question: The user's question.
        context: Retrieved transaction context.

    Returns:
        Generated answer string, or a fallback message if the LLM is unavailable.
    """
    if not HF_TOKEN:
        return (
            "LLM service is unavailable. Please configure HF_TOKEN in your .env file "
            "or use the analytics questions (e.g., 'What is total revenue?')."
        )

    try:
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        llm = _get_llm()

        
        template = (
            "{system}\n\n"
            "Retrieved context:\n{context}\n\n"
            "User question:\n{question}\n\n"
            "Answer:"
        )
        prompt = PromptTemplate.from_template(template)

        chain = prompt | llm | StrOutputParser()

        response = chain.invoke({
            "system": _SYSTEM_PROMPT,
            "context": context,
            "question": question,
        })

        return response.strip() if response else "No response generated."

    except Exception as e:
        return (
            "The analytical components are working, but the free LLM service is "
            "currently unavailable. Please configure a valid Hugging Face token/model "
            f"or retry later. (Error: {type(e).__name__})"
        )


def ask_rag(question: str) -> dict:
    """
    Full RAG pipeline: retrieve context → generate answer.

    Returns:
        dict with question, type, answer, and retrieved_documents count.
    """
    try:
        retrieved_docs = retrieve(question)
    except FileNotFoundError as e:
        return {
            "question": question,
            "type": "rag",
            "answer": str(e),
            "retrieved_documents": 0,
        }

    context = "\n".join(retrieved_docs)
    answer = generate_answer(question, context)

    return {
        "question": question,
        "type": "rag",
        "answer": answer,
        "retrieved_documents": len(retrieved_docs),
    }
