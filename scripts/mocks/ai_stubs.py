"""Simple deterministic AI stubs for offline development and tests."""
from typing import List, Dict, Any

def bert_classify(text: str) -> float:
    """
    Return a deterministic vendor probability based on simple heuristics.
    This is only for offline testing; real implementation must call remote API.
    """
    text_lower = text.lower()
    score = 0.0
    keywords = ["invoice", "payment", "quote", "proposal", "contract", "purchase order", "po"]
    for kw in keywords:
        if kw in text_lower:
            score += 0.2
    # clamp
    if score > 1.0:
        score = 1.0
    return score

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Return zero vectors of fixed length (for FAISS test)."""
    dim = 8
    return [[0.0] * dim for _ in texts]

def llm_extract(prompt: str) -> Dict[str, Any]:
    """Return a minimal valid extraction for testing purposes."""
    return {
        "person_name": None,
        "person_email": None,
        "phone_numbers": [],
        "organization": None,
        "domain": None,
        "category": "other",
        "confidence": 0.0,
        "evidence": []
    }


