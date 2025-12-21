#!/usr/bin/env python3

import os
import requests
from dotenv import load_dotenv
from pinecone import Pinecone
from typing import List

# ---------- config ----------
INDEX_NAME = "vendor-emails"
EMBED_ENDPOINT = "https://router.huggingface.co/hf-inference/models/BAAI/bge-base-en-v1.5"
TOP_K = 5
# ----------------------------


def mean_pool(token_embeddings: List[List[float]]) -> List[float]:
    dim = len(token_embeddings[0])
    pooled = [0.0] * dim
    for vec in token_embeddings:
        for i, v in enumerate(vec):
            pooled[i] += v
    return [v / len(token_embeddings) for v in pooled]



def embed_query(text: str) -> List[float]:
    headers = {
        "Authorization": f"Bearer {os.environ['HF_TOKEN']}",
        "Content-Type": "application/json",
    }
    payload = {"inputs": text}

    resp = requests.post(EMBED_ENDPOINT, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()

    data = resp.json()

    # HF router returns token embeddings → mean pool
    if isinstance(data, list) and isinstance(data[0], list):
        return mean_pool(data)

    raise ValueError(f"Unexpected embedding response: {type(data)}")

    pooled = [0.0] * dim
    for token_vec in data:
        for i, v in enumerate(token_vec):
            pooled[i] += v
    return [v / len(data) for v in pooled]


def main():
    load_dotenv(".env")

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(INDEX_NAME)

    print("🔎 Semantic search ready (Ctrl+C to exit)\n")

    while True:
        query = input("Enter search text: ").strip()
        if not query:
            continue

        vector = embed_query(query)

        res = index.query(
            vector=vector,
            top_k=TOP_K,
            include_metadata=True,
        )

        print("\n--- RESULTS ---")
        for match in res["matches"]:
            meta = match.get("metadata", {})
            print(f"\nScore: {match['score']:.4f}")
            print("Email ID:", meta.get("email_id"))
            print("Chunk:", meta.get("chunk_index"))
            print("Subject:", meta.get("subject"))
            print("Sender domain:", meta.get("sender_domain"))
        print("----------------\n")


if __name__ == "__main__":
    main()
