import os
import requests
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv(".env")

HF_ENDPOINT = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
HF_TOKEN = os.environ["HF_TOKEN"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME = "vendor-emails-test"  # use test index

text = "Test email about invoice and payment discussion."

# ---- Call HF embedding API ----
resp = requests.post(
    HF_ENDPOINT,
    headers={
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    },
    json={"inputs": text},
    timeout=30,
)
resp.raise_for_status()
data = resp.json()

print("HF raw response type:", type(data))
print("Token count:", len(data))
print("Vector dim:", len(data[0]))

# Mean pooling
dim = len(data[0])
pooled = [0.0] * dim
for token_vec in data:
    for i, v in enumerate(token_vec):
        pooled[i] += v
pooled = [v / len(data) for v in pooled]

print("Pooled vector dim:", len(pooled))

# ---- Pinecone upsert ----
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

index.upsert(
    vectors=[
        {
            "id": "test-email-1",
            "values": pooled,
            "metadata": {"source": "manual-test"},
        }
    ]
)

stats = index.describe_index_stats()
print("Pinecone stats after upsert:", stats)
