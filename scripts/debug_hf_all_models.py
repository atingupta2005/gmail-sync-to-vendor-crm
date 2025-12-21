from dotenv import load_dotenv
import os
import requests

load_dotenv(".env")

HF_TOKEN = os.getenv("HF_TOKEN")
assert HF_TOKEN, "HF_TOKEN missing"

MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "intfloat/e5-base-v2",
    "intfloat/multilingual-e5-base",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-small-en-v1.5",
]

TEXT = "This is a test email about training delivery and invoice."

for model in MODELS:
    print("\n==============================")
    print("MODEL:", model)

    url = f"https://router.huggingface.co/hf-inference/models/{model}"

    try:
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/json",
            },
            json={"inputs": [TEXT]},
            timeout=30,
        )

        print("STATUS:", resp.status_code)

        if resp.status_code != 200:
            print("BODY:", resp.text[:200])
            continue

        data = resp.json()

        if isinstance(data, list) and isinstance(data[0], list):
            print("VECTOR DIM:", len(data[0]))
            print("✅ WORKS")
        else:
            print("❌ Unexpected response format")

    except Exception as e:
        print("❌ ERROR:", str(e))
