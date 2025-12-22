import json
from pathlib import Path

OLD = Path("data/state/processing_registry.jsonl")
NEW = Path("data/state/processing_registry_step2b_vendor_scoring.jsonl")

seen = set()

NEW.parent.mkdir(parents=True, exist_ok=True)

with OLD.open("r", encoding="utf-8") as fin, NEW.open("w", encoding="utf-8") as fout:
    for line in fin:
        try:
            rec = json.loads(line)
        except Exception:
            continue

        if rec.get("last_completed_step") != "step2b_vendor_scoring":
            continue

        email_id = rec.get("email_id")
        if not email_id or email_id in seen:
            continue

        model_versions = rec.get("model_versions") or {}
        model_version = model_versions.get("step2b_model_version")
        if not model_version:
            continue

        fout.write(json.dumps({
            "email_id": email_id,
            "content_hash": rec.get("content_hash"),
            "last_completed_step": "step2b_vendor_scoring",
            "model_versions": {"step2b_model_version": model_version},
            "timestamp": rec.get("updated_at") or rec.get("timestamp"),
            "error": None,
        }) + "\n")

        seen.add(email_id)

print(f"Migrated {len(seen)} Step-2B records")
