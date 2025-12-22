import json
from pathlib import Path

OLD = Path("data/state/processing_registry.jsonl")
NEW = Path("data/state/processing_registry_step3_cleanup.jsonl")

seen = set()
NEW.parent.mkdir(parents=True, exist_ok=True)

with OLD.open() as fin, NEW.open("w") as fout:
    for line in fin:
        try:
            rec = json.loads(line)
        except Exception:
            continue

        if rec.get("last_completed_step") != "step3_cleanup":
            continue

        eid = rec.get("email_id")
        if not eid or eid in seen:
            continue

        fout.write(json.dumps(rec) + "\n")
        seen.add(eid)

print(f"Migrated {len(seen)} cleanup records")
