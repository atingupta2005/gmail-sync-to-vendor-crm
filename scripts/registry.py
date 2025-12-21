import json
import os
from typing import Dict, Optional, Any

def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

def append_registry_entry(registry_path: str, entry: Dict[str, Any]) -> None:
    ensure_dir(registry_path)
    with open(registry_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

def load_registry_index(registry_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load the registry JSONL and return a mapping email_id -> latest entry.
    This is simple and safe for script-first development; for large registries
    consider an indexed DB later.
    """
    if not os.path.exists(registry_path):
        return {}
    mapping: Dict[str, Dict[str, Any]] = {}
    with open(registry_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            email_id = rec.get("email_id")
            if not email_id:
                continue
            mapping[email_id] = rec
    return mapping

def get_registry_entry(registry_path: str, email_id: str) -> Optional[Dict[str, Any]]:
    mapping = load_registry_index(registry_path)
    return mapping.get(email_id)


