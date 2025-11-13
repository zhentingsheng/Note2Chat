from typing import Any, Dict, List
import json


def read_json(path: str) -> Any:
    """Read a JSON file and return its parsed content."""
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a *JSON Lines* file and return a list of items."""
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items