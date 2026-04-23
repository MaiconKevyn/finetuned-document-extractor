import json
import re
from typing import Optional


def extract_json_from_text(text: str) -> Optional[dict]:
    """Extract the first JSON object found in text. Returns None on failure."""
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        return None
    return None
