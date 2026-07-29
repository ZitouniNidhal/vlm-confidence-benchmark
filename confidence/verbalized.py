import re

def extract_verbalized_confidence(answer_text: str) -> int | None:
    match = re.search(r"(\d{1,3})\s*%", answer_text)
    if match:
        value = int(match.group(1))
        return min(100, max(0, value))
    return None

def normalize_confidence(confidence: int | None) -> float:
    if confidence is None:
        return 0.0
    return confidence / 100.0

def has_confidence_statement(answer_text: str) -> bool:
    return bool(re.search(r"\d{1,3}\s*%", answer_text))
