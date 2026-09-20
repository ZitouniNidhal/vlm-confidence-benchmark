import re


QUALITATIVE_CONFIDENCE_MAP = [
    (r"\b(?:very high|extremely high|almost certain)\b", 95),
    (r"\b(?:high|highly|fairly high|quite confident)\b", 90),
    (r"\b(?:medium|moderate|moderately|somewhat)\b", 60),
    (r"\b(?:very low|extremely low)\b", 10),
    (r"\b(?:low|unlikely|slight|low confidence)\b", 30),
    (r"\b(?:uncertain|unsure|not confident|not certain)\b", 20),
]


def extract_verbalized_confidence(answer_text: str) -> int | None:
    """
    Extract a confidence percentage (0-100) from model output text.
    Handles patterns like:
      - "80%", "80 percent", "confidence: 85%", "confidence: 0.85"
      - Fractions: "9/10", "85/100", "9 out of 10", "85 out of 100"
      - Qualitative terms: "high confidence", "medium confidence", "low confidence", "uncertain"
    """
    if not answer_text:
        return None

    # Match percentage patterns e.g. "85%", "85 percent"
    pct_match = re.search(r"(\d{1,3})\s*(?:%|percent\b)", answer_text, re.IGNORECASE)
    if pct_match:
        val = int(pct_match.group(1))
        return min(100, max(0, val))

    # Match fraction patterns e.g. "85/100", "9/10", "85 out of 100"
    frac_match = re.search(r"(\d{1,3})\s*(?:/|out of)\s*(\d{1,3})", answer_text, re.IGNORECASE)
    if frac_match:
        num, den = float(frac_match.group(1)), float(frac_match.group(2))
        if den > 0:
            ratio = min(1.0, max(0.0, num / den))
            return int(round(ratio * 100))

    # Match explicit confidence word followed by score e.g. "confidence of 85", "confidence: 80"
    conf_word_match = re.search(r"(?:confidence|certainty)\s*(?:is|of|:)?\s*(\d{1,3})(?!\d|\.\d)", answer_text, re.IGNORECASE)
    if conf_word_match:
        val = int(conf_word_match.group(1))
        if 0 <= val <= 100:
            return val

    # Match decimal probability e.g. "confidence of 0.85"
    dec_match = re.search(r"(?:confidence|certainty)\s*(?:is|of|:)?\s*(0\.\d+)", answer_text, re.IGNORECASE)
    if dec_match:
        val = float(dec_match.group(1))
        return min(100, max(0, int(round(val * 100))))

    # Match qualitative confidence phrases e.g. "high confidence", "uncertain"
    for pattern, val in QUALITATIVE_CONFIDENCE_MAP:
        if re.search(pattern, answer_text, re.IGNORECASE):
            return val

    return None


def normalize_confidence(confidence: int | None) -> float:
    """Normalize 0-100 confidence value to 0.0-1.0 float."""
    if confidence is None:
        return 0.0
    return float(confidence / 100.0)


def has_confidence_statement(answer_text: str) -> bool:
    """Check if the text contains an explicit confidence statement."""
    return extract_verbalized_confidence(answer_text) is not None
