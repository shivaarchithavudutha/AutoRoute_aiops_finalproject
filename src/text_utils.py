"""Text cleaning shared by the pipeline, the model wrapper and the gateway.

Keeping ONE cleaning function everywhere prevents train/serve skew.
The function is idempotent: cleaning already-clean text changes nothing.
"""
import re

_REDACTED = re.compile(r"x{2,}", re.IGNORECASE)      # CFPB masks personal data as XXXX
_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_SPACES = re.compile(r"\s+")


def clean_text(text) -> str:
    """Lowercase, drop XXXX redactions and punctuation, collapse whitespace."""
    if text is None:
        return ""
    text = str(text).lower()
    text = _REDACTED.sub(" ", text)
    text = _NON_ALNUM.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def tokens(text: str):
    """Split cleaned text into words."""
    return clean_text(text).split()
