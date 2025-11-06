"""
Utility helpers for data cleansing and transliteration used across the
Larth Etruscan/NLP code base.

The normalisation rules are derived from the description in the paper
“Larth: dataset and machine translation for Etruscan” (Table 1) and from
the preprocessing steps implemented in the original notebooks.  They
collapse the heterogeneous transcription symbols employed by ETP and
CIEP to a restricted Latin alphabet and standardise punctuation so that
the downstream tokenisers operate on a predictable character set.
"""

from __future__ import annotations

import math
import re
from typing import Dict

__all__ = [
    "to_latin",
    "to_extended_latin",
    "greek_to_latin",
    "others",
    "replace",
    "parenthesis_re_no_space",
    "parenthesis_re",
    "curly_brakets_re",
    "brakets_re",
    "low_brakets_re",
    "not_alphanum_re",
    "date_re",
    "T_re",
    "C_re",
    "A_re",
    "tags",
]

# ---------------------------------------------------------------------------
# Transliteration helpers
# ---------------------------------------------------------------------------

# NOTE: Longer keys appear before their substrings to avoid accidental
# partial replacements (e.g. θ should become "th" before σ → "s").
_TO_LATIN_PAIRS: Dict[str, str] = {
    # Multi-character replacements / separators
    "…": " ",
    "•": " ",
    # Greek letters and variants that appear in the corpora
    "Θ": "th",
    "θ": "th",
    "Φ": "ph",
    "φ": "ph",
    "Χ": "ch",
    "χ": "ch",
    "Σ": "s",
    "σ": "s",
    "ς": "s",
    # Latin characters with diacritics used in ETP/CIEP
    "ê": "e",
    "ḕ": "e",
    "ẹ": "e",
    "é": "e",
    "è": "e",
    "á": "a",
    "à": "a",
    "ó": "o",
    "ò": "o",
    "í": "i",
    "ì": "i",
    "ú": "u",
    "ù": "u",
    "ś": "s",
    "š": "sh",
    "ž": "zh",
    "ḥ": "h",
    "Ḥ": "h",
    "ṇ": "n",
    "Ṇ": "n",
    "ṭ": "t",
    "Ṭ": "t",
    "ṿ": "v",
    "Ṿ": "v",
    # Combining marks – they are removed entirely.
    "\u0323": "",
}

# A slightly more permissive variant that still collapses punctuation but keeps
# the Greek symbols (useful when a downstream consumer expects extended Latin).
_TO_EXTENDED_LATIN_PAIRS: Dict[str, str] = {
    **_TO_LATIN_PAIRS,
    # Restore the primary Greek graphemes so that the caller can decide whether
    # to transliterate them or leave them untouched.
    "Θ": "Θ",
    "θ": "θ",
    "Φ": "Φ",
    "φ": "φ",
    "Χ": "Χ",
    "χ": "χ",
    "Σ": "Σ",
    "σ": "σ",
    "ς": "ς",
}


def replace(text: str | float | None, mapping: Dict[str, str]) -> str | float | None:
    """
    Replace substrings according to ``mapping``.

    ``mapping`` may contain multi-character keys (e.g. θ → ``"th"``).  Keys are
    applied in descending order of length to favour longer matches first.
    Non-string inputs (``None`` or NaNs) are returned unchanged.
    """

    if text is None:
        return None
    if isinstance(text, float) and math.isnan(text):
        return text
    if not isinstance(text, str):
        return text

    for src in sorted(mapping.keys(), key=len, reverse=True):
        text = text.replace(src, mapping[src])
    return text


to_latin = _TO_LATIN_PAIRS
to_extended_latin = _TO_EXTENDED_LATIN_PAIRS

# Translation tables used with str.translate for quick character-level cleanup.
_greek_base = {
    "Α": "A",
    "Β": "B",
    "Γ": "G",
    "Δ": "D",
    "Ε": "E",
    "Ζ": "Z",
    "Η": "E",
    "Θ": "TH",
    "Ι": "I",
    "Κ": "K",
    "Λ": "L",
    "Μ": "M",
    "Ν": "N",
    "Ξ": "X",
    "Ο": "O",
    "Π": "P",
    "Ρ": "R",
    "Σ": "S",
    "Τ": "T",
    "Υ": "Y",
    "Φ": "PH",
    "Χ": "CH",
    "Ψ": "PS",
    "Ω": "O",
    "α": "a",
    "β": "b",
    "γ": "g",
    "δ": "d",
    "ε": "e",
    "ζ": "z",
    "η": "e",
    "θ": "th",
    "ι": "i",
    "κ": "k",
    "λ": "l",
    "μ": "m",
    "ν": "n",
    "ξ": "x",
    "ο": "o",
    "π": "p",
    "ρ": "r",
    "σ": "s",
    "ς": "s",
    "τ": "t",
    "υ": "y",
    "φ": "ph",
    "χ": "ch",
    "ψ": "ps",
    "ω": "o",
}

greek_to_latin = str.maketrans(_greek_base)

others = str.maketrans(
    {
        "·": " ",
        "•": " ",
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "«": '"',
        "»": '"',
        "–": "-",
        "—": "-",
        "‑": "-",
        "\u00a0": " ",
    }
)

# ---------------------------------------------------------------------------
# Regular expressions frequently used while cleaning intermediate data dumps.
# ---------------------------------------------------------------------------

parenthesis_re_no_space = re.compile(r"\(([^)\s]+)\)")
parenthesis_re = re.compile(r"\([^)]*\)")
curly_brakets_re = re.compile(r"\{[^}]*\}")
brakets_re = re.compile(r"\[[^\]]*\]")
low_brakets_re = re.compile(r"<[^>]*>")
not_alphanum_re = re.compile(r"[^0-9A-Za-z]+")
date_re = re.compile(r"\b\d{1,4}\s*(?:bce?|ce|bc|ad|b\.c\.|a\.d\.)\b", re.IGNORECASE)

# Keys extracted from the raw CIEP tables (useful when parsing those logs).
T_re = re.compile(r"^\s*T[\w\s\-\.\u03b8\u03a3]*$")
C_re = re.compile(r"^\s*C[\w\s\-\.\(\)\u03b8\u03a3]*$")
A_re = re.compile(r"^\s*A[\w\s\-\.\(\)\u03b8\u03a3]*$")

# Feature columns from ETP_POS.csv (kept here for quick reuse in notebooks).
tags = [
    "city name",
    "place name",
    "name",
    "epithet",
    "theo",
    "cogn",
    "prae",
    "nomen",
    "nom",
    "acc",
    "masc",
    "fem",
    "nas-part",
    "nasa-part",
    "u-part",
    "θ-impv",
    "θ-part",
    "θas-part",
    "as-part",
    "act",
    "pass",
    "non-past",
    "past",
    "impv",
    "jussive",
    "necess",
    "inanim",
    "anim",
    "indef",
    "def",
    "deictic particle",
    "enclitic particle",
    "enclitic conj",
    "dem",
    "adv",
    "art",
    "conj",
    "post",
    "pro",
    "rel",
    "subord",
    "neg",
    "num",
    "1st gen",
    "2nd gen",
    "1st abl",
    "2nd abl",
    "loc",
    "1st pert",
    "2nd pert",
    "1st pers",
    "2nd pers",
    "3rd pers",
    "pl",
    "gen",
    "abl",
    "pert",
    "TAG",
]

