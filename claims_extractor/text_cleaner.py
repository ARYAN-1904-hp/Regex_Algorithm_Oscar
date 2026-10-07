"""
OCR text cleaner for plain-text screenshot dumps.

The screenshot OCR in these batches has a consistent set of character-level
mistakes ("thls" for "this", "O7/12/2026" for "07/12/2026", glued labels
like "Memberresponsibility" / "EDIfle"). The regex extractor is far more
reliable on text with those repaired, so every file is passed through
clean_ocr_text() before any field extraction runs.

Every rule here is deliberately narrow: whole-word replacements of known
misreads, and digit repairs only inside date/money shapes, so real values
are never rewritten.
"""

from __future__ import annotations

import re
from typing import List, Tuple

# ---------------------------------------------------------------------------
# 1. Whole-word OCR misreads ("l"/"1" for "i", dropped spaces in labels).
#    Matched case-insensitively on word boundaries; the replacement keeps the
#    capitalisation of the first letter of the original.
# ---------------------------------------------------------------------------
_WORD_FIXES: List[Tuple[str, str]] = [
    (r"thls", "this"),
    (r"patlent", "patient"),
    (r"verslon", "version"),
    (r"actlons", "actions"),
    (r"medicald", "medicaid"),
    (r"submisslon", "submission"),
    (r"detalls", "details"),
    (r"blrth", "birth"),
    (r"famlly", "family"),
    (r"vlew", "view"),
    (r"cllent", "client"),
    (r"pald", "paid"),
    (r"subsldy", "subsidy"),
    (r"facillity", "facility"),
    (r"responsibillity", "responsibility"),
    (r"responsibilty", "responsibility"),
    (r"communlcatlons", "communications"),
    (r"communicatlons", "communications"),
    (r"emall", "email"),
    (r"appolntment", "appointment"),
    (r"bllling", "billing"),
    (r"suffx", "suffix"),
    (r"connrmchande", "confirm change"),
]

# Labels that OCR glues together (no space) -- restored to their UI spelling.
_GLUED_LABELS: List[Tuple[str, str]] = [
    (r"Memberresponsibility", "Member responsibility"),
    (r"Interestaccrued", "Interest accrued"),
    (r"EDI\s?f[il1]?le", "EDI file"),
    (r"Valet\s?f[il1]?le\s?I?D:?", "Valet file ID:"),
    (r"Staytype", "Stay type"),
    (r"Claimtype", "Claim type"),
    (r"Place\s?ofservice", "Place of service"),
    (r"Memberinfo", "Member info"),
    (r"Webaccount", "Web account"),
    (r"Enrollmenttype", "Enrollment type"),
    (r"Eligib\s?[Il1]lity", "Eligibility"),
    (r"OSCID", "OSC ID"),
    (r"OcelotID", "Ocelot ID"),
    (r"Firstname", "First name"),
    (r"Oscarplan", "Oscar plan"),
    (r"Internationairroviaer", "International Provider"),
    (r"Fil ters", "Filters"),
]

_DATE_CHARS = "0-9OoIl|"
# A date-shaped token whose digits OCR partly read as letters: O7/12/2026.
_DIRTY_DATE_RE = re.compile(
    rf"(?<![A-Za-z0-9])([{_DATE_CHARS}]{{1,2}})/([{_DATE_CHARS}]{{1,2}})/([{_DATE_CHARS}]{{2,4}})(?![A-Za-z0-9])"
)
# "$1,087.00from" -> "$1,087.00 from"; "$o" (letter o for zero) -> "$0".
_MONEY_GLUED_RE = re.compile(r"(\$[\d,]+\.\d{2})(?=[A-Za-z(])")
_DOLLAR_O_RE = re.compile(r"\$[oO](?=\s|$)")
# Date ranges glued by OCR: "05/27/202612/31/2026" -> "05/27/2026 - 12/31/2026".
_GLUED_DATES_RE = re.compile(r"(\d{1,2}/\d{1,2}/\d{4})(?=\d{1,2}/\d{1,2}/\d{4})")
# "(5 of5)" / "(O of 0)" -> "(5 of 5)" / "(0 of 0)".
_VERSION_RE = re.compile(r"\(\s*([0-9Oo]+)\s*of\s*([0-9Oo]+)\s*\)")
# "Cortex-Claim #X" / "Cortex - Claim # X" -> "Cortex - Claim #X".
_CORTEX_HEADER_RE = re.compile(r"Cortex\s*-\s*Claim\s*#\s*", re.IGNORECASE)
# Claim type renderings: "UB  Facility", "UBFacility", "UB - Facility".
_UB_RE = re.compile(r"\bUB\s*[-–—]?\s*Facility\b", re.IGNORECASE)
_HCFA_RE = re.compile(r"\bHCFA\s*[-–—]?\s*Profes\w*", re.IGNORECASE)
# Zero-width / non-breaking characters OCR sometimes emits.
_INVISIBLE_RE = re.compile(r"[​‌‍﻿]")


def _case_like(src: str, repl: str) -> str:
    if src.isupper() and len(src) > 1:
        return repl.upper()
    if src[:1].isupper():
        return repl[:1].upper() + repl[1:]
    return repl


def _fix_word(pattern: str, repl: str):
    rx = re.compile(rf"\b{pattern}\b", re.IGNORECASE)
    return lambda s: rx.sub(lambda m: _case_like(m.group(0), repl), s)


def _fix_claim(s: str) -> str:
    rx = re.compile(r"\bcl[a@]lm(s?)\b", re.IGNORECASE)
    return rx.sub(lambda m: _case_like(m.group(0), "claim" + m.group(1).lower()), s)


_WORD_FIXERS = [_fix_word(p, r) for p, r in _WORD_FIXES]
_GLUED_FIXERS = [
    (re.compile(rf"\b{p}", re.IGNORECASE if p[0].islower() else 0), r) for p, r in _GLUED_LABELS
]


def _repair_date(m: re.Match) -> str:
    parts = [m.group(1), m.group(2), m.group(3)]
    # Only rewrite when at least one real digit is present in every part --
    # otherwise this is not a date at all (e.g. "I/O/lol").
    if not all(re.search(r"\d", p) for p in parts):
        return m.group(0)
    fixed = [p.translate(str.maketrans("OoIl|", "00111")) for p in parts]
    return "/".join(fixed)


def clean_line(line: str) -> str:
    s = _INVISIBLE_RE.sub("", line.replace("\xa0", " ").replace("\t", " "))
    s = _fix_claim(s)
    for fix in _WORD_FIXERS:
        s = fix(s)
    for rx, repl in _GLUED_FIXERS:
        s = rx.sub(repl, s)
    s = _CORTEX_HEADER_RE.sub("Cortex - Claim #", s)
    s = _VERSION_RE.sub(lambda m: f"({m.group(1).replace('O', '0').replace('o', '0')} of {m.group(2).replace('O', '0').replace('o', '0')})", s)
    s = _GLUED_DATES_RE.sub(r"\1 - ", s)
    s = _DIRTY_DATE_RE.sub(_repair_date, s)
    s = _DOLLAR_O_RE.sub("$0", s)
    s = _MONEY_GLUED_RE.sub(r"\1 ", s)
    s = _UB_RE.sub("UB - Facility", s)
    s = _HCFA_RE.sub("HCFA - Professional", s)
    s = s.lstrip(">")  # stray UI caret: ">Member responsibility"
    s = re.sub(r"[ ]{2,}", " ", s)
    return s.strip()


def clean_ocr_text(raw_text: str) -> str:
    """Return a cleaned copy of a plain-text OCR dump: line endings
    normalised, known OCR misreads repaired, blank lines dropped."""
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [clean_line(l) for l in text.split("\n")]
    return "\n".join(l for l in lines if l)
