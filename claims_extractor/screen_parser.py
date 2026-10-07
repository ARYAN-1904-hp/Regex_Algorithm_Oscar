"""
Screen-aware regex extractor for PLAIN-TEXT screenshot OCR dumps.

field_extractor's geometric label->value pairing needs [BOXES] bounding-box
data. The batch .txt files carry only flattened text lines, where a two-
column screen is merged into one line ("Service start date 05/29/2026
Received date 07/18/2026") -- the fuzzy "next token" pairing then grabs the
neighbouring column (Tenant = "oscar ACA requirement (plus 31/60/90 Days").

This module instead:
  1. classifies the screen from its header line (Cortex claim, Appian
     ticket, Appian queue, Rosco member/search, VERINT, other), and
  2. pulls each field with a label-anchored regex whose VALUE pattern is
     strict (a date, a $ amount, an 8-char ID, a closed vocabulary), so
     text from the neighbouring column can never be captured.

Output is a field_extractor.DocumentResult using the same schema field
names, so the existing scenario linking / gap fill / workbook builder run
unchanged on top of it.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .field_extractor import DocumentResult, _contains_negated_word
from .pattern_learner import PERSON, PatternLearner, clean_person, fold
from .text_cleaner import clean_ocr_text

# ---------------------------------------------------------------------------
# Screen types (shown in the workbook's "Detected Application" column)
# ---------------------------------------------------------------------------
APP_CORTEX = "Cortex- About this claim"
APP_CORTEX_EMPTY = "Cortex - Blank/loading page"
APP_APPIAN = "Appian- Home"
APP_APPIAN_QUEUE = "Appian - Queue"
APP_ROSCO = "Rosco"
APP_ROSCO_SEARCH = "Rosco - Search"
APP_VERINT = "VERINT"
APP_GMAIL = "Gmail"
APP_LOOKER = "Looker Studio"
APP_BLANK = "Blank page"
APP_OTHER = "Unknown"

# ---------------------------------------------------------------------------
# Shared value shapes
# ---------------------------------------------------------------------------
DATE = r"\d{1,2}/\d{1,2}/\d{4}"
MONEY = r"\$[\d,]+\.\d{2}"
DATETIME = rf"{DATE}\s+\d{{1,2}}:\d{{2}}\s*[AP]M"
CLAIM_ID = r"[A-Z0-9]{8}"
# Any agent's name -- unicode letters, particles ("de la"), O'Connor,
# OCR-glued "JessicaBrister". Single spaces only, so a name never runs onto
# the next line's label.
NAME = PERSON

_CLAIM_ID_RE = re.compile(rf"^{CLAIM_ID}$")
_TICKET_RE = re.compile(r"\bH[O0]LD\s?-?\s?(\d{7})(?!\d)")
_OSC_RE = re.compile(r"\bOSC\s?(\d{8})\s?-?\s?(\d{2})(?!\d)")

# Words that start the next column after "Member: <name>" on a Cortex page.
_MEMBER_NAME_STOP = {
    "Service", "About", "Valet", "Payment", "See", "Claim", "Notes", "Note",
    "No", "Received", "Billed", "Submission", "Tenant", "Diagnosis",
    "Pay", "Payer", "Prompt", "Charge", "Potential", "Interest", "Is",
}


def _first(pattern: str, text: str, flags: int = 0, group: int = 1) -> Optional[str]:
    m = re.search(pattern, text, flags)
    return m.group(group).strip() if m else None


def _valid_claim_id(v: Optional[str]) -> bool:
    if not v or not _CLAIM_ID_RE.match(v):
        return False
    # Must mix letters and digits -- rules out words like "NEWBORNS" and
    # all-digit numbers.
    if not (re.search(r"[A-Z]", v) and re.search(r"\d", v)):
        return False
    return not _contains_negated_word(v)


def _clean_name(raw: Optional[str]) -> Optional[str]:
    return clean_person(raw)


def _osc_ids(text: str) -> List[str]:
    """All OSC member IDs, normalised to OSC + 10 digits (Rosco renders
    'OSC75847221-02', Cortex/Appian render 'OSC7584722102')."""
    out: List[str] = []
    for m in _OSC_RE.finditer(text):
        v = f"OSC{m.group(1)}{m.group(2)}"
        if v not in out:
            out.append(v)
    return out


def _ticket_ids(text: str) -> List[str]:
    out: List[str] = []
    for m in _TICKET_RE.finditer(text):
        v = f"HOLD-{m.group(1)}"
        if v not in out:
            out.append(v)
    return out


# ---------------------------------------------------------------------------
# 1. Classification
# ---------------------------------------------------------------------------

_LEAD_MARKERS = ("cortex", "claims home inventory", "rosco")


def classify_screen(text: str, learner: Optional[PatternLearner] = None) -> str:
    lines = text.split("\n")
    head = " ".join(lines[:3]).lower()
    if not text.strip() or re.fullmatch(r"\s*<page \d+>\s*", text.lower()):
        return APP_BLANK
    # Headers are compared OCR-folded ("C0rtex", "Clalms Home lnventory");
    # the learner also fuzzy-matches new header variants and remembers them.
    fhead = fold(head, keep_spaces=True)
    lead = learner.match_screen_header(head) if learner else None
    if lead is None:
        lead = next((m for m in _LEAD_MARKERS if fhead.startswith(fold(m, keep_spaces=True))), None)
    if lead in ("verint", "knowledge management") or any(
        fold(m, keep_spaces=True) in fhead for m in ("verint", "knowledge management")
    ):
        return APP_VERINT
    if lead == "cortex":
        if re.search(r"Claim #\s*(Latest|Dashboard)", lines[0]) or len(lines) <= 2:
            return APP_CORTEX_EMPTY
        return APP_CORTEX
    if lead == "claims home inventory":
        if _TICKET_RE.search(text):
            return APP_APPIAN
        return APP_APPIAN_QUEUE
    if lead == "rosco":
        # Rosco search results: "rosco Q OSC74885861-01 ..." + MEMBERS table.
        if re.search(r"\bMEMBERS\b.*\bOSCAR ID\b", text) or re.match(r"\S+\s*q\s*osc", head):
            return APP_ROSCO_SEARCH
        return APP_ROSCO
    if lead == "gmail" or fold("gmail") in fhead:
        return APP_GMAIL
    if lead == "looker studio" or fold("looker studio", keep_spaces=True) in fhead:
        return APP_LOOKER
    return APP_OTHER


# ---------------------------------------------------------------------------
# 2. Cortex claim screen
# ---------------------------------------------------------------------------

_MONEY_LABELS = [
    ("Billed", r"\bBilled\b"),
    ("Allowed", r"\bAllowed\b"),
    ("Pay provider", r"\bPay provider\b"),
    ("Pay member", r"\bPay member\b"),
    ("Member responsibilty", r"\bMember responsibility\b"),
]


def _money_fields(text: str) -> Dict[str, str]:
    """Pair the claim's money labels with their $ amounts, line by line.

    A label directly followed by an amount takes it. OCR also produces
    lines like 'Pay provider Allowed $819.45 (30%) $819.45' where several
    labels come first and the amounts follow -- leftover labels then take
    the leftover amounts in reading order."""
    out: Dict[str, str] = {}
    for line in text.split("\n"):
        labels: List[Tuple[int, int, str]] = []
        for canon, rx in _MONEY_LABELS:
            for m in re.finditer(rx, line, re.IGNORECASE):
                labels.append((m.start(), m.end(), canon))
        if not labels:
            continue
        labels.sort()
        amounts = [(m.start(), m.group(0)) for m in re.finditer(MONEY, line)]
        used = set()
        pending: List[Tuple[int, str]] = []
        for start, end, canon in labels:
            # Directly adjacent amount (only spaces between label and $).
            adj = next(
                (i for i, (pos, _a) in enumerate(amounts)
                 if i not in used and pos >= end and not line[end:pos].strip()),
                None,
            )
            if adj is not None:
                used.add(adj)
                out.setdefault(canon, amounts[adj][1])
            else:
                pending.append((end, canon))
        for end, canon in pending:
            nxt = next(
                (i for i, (pos, _a) in enumerate(amounts) if i not in used and pos >= end),
                None,
            )
            if nxt is not None:
                used.add(nxt)
                out.setdefault(canon, amounts[nxt][1])
    return out


def _cortex_fields(text: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, str]]:
    f: Dict[str, Any] = {}
    x: Dict[str, Any] = {}
    flags: Dict[str, str] = {}
    lines = text.split("\n")

    # Claim ID: the page header is authoritative; the "XXXXXXXX About this
    # claim" title and the claim-lineage row are fallbacks.
    cands = [
        _first(rf"Cortex - Claim #\s*({CLAIM_ID})\b", text),
        _first(rf"^({CLAIM_ID}) About this claim", text, re.M),
        _first(rf"^({CLAIM_ID})\s+(?:{DATE}\s+)?(?:Blocked|Audit)\b", text, re.M),
    ]
    cid = next((c for c in cands if _valid_claim_id(c)), None)
    if cid:
        f["Claim ID"] = cid
    else:
        flags["Claim ID"] = "No valid Claim ID in Cortex header -- needs manual review"

    v = _first(r"Latest version \((\d+ of \d+)\)", text)
    if v:
        x["Claim Version"] = v
    # Full name from the Patient tab beats the (often truncated) header;
    # "0 Notes" is frequently OCR'd as "o Notes".
    v = _first(r"\bMember ([A-Z][a-z]+(?: [A-Z][a-zA-Z'\-]+)+) Change\b", text)
    if not v:
        # Header "Member: <name>" -- keep capitalised words only, stopping
        # at the first label word from the neighbouring column.
        head = _first(r"Member:[ ]*(.+)$", text, re.M) or ""
        words = []
        for w in head.split():
            w = w.rstrip(".")
            if not re.fullmatch(r"[A-Z][a-zA-Z'\-]*", w) or w in _MEMBER_NAME_STOP:
                break
            words.append(w)
        v = " ".join(words) if len(words) >= 2 else None
    if v:
        x["Member Name"] = v

    v = _first(r"Claim state:?\s*(Blocked|Audit|Paid|Denied|Pending|Processed|Approved)\b", text, re.I)
    if v:
        f["Claim State"] = v.capitalize()

    # Blocker: code right after "Blockers (n)", else at the start of the
    # next line (column layout).
    for i, line in enumerate(lines):
        m = re.search(r"Blockers \((\d+)\)\s*(.*)$", line)
        if not m:
            continue
        x["Blocker Count"] = m.group(1)
        code = re.match(r"([A-Z][A-Z_]{3,})\b", m.group(2))
        if not code and i + 1 < len(lines):
            code = re.match(r"([A-Z][A-Z_]{3,})\b", lines[i + 1])
        if code:
            f["Blocker"] = code.group(1)
        break
    v = _first(r"Denials \((\d+)\)", text)
    if v is not None:
        x["Denials Count"] = v

    f.update(_money_fields(text))
    v = _first(rf"Allowed\s+{MONEY}\s*\((\d+)%", text)
    if v:
        x["Allowed %"] = f"{v}%"

    for canon, label in (
        ("Service Start Date", r"Service start date"),
        ("Service End Date", r"Service end date"),
        ("Received date", r"Received date"),
    ):
        v = _first(rf"{label}\s+({DATE})", text, re.I)
        if v:
            f[canon] = v

    # Claim type -- closed vocabulary; labelled value first, then anywhere
    # on the page (the label is sometimes lost by OCR).
    ct = _first(r"Claim type\s+(UB - Facility|HCFA - Professional|NEWBORN)", text, re.I) or _first(
        r"\b(UB - Facility|HCFA - Professional)\b", text
    )
    if ct:
        f["Claim Type"] = {"ub - facility": "UB - Facility", "hcfa - professional": "HCFA-Professional"}.get(
            ct.lower(), ct
        )

    v = _first(r"Provider's claim #\s*([A-Z0-9]{8,}(?:-[A-Z0-9]{4,})?)\b", text)
    if not v:
        # Value displaced to the right of other labels on the same line
        # ("Provider's claim # Stay type Inpatient MBBAC29514982C").
        line = _first(r"(Provider's claim #.*)$", text, re.M) or ""
        for tok in re.findall(r"\b[A-Z0-9]{10,}(?:-[A-Z0-9]{4,})?\b", line):
            if re.search(r"[A-Z]", tok) and re.search(r"\d", tok):
                v = tok
                break
    if v and v != cid:
        f["Provider Claim#"] = v

    v = _first(r"Billing type\s+(\d{1,4})\b", text)
    if v:
        f["Billing Type"] = v
    v = _first(r"Stay type\s+(Inpatient|Outpatient)\b", text, re.I)
    if v:
        f["Stay type"] = v.capitalize()
    v = _first(r"Place of service\s+(\d{2})\b", text)
    if v:
        x["Place of Service"] = v
    v = _first(r"Diagnosis related group\s+\.?(\d{3})\b", text)
    if v:
        x["DRG"] = v
    v = _first(r"Transmission ID\s+(\d{8,})\b", text)
    if v:
        f["Transmission ID"] = v

    # Goal: OCR often truncates PROCESS_CLAIM to PROCESS_CLA.
    v = _first(r"\b(PROCESS_CL\w*)", text) or _first(r"\bGoal\s+([A-Z][A-Z_]{4,})\b", text)
    if v:
        f["Goal"] = "PROCESS_CLAIM" if v.startswith("PROCESS_CL") else v

    v = _first(r"Authorization required\?\s*(Yes(?: \([A-Z]+\))?|No)\b", text)
    if v:
        x["Authorization Required"] = v
    v = _first(r"Is Medicaid reclamation\?\s*(Yes|No)\b", text, re.I)
    if v:
        x["Medicaid Reclamation"] = v
    v = _first(r"International Provider\s+(Yes|No)\b", text, re.I)
    if v:
        x["International Provider"] = v.capitalize()

    # Business rule (as in field_extractor): Submission Type is always
    # Electronic on a claim screen.
    f["Submission Type"] = "Electronic"
    v = _first(r"Vendor source\s+([A-Z]{3,})\b", text)
    if v:
        x["Vendor Source"] = v
    v = _first(r"\bTenant\s+(oscar)\b", text, re.I)
    if v:
        f["Tenant"] = v.lower()

    v = _first(rf"Prompt process due date\s+({DATE})", text, re.I)
    if v:
        x["Prompt Process Due Date"] = v
    v = _first(r"Payment authority\s+([A-Z]{3,})\b", text)
    if v:
        x["Payment Authority"] = v
    v = _first(rf"Interest accrued\s+({MONEY})", text, re.I)
    if v:
        x["Interest Accrued"] = v
    v = _first(r"Excluded from interest\s+(Yes|No)\b", text)
    if v:
        x["Excluded From Interest"] = v

    # Patient tab: matched member's OSC ID.
    oscs = _osc_ids(text)
    if oscs:
        f["OSC ID"] = oscs[0]
    v = _first(r"Patient relationship\s+(Self|Child|Spouse|Dependent|Other)\b", text)
    if v:
        x["Patient Relationship"] = v

    # Ticket reference pasted into a Cortex note links claim <-> ticket.
    tids = _ticket_ids(text)
    if len(tids) == 1:
        f["Ticket ID"] = tids[0]
    elif len(tids) > 1:
        flags["Ticket ID"] = f"Multiple Ticket IDs on page ({', '.join(tids)}) -- left blank"

    v = _first(r"Workflow:\s*\n?\s*([A-Z][A-Z_]{3,})\b", text)
    if v:
        x["Note Workflow"] = v

    # Which tab/section is visible -- useful when reviewing gaps.
    sections = [s for s, rx in (
        ("About", r"About this claim"),
        ("Payment", r"Payment authority"),
        ("Patient", r"Matched Member|Submitted Info"),
        ("Providers", r"\bBilling provider\b|\bRendering provider\b"),
        ("Notes", r"Save note|Outcome/Changes Made|No notes for this claim"),
    ) if re.search(rx, text)]
    if sections:
        x["Visible Sections"] = ", ".join(sections)
    return f, x, flags


# ---------------------------------------------------------------------------
# 3. Appian ticket screen
# ---------------------------------------------------------------------------

def _appian_fields(text: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, str]]:
    f: Dict[str, Any] = {}
    x: Dict[str, Any] = {}
    flags: Dict[str, str] = {}

    tids = _ticket_ids(text)
    if len(tids) == 1:
        f["Ticket ID"] = tids[0]
    elif tids:
        flags["Ticket ID"] = f"Multiple Ticket IDs on page ({', '.join(tids)}) -- left blank"

    m = re.search(rf"\b([A-Z][A-Z_]+ hold on claim ({CLAIM_ID}))\b", text)
    if m:
        f["Ticket Name"] = m.group(1)
    cands = [m.group(2) if m else None, _first(rf"Ocelot ID\s+({CLAIM_ID})\b", text)]
    cids = [c for c in dict.fromkeys(cands) if _valid_claim_id(c)]
    if len(cids) == 1:
        f["Claim ID"] = cids[0]
    elif len(cids) > 1:
        flags["Claim ID"] = f"Ticket title and Ocelot ID disagree ({', '.join(cids)}) -- needs manual review"
        f["Claim ID"] = cids[1]  # Ocelot ID is the explicit claim field

    m = re.search(
        r"\b(P\d)\s+(In Progress|Resolved|Paused|Escalated|Withdrawn|Closed|New|Open|Pending)"
        r"(?:\s*\(([A-Za-z ]+)\)?)?",
        text,
    )
    if m:
        f["Ticket State"] = f"{m.group(1)} {m.group(2)}" + (f" ({m.group(3).strip()})" if m.group(3) else "")

    v = _first(rf"\bDue\s+({DATE})", text)
    if v:
        f["Due"] = v
    v = _clean_name(_first(rf"Assignee[ ]+({NAME})", text))
    if v:
        f["Assignee"] = v
    v = _first(r"Skill:?\s+([A-Z][A-Z_]{2,})\b", text)
    if v:
        f["Skill"] = v
    v = _first(r"\((\d+)\s*working days?\)", text, re.I)
    if v:
        f["SLA"] = f"{v} working days"
    v = _first(r"\bSLA\s+(.+?\(\d+\s*working days?\))", text, re.I)
    if v:
        x["SLA Name"] = v
    v = _first(r"Submitter\s+([A-Z][a-z]+)\b", text)
    if v:
        f["Submitter"] = v

    for canon, label in (
        ("Created On", "Created On"),
        ("Updated On", "Updated On"),
        ("Resolved On", "Resolved On"),
        ("Closed On", "Closed On"),
    ):
        v = _first(rf"{label}\s+({DATETIME})", text)
        if v:
            f[canon] = re.sub(r"\s+", " ", v)

    for canon, label, target in (
        ("Updated By", "Updated By", f),
        ("Resolved By", "Resolved By", f),
        ("Closed By", "Closed By", x),
    ):
        v = _clean_name(_first(rf"{label}[ ]+({NAME})", text))
        if v:
            target[canon] = v
    v = _first(r"Created By\s+([\w.]+)", text)
    if v:
        x["Created By"] = v
    # Business rule from field_extractor: Resolved By mirrors Updated By.
    if "Updated By" in f:
        f["Resolved By"] = f["Updated By"]

    v = _first(r"^Queue\s+(.+)$", text, re.M)
    if v:
        x["Queue"] = v
    v = _first(r"Member ID\s+([A-Z][A-Z0-9]{7})\b", text)
    if v:
        f["Member ID"] = v
    oscs = _osc_ids(text)
    if oscs:
        f["OSC ID"] = oscs[0]
    v = _first(r"^State\s+([A-Z]{2})\b", text, re.M)
    if v:
        f["Member State"] = v
    v = _first(r"^Code\s+([A-Z][A-Z_]{2,})\b", text, re.M) or _first(r"\bCode\s+([A-Z][A-Z_]{2,})\b", text)
    if v:
        f["Code"] = v
    v = _first(r"\bTenant\s+(Oscar)\b", text, re.I)
    if v:
        f["Tenant"] = v.lower()
    for label in ("Issue", "LOB", "System"):
        v = _first(rf"^{label}\s+(\w+)\b", text, re.M)
        if v:
            x[label] = v
    v = _first(r"Source\s+(Automat\b[^)]*\)?)", text)
    if v:
        x["Source"] = v

    v = _first(r"^Resolution\s+((?!Details)[A-Z][a-z]+(?: \([A-Za-z ]+\))?)", text, re.M)
    if v:
        f["Resolution"] = v
    v = _first(r"^Reason\s+([A-Z][a-z]+(?: [a-z]+)*)$", text, re.M)
    if v:
        f["Reason"] = v
    return f, x, flags


def _appian_queue_fields(text: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, str]]:
    x: Dict[str, Any] = {}
    v = _clean_name(_first(rf"Welcome,?[ ]+({NAME})", text))
    if v:
        x["Agent"] = v
    v = _first(r"Primary Queue:\s*(.+?)(?:\s*\(|$)", text, re.M)
    if v:
        x["Queue"] = v
    v = _first(r"(\d+H \d+M \d+S)", text)
    if v:
        x["Time Worked"] = v
    return {}, x, {}


# ---------------------------------------------------------------------------
# 4. Rosco (member profile / search), VERINT
# ---------------------------------------------------------------------------

def _rosco_fields(text: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, str]]:
    f: Dict[str, Any] = {}
    x: Dict[str, Any] = {}
    oscs = _osc_ids(text)
    if oscs:
        f["OSC ID"] = oscs[0]
    m = re.search(r"OSC\s?\d{8}\s?-?\s?\d{2}\s?-?\s?(R2\d[A-Z0-9]{5})\b", text)
    if m:
        f["Member ID"] = m.group(1)
    lines = text.split("\n")
    if len(lines) > 1:
        v = _first(r"^(.+?)\s+Member ?info\b", lines[1])
        if v:
            x["Member Name"] = v
    # Claims first: the "Viewing information for" plan banner shows on the
    # Claims tab too, so it only means Policy details when nothing else matches.
    view = next((name for name, rx in (
        ("Claims", r"Claims Export CSV|Medical claims"),
        ("Policy details", r"Viewing information for"),
        ("Member info", r"Basic info"),
    ) if re.search(rx, text)), None)
    if view:
        x["Rosco View"] = view
    v = _first(r"Viewing information for:\s*(.+?\(\d{1,2}/\d{1,2}/\d{4}\s*-?\s*\d{1,2}/\d{1,2}/\d{4}\))", text)
    if v:
        x["Plan"] = v
    if view == "Claims":
        ids = [c for c in re.findall(rf"^({CLAIM_ID})\s+\$", text, re.M) if _valid_claim_id(c)]
        if ids:
            # Other claims of this member -- informational only, NOT the
            # claim being worked (kept out of Claim ID so it never links).
            x["Member Claims Listed"] = ", ".join(dict.fromkeys(ids))
    return f, x, {}


def _rosco_search_fields(text: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, str]]:
    f: Dict[str, Any] = {}
    x: Dict[str, Any] = {}
    q = _first(r"^rosco\s*Q\s*(\S+)", text, re.I | re.M)
    if q:
        x["Search Query"] = q
        qo = _osc_ids(q)
        if qo:
            f["OSC ID"] = qo[0]
    v = _first(r"(\d+)\s+MEMBERS\b", text)
    if v:
        x["Members Found"] = v
    return f, x, {}


def _verint_fields(text: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, str]]:
    x: Dict[str, Any] = {}
    v = _first(r"\b(KM\d{6,})\b", text)
    if v:
        x["KM Article"] = v
    v = _first(r"Search\s+(?:\d+\s+)?([A-Za-z][A-Za-z ]+?)\s+\d{1,2}/\d{1,2}/\d{2}", text)
    if v:
        x["KM Title"] = v
    return {}, x, {}


_HANDLERS = {
    APP_CORTEX: _cortex_fields,
    APP_APPIAN: _appian_fields,
    APP_APPIAN_QUEUE: _appian_queue_fields,
    APP_ROSCO: _rosco_fields,
    APP_ROSCO_SEARCH: _rosco_search_fields,
    APP_VERINT: _verint_fields,
}


# Every label the regexes above read -- the learner treats these as known
# and only reports genuinely new "Label <value>" pairs.
KNOWN_FIELD_LABELS = [
    "Service start date", "Service end date", "Received date", "Prompt process due date",
    "Interest accrued", "Billed", "Allowed", "Pay provider", "Pay member", "Member responsibility",
    "Due", "Created On", "Updated On", "Resolved On", "Closed On", "Viewing information for",
]


def extract_screen(
    raw_text: str, document_index: int = 1, learner: Optional[PatternLearner] = None,
    source_name: str = "",
) -> Tuple[DocumentResult, str]:
    """Clean + classify + extract one plain-text screenshot dump. Returns
    the DocumentResult and the cleaned text (for review/export).

    With a PatternLearner, learned OCR/label corrections are applied before
    extraction, names are canonicalised, and new patterns are recorded."""
    text = clean_ocr_text(raw_text)
    fixes: List[str] = []
    if learner:
        text, fixes = learner.correct_text(text)
    app = classify_screen(text, learner)
    handler = _HANDLERS.get(app)
    if handler:
        fields, extra, flags = handler(text)
    else:
        fields, extra, flags = {}, {}, {}
        if app == APP_CORTEX_EMPTY:
            flags["Screen"] = "Cortex page captured before the claim loaded -- no data on screen"
    result = DocumentResult(document_index, app, fields, extra, flags)
    if learner:
        if app == APP_OTHER:
            learner.record_unknown_screen(text, source_name)
        learner.postprocess(result, text)
        if fixes:
            result.extra_fields["Auto-Corrections"] = "; ".join(dict.fromkeys(fixes))
    return result, text


def learn_batch(learner: PatternLearner, raw_texts: List[str], batch_name: str = "") -> None:
    """Pass 1: let the learner study the whole batch before extraction."""
    learner.learn_batch([clean_ocr_text(t) for t in raw_texts], classify_screen, KNOWN_FIELD_LABELS, batch_name)
