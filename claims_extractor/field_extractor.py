from __future__ import annotations

import difflib
import os
import re
import datetime as _dt

# Bump on every change -- lets a user verify which copy is actually running:
#   python -c "from claims_extractor.field_extractor import EXTRACTOR_VERSION; print(EXTRACTOR_VERSION)"
EXTRACTOR_VERSION = "2026-08-11.11"
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Optional, Tuple


_FALLBACK_SCHEMA_ROWS: List[Tuple[str, str]] = [
    ("Appian - Popup", "Sub status"),
    ("Appian - Popup", "Reason"),
    ("Appian- Home", "Ticket ID"),
    ("Appian- Home", "Ticket Name"),
    ("Appian- Home", "Ticket State"),
    ("Appian- Home", "Member State"),
    ("Appian- Home", "Claim ID"),
    ("Appian- Home", "Member ID"),
    ("Appian- Home", "Skill"),
    ("Appian- Home", "OSC ID"),
    ("Appian- Home", "Due"),
    ("Appian- Home", "SLA"),
    ("Appian- Home", "Created On"),
    ("Appian- Home", "Updated On"),
    ("Appian- Home", "Code"),
    ("Appian- Home", "Assignee"),
    ("Appian- Home", "Submitter"),
    ("Appian- Home", "Updated By"),
    ("Appian- Home", "Resolved By"),
    ("Appian- Home", "Resolved On"),
    ("Appian- Home", "Resolution"),
    ("Appian- Home", "Closed On"),
    ("Cortex - Patient - Matched member", "Claim ID"),
    ("Cortex - Patient - Matched member", "OSC ID"),
    ("Cortex - Patient - Submitted Info", "Claim ID"),
    ("Cortex - Patient - Submitted Info", "OSC ID"),
    ("Cortex - Patient - Submitted Info - Blocker", "Skill"),
    ("Cortex - Patient - Submitted Info - Blocker", "Orignally block status"),
    ("Cortex- About this claim", "Claim ID"),
    ("Cortex- About this claim", "Claim State"),
    ("Cortex- About this claim", "Blocker"),
    ("Cortex- About this claim", "Billed"),
    ("Cortex- About this claim", "Allowed"),
    ("Cortex- About this claim", "Pay provider"),
    ("Cortex- About this claim", "Pay member"),
    ("Cortex- About this claim", "Member responsibilty"),
    ("Cortex- About this claim", "Service Start Date"),
    ("Cortex- About this claim", "Service End Date"),
    ("Cortex- About this claim", "Provider Claim#"),
    ("Cortex- About this claim", "Stay type"),
    ("Cortex- About this claim", "Received date"),
    ("Cortex- About this claim", "Claim Type"),
    ("Cortex- About this claim", "Billing Type"),
    ("Cortex- About this claim", "Transmission ID"),
    ("Cortex- About this claim", "Goal"),
    ("Cortex- About this claim", "Submission Type"),
    ("Cortex- About this claim", "Tenant"),
    ("Member match Cortex", "Header"),
    ("Member match Cortex", "Member ID (OSC)"),
    # Rosco (document management screens)
    ("Rosco", "Document Title"),
    ("Rosco", "Title"),
    ("Rosco", "Creator"),
    ("Rosco", "Uploaded"),
    ("Rosco", "Filename"),
    # Elbow (person/member profile screens)
    ("Elbow", "First name"),
    ("Elbow", "Last name"),
    ("Elbow", "Date of birth"),
    ("Elbow", "Legal Sex"),
    ("Elbow", "PersonID"),
    ("Elbow", "OSC ID"),
    # Note: "Reason" also occurs standalone under Appian- Home (row 20 of
    # the CSV, index 22) -- already covered by the Appian- Home entry
    # above; duplicate (app, variable) pairs collapse naturally below.
]
# "Reason" belongs to *both* Appian - Popup and Appian- Home in the CSV.
_FALLBACK_SCHEMA_ROWS.append(("Appian- Home", "Reason"))


def _clean_variable_name(raw: str) -> str:
    """CSV variable names sometimes carry a trailing non-breaking space
    (\\xa0) or extra whitespace -- normalize to a clean display name."""
    return re.sub(r"\s+", " ", raw.replace("\xa0", " ")).strip()


def _fallback_schema() -> Dict[str, List[str]]:
    schema: Dict[str, List[str]] = {}
    for app, var in _FALLBACK_SCHEMA_ROWS:
        var = _clean_variable_name(var)
        schema.setdefault(app, [])
        if var not in schema[app]:
            schema[app].append(var)
    return schema


def load_schema() -> Dict[str, List[str]]:
    """Returns the hardcoded (Application -> [Variable, ...]) schema baked
    into this module. The CSV is used only as the reference/spec that
    this schema was transcribed from -- it is never read at runtime."""
    return _fallback_schema()


def _app_family(app: str) -> str:
    """Product-family key for an application name: 'Appian- Home' and
    'Appian - Popup' -> 'appian'; every 'Cortex...' screen -> 'cortex'."""
    return re.split(r"[\s\-]+", app.strip())[0].lower()


def _family_vocabulary(app: str, schema: Dict[str, List[str]]) -> List[str]:
    """The app's own schema labels plus those of its product-family
    siblings. Classification between sibling screens of one product (the
    several Cortex views, Appian home vs popup) is keyword-based and can
    pick the wrong sibling -- previously that shrank the searched label
    vocabulary to the wrong screen's few fields, and everything actually
    on the screen extracted as nothing. Searching the family-wide
    vocabulary is safe (same product, same label style) while still
    excluding unrelated apps' labels."""
    vocab: List[str] = list(schema.get(app, []))
    fam = _app_family(app)
    for other, vars_ in schema.items():
        if other == app or _app_family(other) != fam:
            continue
        for v in vars_:
            if v not in vocab:
                vocab.append(v)
    return vocab


# ===========================================================================
# 2. PPStructure [BOXES] parser
# ===========================================================================

DOC_SPLIT_RE = re.compile(r"(?=# OCR meta \| engine=)")
BOXES_BLOCK_RE = re.compile(r"\[BOXES\](.*?)\[/BOXES\]", re.S)
# bbox coordinates are usually ints, but engine/version changes can emit
# floats and/or spaces after the commas -- accept both so a cosmetic format
# change doesn't silently zero out the whole token stream (a document with
# zero parsed tokens extracts NOTHING, with no error anywhere).
_COORD = r"-?\d+(?:\.\d+)?"
LINE_RE = re.compile(
    r"^LINE\s+\d+\s*\|\s*bbox=\[\s*(?P<x0>" + _COORD + r")\s*,\s*(?P<y0>" + _COORD + r")\s*,\s*(?P<x1>" + _COORD + r")\s*,\s*(?P<y1>" + _COORD + r")\s*\]"
    r"\s*\|\s*conf=(?P<conf>[\d.\-]+)\s*\|\s*text=(?P<text>.*)$"
)
WORD_RE = re.compile(
    r"^WORD\s*\|\s*bbox=\[\s*(?P<x0>" + _COORD + r")\s*,\s*(?P<y0>" + _COORD + r")\s*,\s*(?P<x1>" + _COORD + r")\s*,\s*(?P<y1>" + _COORD + r")\s*\]"
    r"\s*\|\s*text=(?P<text>.*)$"
)


def _as_px(s: str) -> int:
    """Parse a bbox coordinate that may be an int or a float string."""
    return int(float(s))


@dataclass
class Token:
    text: str
    x0: int
    y0: int
    x1: int
    y1: int
    is_line_level: bool  # True if this came from a LINE with no WORD children


@dataclass
class Document:
    index: int  # 1-based order of appearance in the input
    tokens: List[Token] = dc_field(default_factory=list)
    raw_text: str = ""
    # True when no [BOXES] token data could be parsed and the tokens were
    # rebuilt from plain text lines instead (approximate positions only).
    parser_fallback: bool = False


def split_documents(raw_text: str) -> List[str]:
    """Split the full dump into one chunk per '# OCR meta | ...' document."""
    parts = DOC_SPLIT_RE.split(raw_text)
    return [p for p in parts if p.strip()]


def parse_boxes(doc_text: str) -> List[Token]:
    """Extract an ordered token stream from every [BOXES] block in a
    document (there can be more than one, e.g. multiple detected tables
    on one page). WORD-level entries are preferred (finer-grained,
    already column-segmented); a LINE with no WORD children is kept as
    a single token."""
    tokens: List[Token] = []
    blocks = BOXES_BLOCK_RE.findall(doc_text)
    if not blocks:
        # Missing/unclosed [BOXES] markers -- scan the whole document text
        # for LINE/WORD entries directly instead of silently yielding zero
        # tokens (which made the entire document extract as blank).
        blocks = [doc_text]
    for block in blocks:
        current_line_words: List[Token] = []
        pending_line: Optional[re.Match] = None

        def _flush():
            if pending_line is None:
                return
            if current_line_words:
                tokens.extend(current_line_words)
                # A LINE's WORD children don't always cover its full text
                # (seen in real dumps: LINE "Claim state: Blocked" with a
                # single WORD "Claim state:"). The uncovered remainder
                # ("Blocked") is real screen text that was previously LOST
                # entirely -- the label's true value went missing and the
                # geometric pairing grabbed unrelated neighboring text
                # instead. Recover every uncovered segment as its own
                # token, with x-coordinates inferred from the gaps between
                # the covered words.
                m = pending_line
                line_text = m.group("text").strip()
                lx0, lx1 = _as_px(m.group("x0")), _as_px(m.group("x1"))
                ly0, ly1 = _as_px(m.group("y0")), _as_px(m.group("y1"))
                pos = 0
                prev_x1 = lx0
                for w in current_line_words:
                    idx = line_text.find(w.text, pos) if w.text else -1
                    if idx < 0:
                        continue
                    gap = line_text[pos:idx].strip()
                    if gap:
                        tokens.append(
                            Token(gap, prev_x1, ly0, max(w.x0, prev_x1 + 1), ly1, False)
                        )
                    pos = idx + len(w.text)
                    prev_x1 = w.x1
                trailing = line_text[pos:].strip()
                if trailing:
                    tokens.append(
                        Token(trailing, prev_x1, ly0, max(lx1, prev_x1 + 1), ly1, False)
                    )
            else:
                m = pending_line
                tokens.append(
                    Token(
                        text=m.group("text").strip(),
                        x0=_as_px(m.group("x0")),
                        y0=_as_px(m.group("y0")),
                        x1=_as_px(m.group("x1")),
                        y1=_as_px(m.group("y1")),
                        is_line_level=True,
                    )
                )

        for raw_line in block.split("\n"):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            lm = LINE_RE.match(raw_line)
            if lm:
                _flush()
                pending_line = lm
                current_line_words = []
                continue
            wm = WORD_RE.match(raw_line)
            if wm and pending_line is not None:
                current_line_words.append(
                    Token(
                        text=wm.group("text").strip(),
                        x0=_as_px(wm.group("x0")),
                        y0=_as_px(wm.group("y0")),
                        x1=_as_px(wm.group("x1")),
                        y1=_as_px(wm.group("y1")),
                        is_line_level=False,
                    )
                )
        _flush()
    return tokens


def _fallback_line_tokens(doc_text: str) -> List[Token]:
    """Last-resort parse for a document with no machine-readable LINE/WORD
    token data at all: treat every plain nonempty text line as one
    LINE-level token with synthetic stacked bboxes, so the geometric
    label->value pairing still behaves sensibly ('value on the next line'
    becomes 'value below label'). Meta/marker lines are skipped."""
    tokens: List[Token] = []
    y = 0
    for line in doc_text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("["):
            continue
        tokens.append(
            Token(text=line, x0=0, y0=y, x1=max(10, 8 * len(line)), y1=y + 18, is_line_level=True)
        )
        y += 24
    return tokens


def parse_documents(raw_text: str) -> List[Document]:
    docs = []
    for i, chunk in enumerate(split_documents(raw_text), start=1):
        tokens = parse_boxes(chunk)
        fallback = False
        if not tokens:
            # Nothing machine-readable at all -- previously this document
            # would silently extract ZERO fields with no indication why.
            # Fall back to plain text lines and flag it downstream.
            tokens = _fallback_line_tokens(chunk)
            fallback = bool(tokens)
        docs.append(Document(index=i, tokens=tokens, raw_text=chunk, parser_fallback=fallback))
    return docs


# ===========================================================================
# 3. Fuzzy label matching
# ===========================================================================


def _normalize(s: str) -> str:
    s = s.replace("\xa0", " ").lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _match_score(token_norm: str, alias_norm: str) -> float:
    return difflib.SequenceMatcher(None, token_norm, alias_norm).ratio()


def _threshold_for(alias_norm: str) -> float:
    # A single cutoff, validated against real garbled/clean label pairs
    # pulled from both files: true matches scored 0.72-1.00, unrelated
    # words scored 0.00-0.55 (see module docstring / test suite at the
    # bottom of this file). The UI-chrome stopword list and the
    # score-ranked conflict resolution in _extract_fields_for_document
    # handle the residual collision risk from using one flat cutoff.
    return 0.72


# UI chrome that recurs across many screens and happens to fuzzy-resemble
# a real schema label closely enough to cause false positives (e.g. the
# "Ticket Info" breadcrumb vs the "Ticket ID" field, or the "Providers"
# nav tab vs the "Pay provider" field). These are recognized defensively
# and excluded from ever being treated as a label match, regardless of
# fuzzy score.
_UI_CHROME_STOPWORDS = [
    "ticket info", "about this claim", "payment", "patient", "providers",
    "medical claim lines", "readjudicate", "actions", "dashboard",
    "batch history", "testing", "settlements", "search", "create",
    "claims home", "inventory", "create ticket", "unarchive",
    "latest version", "member match cortex", "research by member id",
    "research by name",
]
_UI_CHROME_NORMS = [_normalize(w) for w in _UI_CHROME_STOPWORDS]


def _ui_chrome_score(token_norm: str) -> float:
    """Best fuzzy score of this token against the UI-chrome stopword list."""
    best = 0.0
    for stop in _UI_CHROME_NORMS:
        r = difflib.SequenceMatcher(None, token_norm, stop).ratio()
        if r > best:
            best = r
    return best


def _is_ui_chrome(token_norm: str) -> bool:
    return _ui_chrome_score(token_norm) >= 0.85


def _contains_id_shape(text: str) -> bool:
    """True if this token already contains a validated Claim ID
    (e.g. a page title like 'Claim #7ATXE61J'). Such tokens are values/
    titles, not plain labels, and must never be fuzzy-matched as one --
    otherwise the *next* token gets wrongly grabbed as that label's
    value. Deliberately checks ONLY the digit-first Claim ID shape:
    the letter-first Member ID shape matches countless ordinary 8-letter
    English words (e.g. "Assignee", "Resolved" are each exactly 8
    letters) and would wrongly block real labels."""
    return bool(_find_compact_id(text, CLAIM_ID_RE))


@dataclass
class LabelHit:
    canonical: str
    score: float
    token_index: int
    # Number of consecutive tokens this label spans (usually 1; >1 when a
    # multi-word label was OCR-split across adjacent same-row tokens, e.g.
    # "Service" + "Start Date" re-joined into one "Service Start Date" hit).
    span: int = 1


# ---------------------------------------------------------------------
# Geometry helpers -- every Token carries its OCR bounding box, which is
# far more reliable for pairing a label with its value than raw stream
# order (values regularly sit BELOW their label, or several label/value
# pairs share one visual row).
# ---------------------------------------------------------------------


def _v_overlap_frac(ay0: int, ay1: int, by0: int, by1: int) -> float:
    """Vertical overlap of two boxes as a fraction of the shorter height."""
    ov = min(ay1, by1) - max(ay0, by0)
    ref = max(1, min(ay1 - ay0, by1 - by0))
    return ov / ref


def _h_overlap_frac(ax0: int, ax1: int, bx0: int, bx1: int) -> float:
    """Horizontal overlap of two boxes as a fraction of the narrower width."""
    ov = min(ax1, bx1) - max(ax0, bx0)
    ref = max(1, min(ax1 - ax0, bx1 - bx0))
    return ov / ref


def _same_visual_row(a: Token, b: Token) -> bool:
    """True when b sits directly to the right of a on the same visual row,
    close enough that the two could be fragments of one label/phrase."""
    if _v_overlap_frac(a.y0, a.y1, b.y0, b.y1) < 0.5:
        return False
    gap = b.x0 - a.x1
    return -20 <= gap <= 3 * max(a.y1 - a.y0, b.y1 - b.y0, 1)


# Known status/state VALUE words that must never be treated as a LABEL
# candidate, even though some of them coincidentally fuzzy-match a real
# label's text (e.g. "Blocked" scores 0.86 against "Blocker" -- same root
# word). Left unguarded, the conflict-resolution pass in
# _extract_fields_for_document sees that high-scoring "label" hit on the
# very next token and concludes the preceding real label (here "Claim
# State") must be a bare header with no value -- silently dropping the
# actual value ("Claim state: Blocked" never gets captured).
_KNOWN_STATE_VALUE_NORMS = {
    _normalize(w) for w in [
        "Blocked", "Audit", "Resolved", "Suppressed", "Withdrawn",
        "In Progress", "Closed", "Open", "Escalated", "Pending",
    ]
}


def _best_alias(
    token_norm: str, alias_norms: List[Tuple[str, str]]
) -> Tuple[Optional[str], float]:
    """Best above-threshold canonical label for a normalized token text."""
    best_canonical, best_score = None, 0.0
    for canonical, anorm in alias_norms:
        score = _match_score(token_norm, anorm)
        if score >= _threshold_for(anorm) and score > best_score:
            best_canonical, best_score = canonical, score
    return best_canonical, best_score


def find_label_hits(
    tokens: List[Token], variable_names: List[str]
) -> List[LabelHit]:
    """For every token, find the best-matching canonical variable name (if
    any) above threshold. Multi-word labels that OCR split across adjacent
    same-row tokens (e.g. 'Service' + 'Start Date') are also recognized by
    scoring the joined text of 2- and 3-token windows; a joined hit is only
    kept when it beats every single-token hit it overlaps. Returns hits in
    token order."""
    alias_norms = [(v, _normalize(v)) for v in variable_names]
    hits: List[LabelHit] = []
    for i, tok in enumerate(tokens):
        tnorm = _normalize(tok.text)
        if not tnorm:
            continue
        if tnorm in _KNOWN_STATE_VALUE_NORMS:
            continue
        if _contains_id_shape(tok.text):
            continue
        best_canonical, best_score = _best_alias(tnorm, alias_norms)
        # UI chrome only blocks a token when the chrome resemblance is at
        # least as strong as the best real-label match. (Previously this
        # was an unconditional block, which let the "create" stopword
        # swallow the real "Created" label token at 0.92 similarity --
        # starving "Created On" of any chance to match.)
        chrome = _ui_chrome_score(tnorm)
        if chrome >= 0.85 and chrome >= best_score:
            continue
        if best_canonical:
            hits.append(LabelHit(best_canonical, best_score, i))

    # Second pass: re-join multi-word labels that OCR split across several
    # adjacent tokens on the same visual row. Without this, "Service" +
    # "Start Date" never matches "Service Start Date" (each fragment alone
    # scores below threshold) and the field silently extracts nothing.
    for size in (2, 3):
        for i in range(len(tokens) - size + 1):
            group = tokens[i : i + size]
            if any(not g.text.strip() for g in group):
                continue
            if not all(_same_visual_row(group[k], group[k + 1]) for k in range(size - 1)):
                continue
            joined = " ".join(g.text for g in group)
            jnorm = _normalize(joined)
            if not jnorm or jnorm in _KNOWN_STATE_VALUE_NORMS:
                continue
            if _contains_id_shape(joined):
                continue
            best_canonical, best_score = _best_alias(jnorm, alias_norms)
            if not best_canonical:
                continue
            chrome = _ui_chrome_score(jnorm)
            if chrome >= 0.85 and chrome >= best_score:
                continue
            overlapping = [
                h.score for h in hits
                if h.token_index < i + size and h.token_index + h.span > i
            ]
            if best_score > max(overlapping, default=0.0):
                hits.append(LabelHit(best_canonical, best_score, i, size))

    hits.sort(key=lambda h: h.token_index)
    return hits


# Splitting a token apart is destructive, so it demands a higher confidence
# than the ordinary 0.72 label threshold -- a borderline fuzzy resemblance
# inside a value phrase must never shred that value into fake label/value
# fragments.
_SPLIT_MIN_SCORE = 0.80


def _split_multifield_tokens(
    tokens: List[Token], variable_names: List[str]
) -> List[Token]:
    """One OCR token -- usually a LINE with no WORD children -- can carry
    several fields glued side by side ("Billed $154.09 Allowed $131.00") or
    a label glued onto its own value ("Assignee: Jessica Smith"). Neither
    shape is matchable as-is: the whole token fuzzy-scores below threshold
    against every label, so ALL the fields inside it silently extract
    nothing. This pass scans each multi-word token for embedded
    high-confidence label phrases and splits it into alternating label /
    value tokens, interpolating bbox x-coordinates by character position so
    the geometric pairing downstream still works."""
    alias_norms = [(v, _normalize(v)) for v in variable_names]
    if not alias_norms:
        return list(tokens)
    max_label_words = max(len(a.split()) for _, a in alias_norms)
    out: List[Token] = []
    for tok in tokens:
        words = tok.text.split()
        if len(words) < 2:
            out.append(tok)
            continue
        # Scan left-to-right for embedded label phrases, longest-first.
        occurrences: List[Tuple[int, int]] = []  # (start_word, word_count)
        i = 0
        while i < len(words):
            matched = None
            for size in range(min(max_label_words, len(words) - i), 0, -1):
                seg = " ".join(words[i : i + size])
                segn = _normalize(seg)
                if not segn or segn in _KNOWN_STATE_VALUE_NORMS:
                    continue
                if _contains_id_shape(seg):
                    continue
                canonical, score = _best_alias(segn, alias_norms)
                if canonical and score >= _SPLIT_MIN_SCORE and _ui_chrome_score(segn) < score:
                    matched = (i, size)
                    break
            if matched:
                occurrences.append(matched)
                i += matched[1]
            else:
                i += 1
        label_word_count = sum(sz for _, sz in occurrences)
        # Nothing to split: no embedded label found, or the token IS a
        # label (all words consumed by label matches, no value glued on).
        if not occurrences or label_word_count == len(words):
            out.append(tok)
            continue
        # Build alternating segments: [value-before] label value label value...
        bounds: List[Tuple[int, int]] = []
        pos = 0
        for start, size in occurrences:
            if start > pos:
                bounds.append((pos, start))
            bounds.append((start, start + size))
            pos = start + size
        if pos < len(words):
            bounds.append((pos, len(words)))
        total_chars = max(len(" ".join(words)), 1)
        width = max(tok.x1 - tok.x0, 1)
        for a, b in bounds:
            seg_text = " ".join(words[a:b])
            start_chars = sum(len(w) + 1 for w in words[:a])
            end_chars = start_chars + len(seg_text)
            x0 = tok.x0 + int(width * start_chars / total_chars)
            x1 = tok.x0 + int(width * end_chars / total_chars)
            out.append(
                Token(
                    text=seg_text,
                    x0=x0,
                    y0=tok.y0,
                    x1=max(x1, x0 + 1),
                    y1=tok.y1,
                    is_line_level=tok.is_line_level,
                )
            )
    return out


# ===========================================================================
# 4. ID-shape regex helpers (Ticket ID / Claim ID / Member-OSC ID)
#    Same shape rules as field_extractor.py: Claim ID = digit-first 8
#    alnum chars, Member ID = letter-first 8 alnum chars.
# ===========================================================================

CLAIM_ID_RE = re.compile(r"^\d[A-Z0-9]{7}$")
MEMBER_ID_RE = re.compile(r"^[A-Z][A-Z0-9]{7}$")
# Same 8-char alnum "shape" as above but accepting EITHER a digit-first or
# letter-first leading character. Only ever used when the match is already
# anchored by an explicit keyword (e.g. the literal word "claim" or "member"
# immediately preceding it) -- in that context the keyword itself is the
# label, so the leading-character restriction that disambiguates Claim ID
# from Member ID in unlabeled/ambiguous text is unnecessary and was causing
# real claim IDs that happen to start with a letter (e.g. "Q20K5GSF") to be
# silently dropped.
ANY_LEADING_ID_RE = re.compile(r"^[A-Z0-9][A-Z0-9]{7}$")

# Strict Claim ID shape used by the "claim"-anchored path in
# find_all_claim_ids below, enforced as EXACTLY ONE of two forms -- no
# other length or character mix is ever accepted:
#   (a) CLAIM_ID_ALL_LETTERS_RE -- 8 CAPITAL LETTERS, no digits at all
#       (e.g. "QATXEBIJ").
#   (b) CLAIM_ID_ALNUM_MIX_RE   -- an 8-character CAPITAL alphanumeric
#       mix of letters + digits (e.g. "7ATXE61J", "U54657IJ").
# A bare run of 8 digits, or anything not exactly 8 characters, fails
# both and is never accepted as a Claim ID.
CLAIM_ID_ALL_LETTERS_RE = re.compile(r"^[A-Z]{8}$")
CLAIM_ID_ALNUM_MIX_RE = re.compile(r"^[A-Z0-9]{8}$")

# Ordinary workflow/business words that can coincidentally be all-capital
# and even 8 letters long (screens are often rendered in ALL CAPS), but
# are never a real Claim ID -- e.g. "System Interest Accrued" was being
# misread as a Claim ID via the all-letters fallback because "INTEREST"
# is exactly 8 capital letters. Checked BEFORE the strict 8-letter shape
# is ever accepted, regardless of where the word sits in the window.
_CLAIM_ID_NEGATED_WORDS = {
    "ESCALATE", "ESCALATED", "WITHDREW", "WITHDRAWN", "ACTION",
    "SYSTEM", "INTEREST", "ACCRUED",
    "BLOCKED", "RESOLVED", "SUPPRESSED", "CLOSED", "PENDING",
    "MEMBER", "REPORTED",
}


def _contains_negated_word(candidate: str) -> bool:
    """True if an 8-char ID candidate contains any negated
    workflow/business word as a SUBSTRING, not just as an exact match.
    Some negated words (e.g. "MEMBER", "SYSTEM", "ACTION") are shorter
    than 8 characters, so a candidate can still be assembled out of a
    negated word plus stray digits/letters glued on by OCR -- e.g.
    "88MEMBER" or "MEMBER12" are 8-char alphanumeric strings that would
    otherwise pass CLAIM_ID_ALNUM_MIX_RE untouched. A plain equality
    check misses these; substring containment catches them."""
    upper = candidate.upper()
    return any(neg in upper for neg in _CLAIM_ID_NEGATED_WORDS)
# OCR-confusion tolerant anchors: 'O' is routinely misread as '0' in these
# keyword prefixes ("H0LD-5886366", "0SC7715805201"). Since Ticket ID / OSC
# ID are ONLY ever trusted from these strict scans (the fuzzy pass is
# deliberately overridden), a missed anchor previously meant the field came
# out completely blank for that screenshot.
# The 7 digits end at (?!\d), NOT \b: OCR glues the next word straight
# onto the number ("HOLD-5892385Production Hold") and digit->letter is not
# a \b boundary, so \b would silently drop these. (?!\d) still rejects an
# 8th digit, keeping the exactly-seven-digits rule.
TICKET_ID_RE = re.compile(r"\bH[O0]LD[-\s]?(\d{7})(?!\d)", re.IGNORECASE)
# Optional "-NN" member/claim suffix rides on many OSC IDs in the dumps
# ("OSC78239442-02") and belongs in the final value with the dash dropped
# ("OSC7823944202") -- exactly how the reference workbook renders them.
OSC_ID_RE = re.compile(r"\b[O0]SC\s?T?(\d{6,12})(?:-(\d{1,3}))?\b", re.IGNORECASE)  # 'T' OCR-noise before digits


# ---------------------------------------------------------------------------
# Whitespace tokenization of the raw dump text.
#
# The strict ID / status scans below run on a flat whitespace token stream
# (['#', 'OCR', 'meta', '|', ..., 'HOLD-5892082', ...]) built from the WHOLE
# raw document text -- not just the parsed [BOXES] tokens -- so an ID that
# only appears inside a [TABLE] CELL[...] line (or any other section the
# bbox parser doesn't cover) is still found. OCR also routinely splits an
# ID across two adjacent tokens ("HOLD-" + "5892082", "OSC" + "76031813-03");
# the pairwise joins below stitch those back together.
# ---------------------------------------------------------------------------

def tokenize_text(raw_text: str) -> List[str]:
    """Flat whitespace token stream of the raw dump, e.g.
    ['#', 'OCR', 'meta', '|', 'engine=paddleocr+ppstructure', ...]."""
    return (raw_text or "").split()


# Token-level shapes. Ticket ID: HOLD glued to EXACTLY seven digits; the
# digits may have a trailing WORD glued on ("HOLD-5892385Production") but
# never an eighth digit. Or a bare "HOLD"/"HOLD-" token immediately
# followed by a seven-digit token.
_TICKET_TOKEN_RE = re.compile(r"^H[O0]LD[-–]?(\d{7})(?!\d)", re.IGNORECASE)
_TICKET_PREFIX_TOKEN_RE = re.compile(r"^H[O0]LD[-–]?$", re.IGNORECASE)
_SEVEN_DIGITS_RE = re.compile(r"^(\d{7})(?!\d)")

_OSC_TOKEN_RE = re.compile(r"^[O0]SCT?(\d{6,12})(?:-(\d{1,3}))?[.,;:]?$", re.IGNORECASE)
_OSC_PREFIX_TOKEN_RE = re.compile(r"^[O0]SCT?$", re.IGNORECASE)
_OSC_DIGITS_TOKEN_RE = re.compile(r"^(\d{6,12})(?:-(\d{1,3}))?[.,;:]?$")
CURRENCY_RE = re.compile(r"\$[\d,]+\.\d{2}")

# Literal "MemberID <id>" / "Member ID <id>" field as it appears on Appian
# screens -- distinct from the Cortex/Member-match "Member ID (OSC)" (OSC-
# prefixed) field. Captures the 8-char alnum ID immediately following the
# label, tolerating the OCR gluing the label and value together with no
# space (e.g. "MemberID R21627LL").
MEMBER_ID_LITERAL_RE = re.compile(r"\bMember\s?ID\s+([A-Z][A-Z0-9]{7})\b", re.IGNORECASE)

# Two-letter USPS state codes -- used to validate the literal "State: FL"
# field on Appian screens (the member's US state) so it's never confused
# with the unrelated ticket-status pill (which is also loosely labeled
# "State" in places but has a completely different shape/value).
_US_STATE_CODES = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN",
    "IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV",
    "NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN",
    "TX","UT","VT","VA","WA","WV","WI","WY","DC",
}

# The ticket "State" (e.g. "P3 Resolved (Suppressed)") is rendered as a
# colored status pill with no literal "State:" label anywhere in the OCR
# text -- it can only be recognized by its own shape (priority code
# "P<digit>" followed by a status phrase), never by fuzzy-matching a
# label word. Matching the bare word "State" against arbitrary nearby
# tokens is unreliable (too short, too generic) so it's excluded from
# the generic fuzzy pass entirely and handled here instead.
# (?!\d) not \b after the digits: OCR glues the status word straight onto
# the priority code ("P3InProgress (In Review)") and digit->letter is not
# a \b boundary. Validation still happens downstream in
# _normalize_ticket_state, so a stray P2P-style token can't slip through.
STATUS_PILL_RE = re.compile(r"^P\s*\d+(?!\d)", re.IGNORECASE)
_LABELS_WITHOUT_LITERAL_TEXT = {"Ticket State"}


def _alnum_tokens(s: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", s or "")


_DURATION_TOKEN_RE = re.compile(r"^\d+[HMS]$", re.IGNORECASE)


def _find_compact_id(
    text: str, validator: re.Pattern, min_letters: int = 2, min_digits: int = 0, max_size: int = 4
) -> Optional[str]:
    toks = _alnum_tokens(text)
    n = len(toks)
    for size in range(1, max_size + 1):
        for i in range(0, n - size + 1):
            window = toks[i : i + size]
            if sum(len(t) for t in window) != 8:
                continue
            if size > 1 and all(_DURATION_TOKEN_RE.match(t) for t in window):
                # A clock/duration readout like "3H 45M 58S" -- never a
                # real ID, regardless of its coincidental 8-char length.
                continue
            raw_candidate = "".join(window)
            # Real IDs are always rendered in capitals in the OCR text
            # itself -- require the SOURCE text to already be uppercase
            # (digits have no case and don't count against this) rather
            # than force-uppercasing before validating. Without this, an
            # ordinary lowercase/Title-Case word or phrase (e.g. "System"
            # + a short neighboring token, "Interest", "requires") that
            # merely sums to 8 characters would get coerced into passing
            # the uppercase-only shape regex, even though it was never
            # actually capitalized in the source.
            if not raw_candidate.isupper():
                continue
            candidate = raw_candidate.upper()
            if not validator.match(candidate):
                continue
            if sum(c.isalpha() for c in candidate) < min_letters:
                continue
            if sum(c.isdigit() for c in candidate) < min_digits:
                continue
            return candidate
    return None


def find_all_ticket_ids(full_text: str) -> List[str]:
    """Every distinct Ticket ID mentioned on the page, in first-seen
    order. A single-ticket detail screen will only ever have one; a
    queue/list screen (e.g. "My Open Tickets") can legitimately show
    several -- callers use this to detect that ambiguity rather than
    silently picking whichever one happens to be textually first."""
    seen: List[str] = []

    def _add(digits: str) -> None:
        tid = f"HOLD-{digits}"
        if tid not in seen:
            seen.append(tid)

    # Pass 1 -- regex over the running text (catches "HOLD-5892082" and
    # "HOLD 5892082" inside longer lines).
    for m in TICKET_ID_RE.finditer(full_text):
        _add(m.group(1))

    # Pass 2 -- whitespace token stream. Anchors on a HOLD token: either
    # the seven digits are glued onto it, or they are the very next token.
    # Exactly seven digits, per business rule -- a 6- or 8-digit number
    # next to a stray "HOLD" is never accepted.
    toks = tokenize_text(full_text)
    for i, tok in enumerate(toks):
        m = _TICKET_TOKEN_RE.match(tok)
        if m:
            _add(m.group(1))
            continue
        if _TICKET_PREFIX_TOKEN_RE.match(tok) and i + 1 < len(toks):
            nxt = toks[i + 1].strip(".,;:")
            nm = _SEVEN_DIGITS_RE.match(nxt)
            if nm:
                _add(nm.group(1))
    return seen


def find_ticket_id(full_text: str) -> Optional[str]:
    ids = find_all_ticket_ids(full_text)
    return ids[0] if ids else None


def find_osc_id(full_text: str) -> Optional[str]:
    m = OSC_ID_RE.search(full_text)
    if m:
        # Strict requirement: the raw matched text must itself begin with an
        # OSC prefix (allowing the O->0 OCR confusion) -- not just resemble
        # one after reconstruction. This is a defensive re-check on top of
        # the regex (which already anchors on \b[O0]SC...) so a future edit
        # to OSC_ID_RE can't silently start accepting IDs without a real
        # prefix. The value is always canonicalized to a literal "OSC"
        # prefix, and any "-NN" suffix is appended with the dash dropped
        # ("OSC78239442-02" -> "OSC7823944202").
        if re.match(r"^[O0]SC", m.group(0).strip(), re.IGNORECASE):
            return f"OSC{m.group(1)}{m.group(2) or ''}"

    # Token fallback -- the OSC prefix and its digits split across two
    # whitespace tokens ("OSC" + "76031813-03"), which the running-text
    # regex misses when other characters sit between them in the raw dump.
    toks = tokenize_text(full_text)
    for i, tok in enumerate(toks):
        tm = _OSC_TOKEN_RE.match(tok)
        if tm:
            return f"OSC{tm.group(1)}{tm.group(2) or ''}"
        if _OSC_PREFIX_TOKEN_RE.match(tok) and i + 1 < len(toks):
            dm = _OSC_DIGITS_TOKEN_RE.match(toks[i + 1])
            if dm:
                return f"OSC{dm.group(1)}{dm.group(2) or ''}"
    return None


# Excludes the plural "claims" (nav chrome like "Claims- LM appian") via a
# negative lookahead, but still matches OCR gluing the ID directly onto
# the singular word with no space at all (e.g. "claim7ATXE61J").
_CLAIM_WORD_RE = re.compile(r"claim(?!s)", re.IGNORECASE)

# Separator noise that can sit between the "claim" keyword and its ID --
# a literal '#' (e.g. "claim#U54657IJ"), a colon, a hyphen, or plain
# whitespace, in any combination (e.g. "claim # U54657IJ", "claim: U54657IJ").
# Consumed immediately after the "claim" match so both the keyword AND
# any '#'/':'/'-' glued onto it are negated before we ever look at what
# follows -- the anchor always lands on the real value, never on stray
# punctuation.
_CLAIM_ANCHOR_SEP_RE = re.compile(r"^[\s#:\-]*")


# Claim cards on the Cortex member/claims-list screen render as
# "<ClaimID>-<TENANT>" (e.g. "EFN1MBBZ-OCELOT") with no "claim" keyword
# anywhere near them -- the hyphen-glued ALL-CAPS tenant suffix itself is
# the anchor. Only the glued form is trusted: the expanded/selected card
# is rendered glued, while a space-separated pairing ("KDMVOWYO OCELOT")
# also shows up on partially-visible collapsed cards at the screen edge
# and would create false multi-claim ambiguity.
_TENANT_CARD_RE = re.compile(r"\b([A-Z0-9]{8})-([A-Z]{4,12})\b")


def find_all_claim_ids(full_text: str) -> List[str]:
    """Every distinct Claim ID mentioned on the page, in first-seen order.
    Same multi-value-ambiguity purpose as find_all_ticket_ids above."""
    found_ids: List[str] = []
    for m in _TENANT_CARD_RE.finditer(full_text):
        cand, tenant = m.group(1), m.group(2)
        if _contains_negated_word(cand) or tenant in _CLAIM_ID_NEGATED_WORDS:
            continue
        if not any(c.isalpha() for c in cand):
            continue  # a pure digit run is never a valid Claim ID
        if cand not in found_ids:
            found_ids.append(cand)
    for m in _CLAIM_WORD_RE.finditer(full_text):
        idx = m.end()
        # Negate "claim" itself plus any '#'/':'/'-'/whitespace directly
        # glued onto it (e.g. "claim#U54657IJ") before doing anything else.
        sep_m = _CLAIM_ANCHOR_SEP_RE.match(full_text[idx : idx + 5])
        if sep_m:
            idx += sep_m.end()
        # "Claim ID" as a compound column header (queue/list screens, e.g.
        # "ParentID TicketID Claim ID Skill Status") has no value directly
        # attached to it in the token stream -- anchoring here and reading
        # forward would grab unrelated text. Skip and keep looking.
        # "Claim ID" (compound column header) and "Claim Info" (section
        # heading, same family as "Member Info"/"Additional Info" seen
        # elsewhere on these screens) both have no value directly
        # attached -- anchoring here would grab unrelated following text
        # (e.g. "ClaimInfo" glued onto a footer timestamp). Skip both.
        # "claim." ending a sentence ("...no suppressed holds on this
        # claim.") is plain English, not a label anchoring an ID -- reading
        # forward from it grabs whatever unrelated field/sentence happens
        # to follow. Skip it.
        if full_text[idx : idx + 1] == ".":
            continue
        following = full_text[idx : idx + 5].strip().upper()
        if following.startswith("ID") or following.startswith("INFO"):
            continue
        window = full_text[idx : idx + 40]
        # The word "claim" (plus any '#'/':' separator) is the label here
        # (e.g. "hold on claim Q20K5GSF" / "claim#U54657IJ" / OCR-glued
        # "claim7ATXE61J"). Try the alphanumeric-mix shape first (still
        # the common case -- min_digits=1 guards against a coincidental
        # 8-letter dictionary word like "requires" being mistaken for an
        # ID, since real mixed IDs always combine letters and digits).
        # Step 1 -- alphanumeric-mix shape (still the common case).
        found = _find_compact_id(window, CLAIM_ID_ALNUM_MIX_RE, min_digits=1, max_size=1)

        # Step 2 -- negation check: never accept a known workflow/status
        # word as an ID, whatever shape matched it. Uses substring
        # containment (not exact equality) so a candidate that's a
        # negated word plus glued-on digits/letters (e.g. "88MEMBER",
        # "MEMBER12") is rejected too, not just the bare word itself.
        if found and _contains_negated_word(found):
            found = None

        # Step 3 -- strict all-capital-letters shape (exactly 8 letters,
        # zero digits), only for the rarer all-letter Claim ID. Only the
        # SINGLE immediate next token is considered (not a scan across
        # the whole window) -- a real ID always sits directly next to
        # "claim", whereas an unrelated later word (e.g. "claim System
        # Interest Accrued" -> "INTEREST" is 8 letters but two words
        # away) must never be reached for. Also checked against the RAW
        # (non-uppercased) token, not a normalized one -- an ordinary
        # lowercase English word next to "claim" (e.g. "...claim is for
        # a newborn and requires manual review") is exactly 8 letters
        # and would otherwise slip through; real all-letter Claim IDs
        # are always rendered in capitals in the OCR text itself. And
        # never accepted if it contains a negated word as a substring
        # (Step 2's list/logic).
        if not found:
            next_toks = _alnum_tokens(window)
            if next_toks:
                candidate = next_toks[0]
                if (
                    len(candidate) == 8
                    and candidate.isalpha()
                    and candidate.isupper()
                    and not _contains_negated_word(candidate)
                ):
                    found = candidate
        if found and found not in found_ids:
            found_ids.append(found)
    if found_ids:
        return found_ids
    # standalone short mentions, e.g. a sidebar line that's just the ID --
    # with no "claim" keyword anchoring it, stick to the strict digit-first
    # shape to avoid colliding with unrelated letter-first IDs (Member ID
    # has the exact same length/shape).
    for line in full_text.split("\n"):
        if len(_alnum_tokens(line)) <= 4:
            found = _find_compact_id(line, CLAIM_ID_RE)
            if found and _contains_negated_word(found):
                found = None
            if found and found not in found_ids:
                found_ids.append(found)
    return found_ids


\
# ---------------------------------------------------------------------------
# Claim ID rendered BELOW the ticket title (Appian ticket screens)
#
# The title line ends at the word "claim" and the ID itself is a separate OCR
# line underneath it (real dump: LINE "NEWBORNhold on claim ..." followed by
# LINE "KYBPW7Q0"). Neither text scan above can reach it:
#
#   * the "claim"-anchored scan reads FORWARD IN STREAM ORDER from the
#     keyword, so it walks into the toolbar text that shares the title's
#     visual row ("Ticket Info Re-triage PAUSE ESCALATE WITHDRAW RESOLVE")
#     and never sees the next line;
#   * the standalone-line fallback deliberately requires the digit-first
#     shape, so a letter-first ID like "KYBPW7Q0" is rejected there.
#
# The result was a screenshot with a perfectly readable Claim ID extracting
# none at all -- and since both Dataset A and Dataset B are Claims-only, the
# entire row silently disappeared from the workbook (its Ticket ID, status
# pill and dates went with it).
#
# This pass uses the OCR bounding boxes instead: the title token IS the
# label, so the ID is whatever 8-character ID-shaped token sits directly
# BELOW it (or immediately to its right). Because the match is anchored on
# the title AND on geometry, accepting either a digit-first or letter-first
# leading character is safe here -- the ambiguity that forces the digit-first
# rule on unanchored text does not exist. The literal Member ID (identical
# shape) is excluded explicitly.
# ---------------------------------------------------------------------------

_CLAIM_TITLE_TAIL_RE = re.compile(r"\bclaim$")
_CLAIM_TITLE_MAX_RIGHT_GAP_PX = 400


def find_claim_id_under_title(
    tokens: List[Token], full_text: str
) -> Optional[str]:
    """The Claim ID sitting directly under (or beside) a ticket title that
    ends with the word "claim". Returns None when there is no such title or
    no ID-shaped token adjacent to it."""
    exclude: set = set()
    mid = find_member_id_literal(full_text)
    if mid:
        exclude.add(mid.upper())
    for k, tok in enumerate(tokens):
        if _normalize(tok.text) in ("member id", "memberid") and k + 1 < len(tokens):
            exclude.add(tokens[k + 1].text.strip().upper())

    for i, title in enumerate(tokens):
        if not _CLAIM_TITLE_TAIL_RE.search(_normalize(title.text)):
            continue
        title_h = max(title.y1 - title.y0, 1)
        max_below_gap = max(3 * title_h, 70)
        best: Optional[Tuple[int, str]] = None
        for j, cand in enumerate(tokens):
            if j == i:
                continue
            text = cand.text.strip()
            if len(text) != 8 or not text.isalnum() or not text.isupper():
                continue
            if not any(c.isalpha() for c in text):
                continue  # a pure digit run is never a valid Claim ID
            candidate = text.upper()
            if candidate in exclude or _contains_negated_word(candidate):
                continue
            if not ANY_LEADING_ID_RE.match(candidate):
                continue
            below = (
                _h_overlap_frac(title.x0, title.x1, cand.x0, cand.x1) >= 0.3
                and cand.y0 >= title.y1 - 3
                and cand.y0 - title.y1 <= max_below_gap
            )
            beside = (
                _v_overlap_frac(title.y0, title.y1, cand.y0, cand.y1) >= 0.5
                and cand.x0 >= title.x1 - 3
                and cand.x0 - title.x1 <= _CLAIM_TITLE_MAX_RIGHT_GAP_PX
            )
            if not (below or beside):
                continue
            cost = (cand.y0 - title.y1) if below else (cand.x0 - title.x1)
            if best is None or cost < best[0]:
                best = (cost, candidate)
        if best:
            return best[1]
    return None


def find_claim_id(full_text: str) -> Optional[str]:
    ids = find_all_claim_ids(full_text)
    return ids[0] if ids else None


def find_member_id_literal(full_text: str) -> Optional[str]:
    """Literal 'MemberID <id>' Appian field (distinct from the OSC-prefixed
    Member ID (OSC) field used on Cortex/Member-match screens). A workflow
    word that happens to be 8 letters ('REPORTED' after a garbled 'Member
    ID' label) is never accepted."""
    for m in MEMBER_ID_LITERAL_RE.finditer(full_text):
        cand = m.group(1).upper()
        if not _contains_negated_word(cand):
            return cand
    return None


# ---------------------------------------------------------------------------
# "Code" -- plain regex pickup: the word(s) coming right after a
# standalone "Code" label anywhere in the running text, e.g.
# "Code WEAK_MEMBER_MATCH" -> WEAK_MEMBER_MATCH, "Code NEWBORN" ->
# NEWBORN, "Code: Invalid Pend" -> Invalid Pend. Third coverage path
# after the [TABLE] grid and the bbox pass. "Standalone" is the critical
# part: "Manage Code 81479 (HCPCS)" and "no_contract_found Type Code
# Description" contain the word Code as the SECOND word of a different
# label, so a match is rejected whenever an alphabetic word immediately
# precedes it.
# ---------------------------------------------------------------------------

_CODE_REGEX_RE = re.compile(
    # [ \t:] only -- NOT \s -- so the match can never cross a newline: in
    # the bbox dump a bare "text=Code" line is followed by "  WORD | ..."
    # on the NEXT line, and \s+ would swallow the newline and capture the
    # literal token "WORD" as a phantom code value.
    r"\bCode[: \t]+([A-Za-z][A-Za-z0-9_]*(?:[ ][A-Za-z][A-Za-z0-9_]*){0,2})"
)
_WORD_BEFORE_RE = re.compile(r"([A-Za-z][A-Za-z0-9_]*)[ \t]*$")


def find_code_regex(scan_text: str) -> Optional[str]:
    """The value regex-captured after a STANDALONE 'Code' label in the
    running text. Returns it when every occurrence in the dump agrees;
    None when there are no matches or occurrences conflict (defer to the
    geometric passes rather than guess)."""
    if not scan_text:
        return None
    found: List[str] = []
    for m in _CODE_REGEX_RE.finditer(scan_text):
        line_start = scan_text.rfind("\n", 0, m.start()) + 1
        wb = _WORD_BEFORE_RE.search(scan_text[line_start : m.start()])
        if wb and _normalize(wb.group(1)) not in ("text", "cell", "row", "line", "word"):
            continue  # "Manage Code ...", "Type Code ..." -- compound label
        # Trim the greedy capture down to the actual value. An ALL-CAPS /
        # underscore code (WEAK_MEMBER_MATCH, NEWBORN) is always a single
        # word -- whatever follows it ("Tenant Oscar") is the next label,
        # not part of the value. A title-case phrase ("Invalid Pend")
        # extends word by word but stops at any known neighboring label.
        words = m.group(1).split()
        first = words[0].rstrip(".,;:")
        value: Optional[str] = None
        if re.fullmatch(r"[A-Z][A-Z0-9_]{2,}", first):
            if _is_plausible_code_value(first):
                value = first
        else:
            stop_words = {"tenant", "system", "lob", "issue", "interest",
                          "date", "queue", "source", "state", "labels"}
            kept: List[str] = []
            for w in words:
                if _normalize(w) in stop_words:
                    break
                kept.append(w.rstrip(".,;:"))
            while kept:
                cand = " ".join(kept)
                if _is_plausible_code_value(cand):
                    value = cand
                    break
                kept.pop()
        if value and value not in found:
            found.append(value)
    if len(found) == 1:
        return found[0]
    return None


# ---------------------------------------------------------------------------
# "Code" -- geometric (nearest-bbox) value pickup.
#
# The Appian "Code" field's value sits immediately to the right of the
# literal "Code" label token; OCR sometimes drifts it, or fuzzy label
# matching grabs an unrelated neighboring word ("Count") instead. The value
# is therefore chosen purely geometrically: among all candidate tokens, take
# the one whose bounding box is NEAREST to the label's box, strongly
# preferring a token on the label's own visual row to its right (the real
# layout), falling back to plain center-to-center distance otherwise.
# ---------------------------------------------------------------------------

# A real Code value is a short workflow code: "NEWBORN",
# "WEAK_MEMBER_MATCH", "Invalid Pend" -- never a sentence, a date, or a
# number-only token.
_CODE_VALUE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*(?:[ _][A-Za-z0-9_]+){0,2}$")
_CODE_GLUED_RE = re.compile(r"^Code[:\s]+(?P<v>[A-Za-z][A-Za-z0-9_ ]{2,40})$", re.IGNORECASE)


def _is_plausible_code_value(text: str) -> bool:
    t = (text or "").strip().rstrip(".,;:")
    if not t or len(t) > 40:
        return False
    if not _CODE_VALUE_RE.match(t):
        return False
    norm = _normalize(t)
    if _is_ui_chrome(norm):
        return False
    for junk in _FIELD_VALUE_REJECTS.get("Code", []):
        if _matches_phrase(norm, _normalize(junk)):
            return False
    return True


# A parsed [TABLE] grid cell: "CELL[15,2] = Code NEWBORN" -> (15, 2, text).
_TABLE_CELL_RE = re.compile(r"^\s*CELL\[(\d+),(\d+)\]\s*=\s*(.*\S)\s*$", re.MULTILINE)
# Label+value glued inside ONE cell, anchored at the very start of the
# cell so "Manage Code 81479 (HCPCS)" (a procedure code) and
# "no_contract_found Type Code Description" (the word mid-sentence) can
# never match.
_CELL_CODE_GLUED_RE = re.compile(r"^Code[:\s]+(?P<v>\S.*)$", re.IGNORECASE)


def find_code_in_table_cells(raw_text: str) -> Optional[str]:
    """The 'Code' value resolved from the layout engine's [TABLE] grid --
    the structured first choice, since the table reconstruction has
    already solved the layout problem. Two shapes are handled:

      1. glued in one cell:      CELL[15,2] = Code NEWBORN
      2. label-only cell:        CELL[9,0] = Code   ->  value in the
         NEAREST cell horizontally on the SAME grid row to the right
         (smallest column offset), e.g. CELL[9,1] / CELL[9,2].

    Ambiguity guard: the same screen repeated across pages yields the
    same value many times (fine), but two DIFFERENT plausible values
    mean the grid is unreliable -- return None and let the bbox pass
    decide."""
    if not raw_text:
        return None
    cells: Dict[Tuple[int, int], str] = {}
    for m in _TABLE_CELL_RE.finditer(raw_text):
        cells[(int(m.group(1)), int(m.group(2)))] = m.group(3)

    found: List[str] = []
    for (r, c), text in cells.items():
        gm = _CELL_CODE_GLUED_RE.match(text.strip())
        if gm and _is_plausible_code_value(gm.group("v")):
            found.append(gm.group("v").strip())
            continue
        if _normalize(text) == "code":
            # nearest horizontal neighbor: same row, closest column right
            for (r2, c2) in sorted(
                (k for k in cells if k[0] == r and k[1] > c),
                key=lambda k: k[1] - c,
            ):
                if _is_plausible_code_value(cells[(r2, c2)]):
                    found.append(cells[(r2, c2)].strip().rstrip(".,;:"))
                    break
    if not found:
        return None
    distinct = set(found)
    if len(distinct) > 1:
        return None
    return found[0]


def find_code_nearest_bbox(tokens: List[Token]) -> Optional[str]:
    """Value of the 'Code' field, resolved by nearest bounding box to the
    literal 'Code' label token. Returns None when no label (or no plausible
    value near it) exists on this screen."""
    label: Optional[Token] = None
    for tok in tokens:
        stripped = tok.text.strip()
        # OCR glued label+value into one token ("Code NEWBORN") -- the
        # value is right there, no geometry needed.
        gm = _CODE_GLUED_RE.match(stripped)
        if gm and _is_plausible_code_value(gm.group("v")):
            return gm.group("v").strip()
        if _normalize(stripped) == "code":
            label = tok
            # keep scanning: a later glued form is even more specific
    if label is None:
        return None

    lcx = (label.x0 + label.x1) / 2.0
    lcy = (label.y0 + label.y1) / 2.0
    best: Optional[Token] = None
    best_dist: Optional[float] = None
    for tok in tokens:
        if tok is label:
            continue
        if not _is_plausible_code_value(tok.text):
            continue
        cx = (tok.x0 + tok.x1) / 2.0
        cy = (tok.y0 + tok.y1) / 2.0
        dist = ((cx - lcx) ** 2 + (cy - lcy) ** 2) ** 0.5
        # The genuine layout is label -> value on the same visual row,
        # value to the right: strongly prefer that arrangement so a chrome
        # word that happens to sit directly above/below never wins. Row
        # membership is judged by raw vertical overlap of the two boxes --
        # NOT by _same_visual_row, whose horizontal-gap limit rejects the
        # real value when the form leaves a wide gap between label and
        # value columns (seen on Appian: "Code" at x1=1455, "NEWBORN" at
        # x0=1636).
        v_frac = _v_overlap_frac(label.y0, label.y1, tok.y0, tok.y1)
        if v_frac >= 0.5 and tok.x0 >= label.x1 - 5:
            dist *= 0.2
        if best_dist is None or dist < best_dist:
            best, best_dist = tok, dist
    if best is None:
        return None
    return best.text.strip().rstrip(".,;:")


# ---------------------------------------------------------------------------
# Labeled date -- geometric (nearest-bbox) fallback pickup.
#
# Used for "Received date" (the Dataset A "Claim Received date" column) when
# the regex/fuzzy label->value pairing captured nothing: locate the literal
# label token and take the nearest DATE-shaped token that sits where a form
# value legitimately lives -- on the label's own visual row to its RIGHT, or
# stacked directly BELOW it. A date anywhere else (e.g. the Service start
# date rendered to the LEFT of a valueless "Received date" label) is never
# accepted: grabbing it would silently store the wrong field's value, which
# is worse than leaving the cell blank for gap fill to repair.
# ---------------------------------------------------------------------------

_DATE_VALUE_RE = re.compile(r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b")
# Value glued onto the label in one token ("Received date 07/20/2026").
_RECEIVED_GLUED_RE = re.compile(
    r"^Rec(?:ei|ie)ved\s*date[:\s]+(\d{1,2}/\d{1,2}/\d{2,4})\b", re.IGNORECASE
)
_RECEIVED_LABEL_NORMS = {"received date", "recieved date", "receiveddate"}

# Geometry guards (px): how far right on the same row / how far straight
# below a value may sit before it is considered unrelated to the label.
_LABELED_DATE_MAX_RIGHT_GAP_PX = 700
_LABELED_DATE_MAX_BELOW_GAP_PX = 100


def find_received_date_nearest_bbox(tokens: List[Token]) -> Optional[str]:
    """The 'Received date' value resolved by nearest bounding box to the
    literal label token. Returns just the date string, or None when the
    screen has no label, or no date token in a legitimate value position
    (right of the label on its row, or stacked directly below it)."""
    label: Optional[Token] = None
    for tok in tokens:
        stripped = tok.text.strip()
        gm = _RECEIVED_GLUED_RE.match(stripped)
        if gm:
            return gm.group(1)
        if _normalize(stripped) in _RECEIVED_LABEL_NORMS:
            label = tok
    if label is None:
        return None

    lcx = (label.x0 + label.x1) / 2.0
    lcy = (label.y0 + label.y1) / 2.0
    best: Optional[str] = None
    best_dist: Optional[float] = None
    for tok in tokens:
        if tok is label:
            continue
        dm = _DATE_VALUE_RE.search(tok.text)
        if not dm:
            continue
        v_frac = _v_overlap_frac(label.y0, label.y1, tok.y0, tok.y1)
        same_row_right = (
            v_frac >= 0.5
            and tok.x0 >= label.x1 - 5
            and (tok.x0 - label.x1) <= _LABELED_DATE_MAX_RIGHT_GAP_PX
        )
        tcx = (tok.x0 + tok.x1) / 2.0
        directly_below = (
            0 < (tok.y0 - label.y1) <= _LABELED_DATE_MAX_BELOW_GAP_PX
            and abs(tcx - lcx) <= max(150, (label.x1 - label.x0))
        )
        if not (same_row_right or directly_below):
            continue
        dist = ((tcx - lcx) ** 2 + ((tok.y0 + tok.y1) / 2.0 - lcy) ** 2) ** 0.5
        if same_row_right:
            dist *= 0.2  # the genuine form layout -- prefer it strongly
        if best_dist is None or dist < best_dist:
            best, best_dist = dm.group(1), dist
    return best


# ===========================================================================
# 5. Value post-processing
# ===========================================================================


def _clean_value(v: str) -> str:
    v = (v or "").strip()
    v = v.strip(" :-\u2013\u2014\"'()")
    v = re.sub(r"\s{2,}", " ", v)
    return v.strip()


# Fields where we should trim to just the currency amount (drop any OCR
# noise glued on after it, e.g. "$154.09 10010").
_CURRENCY_FIELDS = {
    "Billed", "Allowed", "Pay provider", "Pay member", "Member responsibilty",
}

# Fields that should always hold a date and/or timestamp. Used by the
# date-shape guard below to catch an unrelated token being wrongly
# captured as one of these (see _extract_fields_for_document).
_DATE_FIELDS = {
    "Created On", "Updated On", "Resolved On", "Closed On", "Due",
    "Received date", "Service Start Date", "Service End Date", "Reported On",
}

# Subset of _DATE_FIELDS that must literally START with a date. These are
# cleaned by _clean_datetime_start: OCR glue is re-spaced, every segment
# that starts with a date keeps only its "date [time] [AM/PM] [TZ]" lead,
# duplicate renderings of the same instant collapse to one, and junk
# segments are eliminated. A value with no leading date anywhere is left
# exactly as OCR'd.
_STRICT_START_DATE_FIELDS = {"Created On", "Updated On", "Closed On", "Resolved On"}

# ---------------------------------------------------------------------
# SLA cleaning -- the raw capture is a noisy pile like
#   "All Holds; 7/23/20264: 09 PM EDT; minus High$) (7 working days;
#    minus High$)(7 working days"
# but the only information the SLA column should carry is the duration:
# "7 working days" (plus an hours figure when one is present, e.g.
# "48 hours"). Everything else -- queue names, stray dates, currency
# fragments, duplicated phrases -- is eliminated.
# ---------------------------------------------------------------------

_SLA_WORKING_DAYS_RE = re.compile(r"(\d+)\s*working\s*days?", re.IGNORECASE)
_SLA_PLAIN_DAYS_RE = re.compile(r"(\d+)\s*(?:business\s*)?days?\b", re.IGNORECASE)
_SLA_HOURS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:hrs?|hours?)\b", re.IGNORECASE)


def _clean_sla(raw: Any) -> str:
    if isinstance(raw, list):
        raw = "; ".join(str(v) for v in raw if v)
    s = _clean_value(str(raw or ""))
    if not s:
        return s
    parts: List[str] = []
    seen: set = set()
    for m in _SLA_WORKING_DAYS_RE.finditer(s):
        p = f"{m.group(1)} working days"
        if p not in seen:
            seen.add(p)
            parts.append(p)
    for m in _SLA_HOURS_RE.finditer(s):
        p = f"{m.group(1)} hours"
        if p not in seen:
            seen.add(p)
            parts.append(p)
    if not parts:
        # no working-days phrase at all -- accept a plain "N days" so a
        # screen that renders the SLA without the word "working" still
        # yields a duration
        for m in _SLA_PLAIN_DAYS_RE.finditer(s):
            p = f"{m.group(1)} days"
            if p not in seen:
                seen.add(p)
                parts.append(p)
    return ", ".join(parts) if parts else s


# ---------------------------------------------------------------------
# Date-time cleaning for Created/Updated/Resolved/Closed On -- raw OCR:
#   "7/23/2026 4:09 PM EDT; 7/23/20264:09 PM EDT Resolved (Suppressed"
# cleans to "7/23/2026 4:09 PM EDT". Rules: a kept segment must START
# with a date; glued year+time / "9: 07" / "PMEDT" spacing is repaired;
# duplicate renderings of the same instant collapse; a bare month/day
# with no year ("7/23") is completed with the CURRENT year; a value with
# no leading date anywhere is returned untouched, exactly as OCR'd.
# ---------------------------------------------------------------------

_DT_LEAD_FULL_RE = re.compile(
    r"^(\d{1,2}/\d{1,2}/\d{4})"
    r"(?:\s*(\d{1,2}:\d{2}(?:\s*[AP]M)?(?:\s*[A-Z]{2,4})?))?",
    re.IGNORECASE,
)
_DT_LEAD_NOYEAR_RE = re.compile(
    r"^(\d{1,2}/\d{1,2})(?![\d/])"
    r"(?:\s*(\d{1,2}:\d{2}(?:\s*[AP]M)?(?:\s*[A-Z]{2,4})?))?",
    re.IGNORECASE,
)


_ANY_DATE_SHAPE_RE = re.compile(r"\d{1,2}/\d{1,2}")


def _clean_date_only(raw: Any) -> str:
    """Cleaner for pure DATE fields (Received date, Service Start/End
    Date): keeps ONLY the full date substrings, deduplicated in order --
    "06/19/2026 H112662924410001; 06/19/2026" -> "06/19/2026". A value
    with no full date but starting with a year-less month/day gets the
    current year; otherwise the original is returned untouched."""
    if isinstance(raw, list):
        raw = "; ".join(str(v) for v in raw if v)
    s = _clean_value(str(raw or ""))
    if not s:
        return s
    dates: List[str] = []
    for m in re.finditer(r"\b\d{1,2}/\d{1,2}/\d{4}\b", s):
        if m.group(0) not in dates:
            dates.append(m.group(0))
    if dates:
        return "; ".join(dates)
    nm = re.match(r"^(\d{1,2}/\d{1,2})(?![\d/])", s)
    if nm:
        return f"{nm.group(1)}/{_dt.date.today().year}"
    # No date content at all ("$0.00" grabbed from a neighboring amount
    # field) -> ELIMINATE rather than keep junk in a date column.
    return ""


def _clean_datetime_start(raw: Any) -> str:
    if isinstance(raw, list):
        raw = "; ".join(str(v) for v in raw if v)
    s = _clean_value(str(raw or ""))
    if not s:
        return s
    # Repair OCR glue BEFORE matching: "20264:09" -> "2026 4:09",
    # "9: 07" -> "9:07", "07AMEDT" -> "07 AM EDT", "PMEDT" -> "PM EDT".
    # The AM/PM+timezone split must run FIRST and without word boundaries
    # -- "AMEDT" is one unbroken word, so \b can never separate it.
    t = re.sub(r"(\d)\s*([AP]M)([A-Z]{2,4})\b", r"\1 \2 \3", s)
    t = re.sub(r"(\d{4})(\d{1,2}:)", r"\1 \2", t)
    t = re.sub(r":\s+(\d)", r":\1", t)
    t = re.sub(r"(\d)\s*([AP]M)\b", r"\1 \2", t, flags=re.IGNORECASE)
    t = re.sub(r"\b([AP]M)\s*([A-Z]{2,4})\b", r"\1 \2", t, flags=re.IGNORECASE)

    pieces: List[str] = []
    seen: set = set()
    for seg in re.split(r"[;|]", t):
        seg = seg.strip()
        if not seg:
            continue
        m = _DT_LEAD_FULL_RE.match(seg)
        if m:
            date_part = m.group(1)
        else:
            m = _DT_LEAD_NOYEAR_RE.match(seg)
            if not m:
                continue  # segment does not START with a date -> junk, drop
            date_part = f"{m.group(1)}/{_dt.date.today().year}"
        time_part = m.group(2)
        piece = date_part
        if time_part:
            piece += " " + re.sub(r"\s+", " ", time_part.strip()).upper()
        key = re.sub(r"[^0-9A-Z]", "", piece.upper())
        if key not in seen:
            seen.add(key)
            pieces.append(piece)
    if not pieces:
        return s  # no leading date anywhere -- keep exactly as OCR'd
    return "; ".join(pieces)


# Person-name fields where a workflow-status word can get wrongly grabbed
# as the "value" (seen on the queue/list screen, where a stats-table
# column header like "Escalated Resolved Withdrawn Total Closed" sits
# right next to the real ticket rows and "Withdrawn" got captured as
# "Resolved By"). None of these words can ever be a legitimate name.
_NAME_FIELDS = {"Assignee", "Resolved By", "Updated By", "Submitter"}
_STATUS_WORD_RE = re.compile(
    r"^(withdrawn|resolved|escalated|closed|suppressed|status|action|"
    r"in progress|paused|open|pending)$",
    re.IGNORECASE,
)
_DATE_SHAPE_RE = re.compile(r"\d")

# The "Blocker" field is expected to be a descriptive status/reason
# string, never a number -- any digit at all (whole number or decimal)
# is unconditionally flagged for manual review, at any cost, regardless
# of how confident the fuzzy label match otherwise was.
_BLOCKER_FIELDS = {"Blocker"}
_ANY_NUMBER_RE = re.compile(r"\d")


def _looks_like_date(value: str) -> bool:
    """Very deliberately permissive -- just requires at least one digit,
    since OCR mangles date formatting heavily (missing spaces/slashes,
    glued AM/PM/timezone, etc.) and the goal here is only to reject
    obviously-wrong non-date tokens, not to validate real date formats."""
    return bool(_DATE_SHAPE_RE.search(value))


def _postprocess_value(canonical: str, raw_value: str) -> str:
    value = _clean_value(raw_value)
    if canonical in _CURRENCY_FIELDS:
        m = CURRENCY_RE.search(value)
        if m:
            return m.group(0)
    return value


# ---------------------------------------------------------------------
# 5b. Due-date resolution -- when the fuzzy label pass collects more
#     than one candidate value for "Due" (e.g. two nearby date-shaped
#     strings both loosely matched the label), the greatest/latest date
#     wins and is re-rendered as dd/mm/yy regardless of the format it
#     was originally OCR'd in.
# ---------------------------------------------------------------------

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

# Each pattern is tried in order against the *whole* cleaned value.
_DATE_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # 25/07/2026, 25-07-2026, 25.07.2026 (day-first -- matches the
    # locale used throughout the source screens)
    (re.compile(r"^(?P<d>\d{1,2})[/\-.](?P<m>\d{1,2})[/\-.](?P<y>\d{2,4})$"), "dmy"),
    # 2026-07-25 (ISO, unambiguous regardless of locale)
    (re.compile(r"^(?P<y>\d{4})[/\-.](?P<m>\d{1,2})[/\-.](?P<d>\d{1,2})$"), "ymd"),
    # 25 Jul 2026 / 25-Jul-2026
    (re.compile(r"^(?P<d>\d{1,2})\s*[-\s]\s*(?P<mon>[A-Za-z]{3,9})\.?\s*[-\s,]\s*(?P<y>\d{2,4})$"), "dmon_y"),
    # Jul 25, 2026 / Jul 25 2026
    (re.compile(r"^(?P<mon>[A-Za-z]{3,9})\.?\s+(?P<d>\d{1,2}),?\s+(?P<y>\d{2,4})$"), "mon_d_y"),
]


def _normalize_year(y: int) -> int:
    """2-digit years: 00-69 -> 2000-2069, 70-99 -> 1970-1999 (standard
    pivot-year convention; fine here since these are near-term ticket
    dates, never 1970s data)."""
    if y < 100:
        return 2000 + y if y < 70 else 1900 + y
    return y


def _parse_date(raw: str) -> Optional[_dt.date]:
    """Best-effort parse of a date-shaped string into a real date.
    Returns None if the string doesn't match any known format or
    describes an invalid calendar date (e.g. 31/02/2026)."""
    s = _clean_value(raw or "")
    if not s:
        return None
    for pattern, kind in _DATE_PATTERNS:
        m = pattern.match(s)
        if not m:
            continue
        try:
            if kind == "dmy":
                d, mo, y = int(m.group("d")), int(m.group("m")), _normalize_year(int(m.group("y")))
            elif kind == "ymd":
                y, mo, d = _normalize_year(int(m.group("y"))), int(m.group("m")), int(m.group("d"))
            else:  # "dmon_y" or "mon_d_y"
                mo = _MONTHS.get(m.group("mon")[:3].lower())
                if not mo:
                    continue
                d, y = int(m.group("d")), _normalize_year(int(m.group("y")))
            return _dt.date(y, mo, d)
        except ValueError:
            continue  # syntactically matched but not a real calendar date
    return None


def _format_ddmmyy(d: _dt.date) -> str:
    return d.strftime("%d/%m/%y")


# Same shapes as _DATE_PATTERNS but WITHOUT the trailing "$" anchor, so a
# date can be matched at the very start of a value that also carries
# trailing text (e.g. a glued-on time/timezone: "25/07/2026 10:00 AM").
_DATE_PATTERNS_LEADING: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"^(?P<d>\d{1,2})[/\-.](?P<m>\d{1,2})[/\-.](?P<y>\d{2,4})"), "dmy"),
    (re.compile(r"^(?P<y>\d{4})[/\-.](?P<m>\d{1,2})[/\-.](?P<d>\d{1,2})"), "ymd"),
    (re.compile(r"^(?P<d>\d{1,2})\s*[-\s]\s*(?P<mon>[A-Za-z]{3,9})\.?\s*[-\s,]\s*(?P<y>\d{2,4})"), "dmon_y"),
    (re.compile(r"^(?P<mon>[A-Za-z]{3,9})\.?\s+(?P<d>\d{1,2}),?\s+(?P<y>\d{2,4})"), "mon_d_y"),
]


def _parse_leading_date(raw: str) -> Optional[Tuple[_dt.date, str]]:
    """Parse a date anchored at the START of the string (unlike
    _parse_date, which requires the WHOLE string to be just a date).
    Returns (date, remainder_of_string_after_the_date) or None if no
    known date shape is found at the start."""
    s = _clean_value(raw or "")
    if not s:
        return None
    for pattern, kind in _DATE_PATTERNS_LEADING:
        m = pattern.match(s)
        if not m:
            continue
        try:
            if kind == "dmy":
                d, mo, y = int(m.group("d")), int(m.group("m")), _normalize_year(int(m.group("y")))
            elif kind == "ymd":
                y, mo, d = _normalize_year(int(m.group("y"))), int(m.group("m")), int(m.group("d"))
            else:  # "dmon_y" or "mon_d_y"
                mo = _MONTHS.get(m.group("mon")[:3].lower())
                if not mo:
                    continue
                d, y = int(m.group("d")), _normalize_year(int(m.group("y")))
            return _dt.date(y, mo, d), s[m.end():]
        except ValueError:
            continue  # syntactically matched but not a real calendar date
    return None


def _format_ddmmyyyy(d: _dt.date) -> str:
    return d.strftime("%d/%m/%Y")


_FULL_DATE_SUBSTRING_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")


def _resolve_due_date(value: Any) -> Any:
    """The Due field keeps ONLY a full date -- everything else that got
    captured alongside it ("07/15/2026; Invalid Pend hold on claim
    O3vYEXK8") is eliminated. Every full date substring is collected from
    every candidate, the latest one wins, and it is returned EXACTLY as it
    appeared in the OCR (no re-rendering). Month-first parsing is tried
    before day-first (the corpus locale is US). No full date anywhere ->
    empty, never junk."""
    candidates = value if isinstance(value, list) else [value]
    found: List[Tuple[_dt.date, str]] = []
    for cv in candidates:
        if not isinstance(cv, str):
            continue
        for m in _FULL_DATE_SUBSTRING_RE.finditer(cv):
            a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            d: Optional[_dt.date] = None
            try:
                d = _dt.date(y, a, b)  # month-first (US-format screens)
            except ValueError:
                try:
                    d = _dt.date(y, b, a)  # day-first fallback
                except ValueError:
                    pass
            if d is not None:
                found.append((d, m.group(0)))
    if not found:
        return ""
    return max(found, key=lambda pair: pair[0])[1]


# Dropdown/select placeholder text that sometimes gets OCR'd out of an
# unfilled "Reason" field (e.g. "-- Select --", "Select", "Select...").
# When this shows up as the *value*, it isn't a real reason -- it means
# the field was left unset in the UI -- so it gets flagged for manual
# review instead of being silently treated as legitimate data.
_SELECT_PLACEHOLDER_RE = re.compile(r"^-{0,3}\s*select\b", re.IGNORECASE)
# Broader unfilled-input placeholder text seen across several apps (not
# just the Reason dropdown) -- e.g. Member match Cortex shows "Enter a
# value" verbatim in its input box when no lookup has actually been
# performed, and Appian a few other placeholders. Any field carrying one
# of these is effectively empty and should be flagged for manual review
# rather than silently treated as real data.
_GENERIC_PLACEHOLDER_RE = re.compile(
    r"^(-{0,3}\s*select\b.*|enter a value|n/a|--+)$", re.IGNORECASE
)


# Labels that are *recognized* (so they don't get accidentally swallowed
# as some other field's value) but deliberately not kept in the output --
# section headers / instructional text, not real field values.
# "Place of service" and "(System) Interest Accrued" are labels of
# NEIGHBORING fields on the Cortex claim screen that aren't in the schema:
# unrecognized, they look like free text sitting right next to a real
# label (Claim Type / Skill / Code) and get wrongly grabbed as its value.
_IGNORE_LABELS = {
    "Reason Description", "Reported On", "Reporter", "Additional Info",
    "Changes Made", "Description", "Reason for changes",
    "Created By", "Closed By", "System",
    "Place of service", "Interest Accrued", "System Interest Accrued",
    # More neighboring labels on the Cortex claim screen (confirmed in the
    # real OCR dumps) that would otherwise be grabbed as values for Claim
    # Type / Submission Type / the date fields:
    "NucleusID", "Vendor source", "Diagnosis related group",
}


# ---------------------------------------------------------------------
# 5c. Field-specific value constraints (business rules)
# ---------------------------------------------------------------------

# Fields whose value is a small CLOSED vocabulary per business rule.
# Maps canonical output value -> raw variants OCR/UI may show for it.
# An extracted value is snapped to the closest canonical value; a value
# resembling none of them is dropped and flagged for manual review --
# these fields can hold nothing else, so anything unrecognized is a
# mis-grab of neighboring screen text, never new data.
_CANONICAL_VALUE_FIELDS: Dict[str, List[Tuple[str, List[str]]]] = {
    # Per business rule, Claim State is ONLY ever Blocked or Audit -- and
    # any "blocking"-type wording on the screen means the claim is Blocked.
    "Claim State": [
        ("Blocked", ["blocked", "block", "blocking", "blocking type"]),
        ("Audit", ["audit", "audited"]),
    ],
    # Submission Type is Electronic. "EDI file" (and OCR garbles of it,
    # e.g. "EDIfle") IS the electronic submission channel -- normalize it
    # to Electronic rather than storing transport jargon or blanking out.
    "Submission Type": [
        ("Electronic", ["electronic", "electronically", "edi file", "edifle"]),
    ],
    # Per business rule, Claim Type is only ever professional or
    # institutional (the real screens display "HCFA-Professional").
    "Claim Type": [
        ("HCFA-Professional", ["hcfa professional", "professional", "hcfa"]),
        ("Institutional", ["institutional", "institution", "institute", "ub institutional"]),
    ],
}

# A blocking-type CODE standing where the Claim State should be (seen in
# the real dumps: "NO_MEMBER_MATCH", "claim_attachment_review") means the
# claim is Blocked -- when a claim is blocked the UI shows the blocking
# reason right next to / in place of the state, and the pairing can grab
# it. Per business rule: detecting a blocking type == the claim is Blocked.
_BLOCKING_CODE_RE = re.compile(r"^[A-Za-z]+(?:_[A-Za-z0-9]+)+$")

# Known mis-grab values for specific fields -- text of another label or
# section that sits next to the real field on screen and gets wrongly
# captured as its value. Matched as substring / fuzzily against the
# normalized extracted value; a hit drops the value and flags the field.
_FIELD_VALUE_REJECTS: Dict[str, List[str]] = {
    "Skill": ["interest accrued", "system interest accrued", "atclm"],
    "Code": ["interest accrued", "system interest accrued", "atclm"],
    # "Manage" is the blocker card's button, "Active holds" its section
    # header, "Tags" a neighboring panel title, "Save note" the notes
    # button, "EDI file" transport jargon (all confirmed in the real
    # dumps) -- none is ever a real blocker description. ("Claim Type"
    # junk like "Place of service" and "NucleusID" is handled by its
    # closed vocabulary above.)
    "Blocker": ["manage", "active holds", "tags", "save note", "edi file", "edifile"],
}

# Real blocker values are short codes ("NO_MEMBER_MATCH") or short label
# phrases ("Payer control number") -- a whole sentence or question grabbed
# from a notes/details panel ("docs were already triaged correctly") is
# never a real blocker.
_BLOCKER_MAX_WORDS = 4

# Skill/Code are short workflow codes (e.g. "NEWBORN") -- symbols like
# '%' or ';' in the value only ever mean OCR junk got grabbed.
_CODE_JUNK_FIELDS = {"Skill", "Code"}
_CODE_JUNK_RE = re.compile(r"[%;]")


def _matches_phrase(value_norm: str, phrase_norm: str) -> bool:
    """True when a normalized value contains or closely resembles a
    normalized phrase (substring OR >= 0.8 fuzzy similarity)."""
    if not value_norm or not phrase_norm:
        return False
    if phrase_norm in value_norm:
        return True
    return difflib.SequenceMatcher(None, value_norm, phrase_norm).ratio() >= 0.8


\
# ---------------------------------------------------------------------------
# Page-wide OCR-tolerant scan for the claim state
#
# The claim state is sometimes unreadable as a label/value pair -- the label is
# garbled, the value is split across OCR lines, or the screen is an Appian view
# that has no Claim State field at all -- yet the word itself IS somewhere in
# the formatted text. This scan looks for it anywhere on the page, tolerating
# the ways OCR mangles a short word:
#
#   "AUDIT"  "AU DIT"  "AUD IT"  "A U D I T"  "AUDlT"  "AUD1T"  "4UDIT"
#   "BLOCKED"  "BLO CKED"  "BL0CKED"  "BLOCKING"
#
# Letters may be separated by whitespace/punctuation/line breaks (up to
# _CLAIM_STATE_PAGE_GAP characters), and each letter also accepts its usual
# OCR look-alikes (I/1/l, O/0, A/4, B/8, E/3, T/7).
#
# The separator deliberately excludes LETTERS AND DIGITS, i.e. whole words are
# NOT allowed between the letters. Allowing them would make "A ... U ... D ...
# I ... T" match nearly every page in the batch (those five letters occur in
# ordinary words constantly) and would silently flip every claim to Audit.
# Widen _CLAIM_STATE_PAGE_GAP if a real screen splits a word more heavily.
# ---------------------------------------------------------------------------

_CLAIM_STATE_PAGE_GAP = 3

# Per-letter OCR look-alike classes.
_OCR_LETTER_CLASSES = {
    "A": "[Aa4]", "B": "[Bb8]", "C": "[Cc]", "D": "[Dd]", "E": "[Ee3]",
    "G": "[Gg9]", "I": "[Ii1l|!]", "K": "[Kk]", "L": "[Ll1|]", "N": "[Nn]",
    "O": "[Oo0]", "T": "[Tt7]", "U": "[UuV]",
}


def _ocr_tolerant_word_re(word: str, gap: int = _CLAIM_STATE_PAGE_GAP) -> re.Pattern:
    """A regex matching one word even when OCR has split it with spaces/line
    breaks/punctuation and confused individual letters. Never matches across
    other words -- the separator excludes alphanumerics by design."""
    parts = [_OCR_LETTER_CLASSES.get(ch.upper(), re.escape(ch)) for ch in word]
    separator = r"[^A-Za-z0-9]{0,%d}" % max(gap, 0)
    return re.compile(separator.join(parts))


# Checked in order, so the more specific state wins: a page that names Audit
# anywhere is treated as Audit (per business rule the two are exclusive).
_PAGE_CLAIM_STATE_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("Audit", _ocr_tolerant_word_re("AUDIT")),
    ("Blocked", _ocr_tolerant_word_re("BLOCKED")),
    ("Blocked", _ocr_tolerant_word_re("BLOCKING")),
]

# UI chrome that contains the state word without describing this claim's
# state ("Audit trail", "Audit log", "Blocked tab"). An occurrence followed by
# one of these is skipped and the scan keeps looking.
_PAGE_STATE_CHROME_RE = re.compile(
    r"^[^A-Za-z0-9]{0,3}(trail|trails|log|logs|logging|history|report|reports|"
    r"tab|tabs|menu|filter|filters|column|queue)\b",
    re.IGNORECASE,
)


def find_claim_state_in_page(text: str) -> Tuple[str, str, bool]:
    """Scan a whole formatted-text page for the claim state.

    Returns (state, matched_text, was_clean) -- state is "" when neither word
    appears. was_clean is False when the match was garbled/split, which is
    worth flagging for a human even though the state itself is unambiguous."""
    haystack = text or ""
    for state, pattern in _PAGE_CLAIM_STATE_PATTERNS:
        for m in pattern.finditer(haystack):
            matched = m.group(0)
            if _PAGE_STATE_CHROME_RE.match(haystack[m.end():m.end() + 12]):
                continue  # "Audit trail" / "Blocked column" -- not a state
            was_clean = bool(re.fullmatch(r"[A-Za-z]+", matched))
            return state, matched, was_clean
    return "", "", True


def _apply_value_constraints(fields: Dict[str, Any], flags: Dict[str, str]) -> None:
    """Enforce the per-field business rules above on already-extracted
    values: snap closed-vocabulary fields to their canonical value, and
    reject known mis-grab values. Mutates fields/flags in place."""
    for field, rules in _CANONICAL_VALUE_FIELDS.items():
        if field not in fields or field in flags:
            continue
        raw = fields[field]
        candidates = raw if isinstance(raw, list) else [raw]
        snapped = None
        for cv in candidates:
            if not isinstance(cv, str):
                continue
            vnorm = _normalize(cv)
            for canonical, variants in rules:
                if any(_matches_phrase(vnorm, _normalize(v)) for v in variants):
                    snapped = canonical
                    break
            if snapped:
                break
        if not snapped and field == "Claim State":
            # A blocking-type code standing in for the state means Blocked.
            for cv in candidates:
                if isinstance(cv, str) and _BLOCKING_CODE_RE.match(cv.strip()):
                    snapped = "Blocked"
                    break
        if snapped:
            fields[field] = snapped
        else:
            stray = fields.pop(field)
            allowed = " / ".join(c for c, _ in rules)
            flags[field] = (
                f"Value {stray!r} is not a recognized {field} value "
                f"(expected {allowed}) -- needs manual review"
            )
    for field, reject_phrases in _FIELD_VALUE_REJECTS.items():
        if field not in fields or field in flags:
            continue
        raw = fields[field]
        candidates = raw if isinstance(raw, list) else [raw]
        kept: List[Any] = []
        rejected = False
        for cv in candidates:
            bad = False
            if isinstance(cv, str):
                vnorm = _normalize(cv)
                if field in _CODE_JUNK_FIELDS and _CODE_JUNK_RE.search(cv):
                    bad = True
                elif field == "Blocker" and len(cv.split()) > _BLOCKER_MAX_WORDS:
                    bad = True
                elif field == "Blocker" and cv[:1].islower():
                    # Real blockers are CODES ("NO_MEMBER_MATCH") or
                    # Capitalized label phrases ("Payer control number") --
                    # a lowercase-starting fragment ("and denials") is a
                    # mid-sentence OCR scrap, never a blocker.
                    bad = True
                elif any(_matches_phrase(vnorm, _normalize(p)) for p in reject_phrases):
                    bad = True
            if bad:
                rejected = True
            else:
                kept.append(cv)
        if rejected and not kept:
            fields.pop(field)
            flags[field] = (
                f"Value {raw!r} looks like neighboring screen text, not a real "
                f"{field} value -- needs manual review"
            )
        elif rejected:
            fields[field] = kept[0] if len(kept) == 1 else kept


# ===========================================================================
# 6. Page-type classification
# ===========================================================================

_APP_SIGNATURES: Dict[str, List[str]] = {
    "Appian- Home": ["claims", "hold", "ticket info", "assignee", "unarchive"],
    "Appian - Popup": ["sub status", "sub-status", "cancel"],
    "Cortex- About this claim": ["about this claim", "payment", "medical claim lines", "readjudicate"],
    # "cortex member" / "claims referrals" / "only medical claims" identify
    # the member's claims-list view (seen in real dumps as a "Cortex
    # -Member" header) -- same product area, previously misclassified as
    # Appian- Home on a single stray "claims" keyword.
    "Cortex - Patient - Matched member": [
        "matched member", "member:", "cortex member", "claims referrals",
        "only medical claims",
    ],
    "Cortex - Patient - Submitted Info": ["submitted info"],
    "Cortex - Patient - Submitted Info - Blocker": ["submitted info", "block status"],
    "Member match Cortex": ["member match cortex", "search by member id"],
    "VERINT": ["verint", "knowledge"],
    "Rosco": ["rosco", "editing document", "document title", "document type"],
    "Elbow": ["elbow", "demographic", "search for person and company", "basic info contract eligibility"],
}

# Apps recognized by an unambiguous literal-name token match (same
# high-confidence, checked-first treatment as VERINT) rather than a
# fuzzy multi-keyword score -- the app's own name appearing verbatim is
# itself the strongest possible signal.
_LITERAL_NAME_APPS = ("verint", "rosco", "elbow")


def classify_document(tokens: List[Token]) -> str:
    joined = _normalize(" ".join(t.text for t in tokens))
    # These are checked first and win outright on a literal token match:
    # each is an unambiguous, high-confidence signal (the app's own name),
    # and should never lose a tie-break to a coincidental 1-keyword score
    # elsewhere (e.g. stray "member" mentions in an unrelated help page).
    for tok in tokens:
        tnorm = _normalize(tok.text)
        if tnorm in _LITERAL_NAME_APPS:
            return tnorm.capitalize() if tnorm != "verint" else "VERINT"
    best_app, best_score = "Unknown", 0
    for app, keywords in _APP_SIGNATURES.items():
        # VERINT stays literal-name only; Rosco/Elbow now also score by
        # keywords (their name token is often OCR-glued into a longer
        # header line, e.g. "O elbow Oscar Search for Person...", which
        # the exact-token pass above can't see).
        if app == "VERINT":
            continue
        score = sum(1 for kw in keywords if _normalize(kw) in joined)
        if score > best_score:
            best_app, best_score = app, score
    return best_app


# ===========================================================================
# 7. Main extraction per document
# ===========================================================================


@dataclass
class DocumentResult:
    document_index: int
    detected_application: str
    fields: Dict[str, Any]
    extra_fields: Dict[str, Any]
    flags: Dict[str, str] = dc_field(default_factory=dict)


# Caps on how far a value is allowed to sit from its label -- keeps the
# geometric search from reaching across the whole screen and grabbing
# unrelated text when the field is genuinely blank.
_MAX_RIGHT_GAP_PX = 600
# Mild penalty applied to below-the-label candidates so that, at equal
# distance, the standard left-to-right "Label: value" reading wins.
_BELOW_WEIGHT = 1.5


def _pick_value_index(
    tokens: List[Token],
    hit: LabelHit,
    all_hits: List[LabelHit],
    claimed: set,
) -> Optional[int]:
    """Choose the token that holds this label's VALUE, using the OCR
    bounding boxes instead of blind next-in-stream adjacency:

      1. nearest token to the RIGHT on the same visual row, else
      2. nearest token directly BELOW the label (column layouts, where a
         whole row of labels is followed by a row of values -- stream
         adjacency pairs those completely wrong or not at all);

    preferring candidates that don't themselves look like a label. Tokens
    already claimed by a stronger hit, and confident (>=0.85) label
    matches, are never eligible. Falls back to the old next-in-stream rule
    when geometry finds nothing."""
    i, span = hit.token_index, hit.span
    span_idxs = set(range(i, i + span))
    strong_label_idxs: set = set()
    any_label_idxs: set = set()
    for h2 in all_hits:
        rng = range(h2.token_index, h2.token_index + h2.span)
        any_label_idxs.update(rng)
        if h2.score >= 0.85:
            strong_label_idxs.update(rng)

    lx0 = min(tokens[k].x0 for k in span_idxs)
    ly0 = min(tokens[k].y0 for k in span_idxs)
    lx1 = max(tokens[k].x1 for k in span_idxs)
    ly1 = max(tokens[k].y1 for k in span_idxs)
    label_h = max(ly1 - ly0, 1)
    max_below_gap = max(4 * label_h, 90)

    preferred: List[Tuple[float, int]] = []  # candidates that match no label
    weak: List[Tuple[float, int]] = []       # weak/borderline label matches
    for k, t in enumerate(tokens):
        if k in claimed or k in span_idxs or k in strong_label_idxs:
            continue
        if not t.text.strip():
            continue
        cost = None
        if _v_overlap_frac(ly0, ly1, t.y0, t.y1) >= 0.5 and t.x0 >= lx1 - 3:
            gap = max(t.x0 - lx1, 0)
            if gap <= _MAX_RIGHT_GAP_PX:
                cost = float(gap)
        elif _h_overlap_frac(lx0, lx1, t.x0, t.x1) >= 0.3 and t.y0 >= ly1 - 3:
            vgap = max(t.y0 - ly1, 0)
            if vgap <= max_below_gap:
                cost = _BELOW_WEIGHT * vgap
        if cost is None:
            continue
        (weak if k in any_label_idxs else preferred).append((cost, k))

    # A weak/borderline label hit shouldn't starve this label of its value
    # (same principle as before) -- but a token matching NO label at all is
    # always the better value candidate when one exists. For currency
    # fields, a candidate carrying an actual $ amount beats a nearer plain
    # word (glued labels like "Billed amount" otherwise leave a stray
    # "amount" fragment sitting closest to the label).
    prefer_currency = hit.canonical in _CURRENCY_FIELDS
    for bucket in (preferred, weak):
        if not bucket:
            continue
        if prefer_currency:
            with_currency = [bc for bc in bucket if CURRENCY_RE.search(tokens[bc[1]].text)]
            if with_currency:
                return min(with_currency)[1]
        return min(bucket)[1]

    j = i + span  # stream-order fallback (the pre-geometry behavior)
    if j < len(tokens) and j not in claimed and j not in strong_label_idxs and tokens[j].text.strip():
        return j
    return None


def _extract_fields_for_document(
    doc: Document, all_variable_names: List[str]
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, str]]:
    fields: Dict[str, Any] = {}
    extra: Dict[str, Any] = {}
    flags: Dict[str, str] = {}

    if doc.parser_fallback:
        flags["Parser"] = (
            "No [BOXES] token data could be parsed for this document -- fell "
            "back to plain text lines with approximate positions; extractions "
            "may need manual review"
        )

    label_vocab = [
        v for v in all_variable_names if v not in _LABELS_WITHOUT_LITERAL_TEXT
    ] + list(_IGNORE_LABELS)

    # One OCR token can carry several fields glued side by side, or a label
    # glued onto its own value -- split those apart first so every label
    # and every value is its own token (see _split_multifield_tokens).
    tokens = _split_multifield_tokens(doc.tokens, label_vocab)

    hits = find_label_hits(tokens, label_vocab)

    # Conflict resolution: a single token can end up matching two
    # different canonical labels at different confidence levels (e.g. a
    # value like "PROCESS_CLAIM" sitting right after "Goal" can *itself*
    # weakly resemble "Provider Claim#"). Process strongest matches
    # first and permanently claim both the label token(s) and the value
    # token, so a weaker/wrong match on an already-claimed token is
    # simply discarded rather than overwriting a correct extraction.
    # The value for each label is chosen GEOMETRICALLY (same visual row
    # to the right first, then directly below -- see _pick_value_index),
    # not by blind next-in-stream adjacency.
    ordered_hits = sorted(hits, key=lambda h: (-h.score, h.token_index))
    claimed: set = set()
    assignments: List[Tuple[str, str]] = []  # (canonical, raw_value_text)

    for hit in ordered_hits:
        span_idxs = range(hit.token_index, hit.token_index + hit.span)
        if any(k in claimed for k in span_idxs):
            continue
        j = _pick_value_index(tokens, hit, hits, claimed)
        claimed.update(span_idxs)
        if j is not None:
            claimed.add(j)
            assignments.append((hit.canonical, tokens[j].text))

    for canonical, raw_value in assignments:
        if canonical in _IGNORE_LABELS:
            continue
        value = _postprocess_value(canonical, raw_value)
        if not value or value == "-":
            continue

        if canonical in _DATE_FIELDS and not _looks_like_date(value):
            # A date-type field grabbed a neighboring token that isn't a
            # date at all -- this happens when the real value is blank in
            # the UI (e.g. "Closed On" on a still-open ticket) and the
            # adjacency heuristic instead reaches for the next unrelated
            # token in the stream (seen in practice: "Closed On": "State").
            # Drop it and flag for manual review rather than storing junk.
            flags[canonical] = (
                f"Non-date value {value!r} rejected for a date field "
                "-- field is likely blank in the source screen; needs manual review"
            )
            continue

        if canonical in _CURRENCY_FIELDS and not _ANY_NUMBER_RE.search(value):
            # A currency field grabbed a token with no digits at all (e.g.
            # the "amount" fragment of a glued "Billed amount" label) --
            # the real value is missing/obscured; flag instead of storing.
            flags[canonical] = (
                f"Non-numeric value {value!r} rejected for a currency field "
                "-- field is likely blank/obscured in the source screen; needs manual review"
            )
            continue

        if canonical in _NAME_FIELDS and _STATUS_WORD_RE.match(value.strip()):
            # A workflow-status word (e.g. "Withdrawn") grabbed as a
            # person's name -- seen on the queue/list screen where a
            # stats-table header ("Escalated Resolved Withdrawn Total
            # Closed") sits right next to the real ticket rows. No real
            # name can equal one of these words, so drop and flag it
            # instead of letting it feed scenario User-inference.
            flags[canonical] = (
                f"Status word {value!r} rejected for a person-name field "
                "-- likely grabbed from an unrelated stats/status column; needs manual review"
            )
            continue

        if canonical in _BLOCKER_FIELDS and _ANY_NUMBER_RE.search(value):
            # Any number (whole or decimal) in Blocker is never legitimate
            # -- flag it unconditionally rather than storing it.
            flags[canonical] = (
                f"Numeric value {value!r} found in Blocker field -- needs manual review"
            )
            continue

        target = fields if canonical in all_variable_names else extra
        if canonical not in target:
            target[canonical] = value
        elif target[canonical] != value:
            existing = target[canonical]
            if isinstance(existing, list):
                if value not in existing:
                    existing.append(value)
            else:
                target[canonical] = [existing, value]

    # A label can occur more than once on a page (summary panel + detail
    # panel); when one occurrence paired with junk (rejected + flagged by
    # the shape guards above) but another paired with a real value, the
    # field is fine -- drop the now-moot rejection flag instead of
    # confusing the review sheet.
    for fname in list(flags):
        if fname in fields:
            del flags[fname]

    # ID-shape passes run on the full concatenated text and OVERRIDE the
    # fuzzy pass for these specific fields when a validated shape is
    # found, since a validated 8-char ID is more trustworthy than a
    # generic "next token" grab (garbled labels can point at the wrong
    # neighbor for these high-value identifiers).
    full_text = "\n".join(t.text for t in tokens)
    # The strict, shape-validated scans get the WHOLE raw dump (which also
    # carries [TABLE] cell text the bbox parser never tokenizes) so an ID
    # that OCR only landed inside a table cell is still found. The bbox-
    # token text is appended as a safety net for dumps with no raw text.
    scan_text = (doc.raw_text or "") + "\n" + full_text

    all_tids = find_all_ticket_ids(scan_text)
    if len(all_tids) == 1:
        fields["Ticket ID"] = all_tids[0]
    elif len(all_tids) > 1:
        # A queue/list screen (e.g. "My Open Tickets") can legitimately
        # show several tickets at once -- there's no reliable way to know
        # which one this particular screenshot is actually "about", so
        # don't guess. Leave the field blank and surface all of them in
        # Flags for a human to resolve. Also clear out anything the
        # generic fuzzy-label pass may have already grabbed for this
        # field, so it's genuinely blank rather than silently wrong.
        fields.pop("Ticket ID", None)
        flags["Ticket ID"] = (
            f"Multiple Ticket IDs detected on this page ({', '.join(all_tids)}) "
            "-- left blank to avoid ambiguity; needs manual review"
        )
    else:
        # Zero validated Ticket IDs on this screen. "Ticket ID" is a
        # strictly shape-validated field -- it must NEVER be populated
        # by the generic fuzzy label-adjacency pass above, only by this
        # regex scan. Clear out any raw junk that pass may have grabbed
        # (e.g. a garbled label token fuzzy-matching "Ticket ID" and
        # pulling in whatever unrelated text sits next to it) so a
        # screen with no real Ticket ID ends up genuinely blank.
        fields.pop("Ticket ID", None)

    all_cids = find_all_claim_ids(full_text)
    if "Claim ID" in all_variable_names:
        if len(all_cids) == 1:
            fields["Claim ID"] = all_cids[0]
        elif len(all_cids) > 1:
            fields.pop("Claim ID", None)
            flags["Claim ID"] = (
                f"Multiple Claim IDs detected on this page ({', '.join(all_cids)}) "
                "-- left blank to avoid ambiguity; needs manual review"
            )
        else:
            # Zero validated Claim IDs on this screen. Same principle as
            # Ticket ID above: "Claim ID" must only ever come from this
            # strict ID-shape scan, never from the generic fuzzy pass.
            # This is exactly the bug behind "System InterestAccrued"
            # showing up as a Claim ID value -- a garbled label token
            # fuzzy-matched "Claim ID" and grabbed that unrelated line of
            # text as its "value" with no shape check at all, and since
            # this override block previously had no branch for the
            # zero-found case, that raw junk was never cleared out.
            #
            # Before giving up: on Appian ticket screens the ID is
            # rendered BELOW a title ending in the word "claim", where
            # neither text scan above can see it (the stream-order scan
            # walks into the toolbar text sharing the title's row, and
            # the standalone-line fallback only accepts digit-first
            # IDs). That is still a shape-validated, anchored match --
            # not a fuzzy label grab -- so it belongs in this override.
            recovered = find_claim_id_under_title(tokens, full_text)
            if not recovered:
                # "Ocelot ID HSZCXE72" / "OcelotID GEWT7FMR": the Appian
                # Claim Info section names the claim's ID behind an
                # explicit Ocelot ID label -- keyword-anchored and
                # shape-validated, so it belongs in this override too.
                om = re.search(
                    r"\bOcelot\s*ID\b[:\s]*([A-Z0-9]{8})\b", scan_text
                )
                if om and CLAIM_ID_ALNUM_MIX_RE.match(om.group(1)) and not _contains_negated_word(om.group(1)):
                    recovered = om.group(1)
            if recovered:
                fields["Claim ID"] = recovered
            else:
                fields.pop("Claim ID", None)

    oid = find_osc_id(scan_text)
    if oid:
        if "OSC ID" in all_variable_names:
            fields["OSC ID"] = oid
        if "Member ID (OSC)" in all_variable_names:
            fields["Member ID (OSC)"] = oid
    else:
        # No validated OSC-prefixed ID found via the strict regex scan.
        # These fields must ONLY ever hold a value that genuinely starts
        # with "OSC" -- anything the generic fuzzy label-adjacency pass
        # may have already grabbed for them is unverified and gets
        # flagged for manual review instead of silently kept.
        for osc_field in ("OSC ID", "Member ID (OSC)"):
            if osc_field in fields:
                stray = fields.pop(osc_field)
                flags[osc_field] = (
                    f"Value {stray!r} does not start with 'OSC' -- needs manual review"
                )

    mid = find_member_id_literal(full_text)
    if "Member ID" in all_variable_names:
        if mid:
            fields["Member ID"] = mid
        else:
            # Zero validated literal "MemberID <id>" found on this screen.
            # Same principle already applied to Claim ID / Ticket ID / OSC
            # ID above: "Member ID" must only ever come from this strict,
            # shape-validated regex scan, never survive as an unvalidated
            # guess from the generic fuzzy label-adjacency pass. Without
            # this branch, a garbled/nearby label (e.g. a "Member info"
            # nav-tab token fuzzy-matching the "Member ID" label) could
            # leave its neighboring text (e.g. "Issues") sitting in this
            # field with no shape check ever applied to it.
            fields.pop("Member ID", None)

    if "Ticket State" in all_variable_names:
        pill_raw = ""
        for k, tok in enumerate(tokens):
            if STATUS_PILL_RE.match(tok.text.strip()):
                # Gather the following tokens too (up to 4): OCR often
                # splits the pill across tokens and truncates the status
                # word ("P3" + "In" + "Prog" + "(In" + "Review)").
                # _normalize_ticket_state anchors on the P<n>, matches the
                # words around a truncation ("Prog" -> In Progress), and
                # trims any trailing junk this over-collects.
                parts = [tok.text]
                for j in range(k + 1, min(k + 5, len(tokens))):
                    if STATUS_PILL_RE.match(tokens[j].text.strip()):
                        break
                    parts.append(tokens[j].text)
                pill_raw = _clean_value(" ".join(parts))
                break
        # Validate/repair the pill IMMEDIATELY: the value must carry both a
        # P1/P2/P3 priority code and a recognized status word ("In
        # Progress", "Resolved", ...) and is rewritten to its canonical
        # form ("P3 In Progress (In Review)") -- truncations, a dropped
        # closing bracket, and glued-on neighbor labels ("... Due") are all
        # repaired here instead of surviving into the workbook.
        if pill_raw:
            canonical, reason = _normalize_ticket_state(pill_raw)
            if canonical:
                fields["Ticket State"] = canonical
            else:
                flags["Ticket State"] = reason
        if "Ticket State" not in fields:
            # Whitespace-token fallback over the whole raw dump: a "P3"
            # token whose following tokens spell a recognized status word
            # (e.g. ... 'P3', 'In', 'Progress', '(In', 'Review)' ...).
            ws_toks = tokenize_text(scan_text)
            for k, wtok in enumerate(ws_toks):
                if not re.match(r"^P\d$", wtok.strip(), re.IGNORECASE):
                    continue
                tail = " ".join(ws_toks[k : k + 5])
                canonical, _reason = _normalize_ticket_state(tail)
                if canonical:
                    fields["Ticket State"] = canonical
                    flags.pop("Ticket State", None)
                    break

    # "Member State" -- the literal Appian "State" field (the member's US
    # state, e.g. "FL"), found as a label token immediately followed by a
    # validated 2-letter USPS code. Deliberately independent of the
    # ticket-status-pill logic above: the two fields share the English
    # word "State" in casual conversation but are unrelated data points
    # with completely different shapes, so conflating them would silently
    # corrupt one or the other.
    if "Member State" in all_variable_names:
        for k, tok in enumerate(tokens):
            if _normalize(tok.text) == "state" and k + 1 < len(tokens):
                candidate = tokens[k + 1].text.strip().upper()
                if candidate in _US_STATE_CODES:
                    fields["Member State"] = candidate
                    break

    # "Code" is resolved GEOMETRICALLY: the value token whose bounding box
    # is nearest the literal "Code" label (same visual row / to the right
    # preferred). When the geometric pickup succeeds it OVERRIDES whatever
    # the fuzzy label-adjacency pass grabbed -- that pass is exactly what
    # produced junk like ['NEWBORN', 'Count'] when a neighboring word sat
    # closer in stream order than the real value did on screen.
    # "Code" resolution order: (1) the [TABLE] grid -- nearest cell
    # horizontally on the label's own grid row; (2) nearest bounding box
    # in the [BOXES] pixel data; (3) plain regex over the whole dump --
    # the word(s) after a standalone "Code" label ("Code
    # WEAK_MEMBER_MATCH"); (4) whatever the fuzzy pass grabbed stays only
    # if all three miss.
    if "Code" in all_variable_names:
        geo_code = (
            find_code_in_table_cells(doc.raw_text or "")
            or find_code_nearest_bbox(tokens)
            or find_code_regex(scan_text)
        )
        if geo_code:
            fields["Code"] = geo_code
            flags.pop("Code", None)

    # "Received date" (Dataset A's "Claim Received date") -- bbox fallback,
    # fired ONLY when the regex/fuzzy pairing captured nothing date-shaped.
    # NOT gated on the app schema: Appian ticket screens also render a
    # literal "Received Date <date>" in their Claim Info section, and the
    # pickup is anchored on that literal label anyway (no label on the
    # page -> it returns None), so it is safe to attempt everywhere.
    existing = fields.get("Received date")
    existing_vals = existing if isinstance(existing, list) else [existing]
    has_date = any(
        isinstance(v, str) and _DATE_VALUE_RE.search(v) for v in existing_vals
    )
    if not has_date:
        geo_received = find_received_date_nearest_bbox(tokens)
        if geo_received:
            fields["Received date"] = geo_received
            flags.pop("Received date", None)

    # "Resolved By" mirrors "Updated By": per business rule, whoever is
    # recorded as last updating the ticket is also treated as its
    # resolver, regardless of whether the OCR text separately carried a
    # distinct "Resolved By" label/value.
    if "Updated By" in fields and "Resolved By" in all_variable_names:
        fields["Resolved By"] = fields["Updated By"]

    # "Due" collapses to the single greatest (latest) date found and is
    # re-rendered as dd/mm/yy, regardless of how many candidate values
    # the fuzzy pass collected or what format each was OCR'd in.
    if "Due" in fields:
        fields["Due"] = _resolve_due_date(fields["Due"])

    # "Created On" / "Updated On" / "Closed On" / "Resolved On" must
    # literally START with a date. Segments that do are kept (glued OCR
    # re-spaced, duplicate renderings of the same instant collapsed, a
    # year-less month/day completed with the current year); junk segments
    # are eliminated. A value with NO date shape anywhere in it is a
    # mis-grab from a neighboring field (e.g. SLA's "minus High$) (7
    # working days" landing in Closed On) and is ELIMINATED outright;
    # only a value that does carry a date somewhere -- just not in the
    # strict leading format -- is kept as OCR'd.
    for canonical in _STRICT_START_DATE_FIELDS:
        if canonical not in fields:
            continue
        cleaned = _clean_datetime_start(fields[canonical])
        if cleaned and not _ANY_DATE_SHAPE_RE.search(cleaned):
            fields.pop(canonical, None)
            flags.pop(canonical, None)
            continue
        if cleaned:
            fields[canonical] = cleaned
            if _DT_LEAD_FULL_RE.match(cleaned):
                flags.pop(canonical, None)

    # Pure DATE fields keep only their date substrings, deduplicated:
    # "06/19/2026 H112662924410001; 06/19/2026" -> "06/19/2026".
    for canonical in ("Received date", "Service Start Date", "Service End Date"):
        if canonical in fields:
            fields[canonical] = _clean_date_only(fields[canonical])

    # SLA carries ONLY the duration: "N working days" (plus an hours
    # figure when one is present). Queue names, stray timestamps, and
    # duplicated fragments are eliminated.
    if "SLA" in fields:
        cleaned_sla = _clean_sla(fields["SLA"])
        if cleaned_sla:
            fields["SLA"] = cleaned_sla
            flags.pop("SLA", None)

    # Business-rule value constraints: Claim State / Claim Type /
    # Submission Type are closed vocabularies (snapped to their canonical
    # value or flagged), and Skill / Code / Blocker have known mis-grab
    # values that must never be stored (see section 5c).
    _apply_value_constraints(fields, flags)

    # Per business rule, Submission Type is ALWAYS "Electronic" --
    # compulsory, never anything else. Set it unconditionally on every
    # screen that carries the field, overriding any OCR-garbled value
    # ("EDIfile Electronic Download Documents", etc.) the fuzzy pass may
    # have grabbed.
    if "Submission Type" in all_variable_names:
        fields["Submission Type"] = "Electronic"
        flags.pop("Submission Type", None)

    # Whole-page fallback for the closed-vocabulary claim fields: on a
    # Cortex claim screen the state/type wording is always somewhere in
    # the page text, even when the label->value pairing failed (garbled
    # label, value lost by OCR) -- recover it rather than leave it blank.
    page_norm = _normalize(full_text)
    on_claim_screen = "about this claim" in page_norm
    # The member claims-list view also shows the claim's type ("Claimtype
    # Professional (HCFA)") but never a Blocked/Audit state -- so it joins
    # the Claim Type fallback below but NOT the Claim State one.
    on_member_claims = "only medical claims" in page_norm or "cortex member" in page_norm
    if on_claim_screen:
        if "Claim State" in all_variable_names and "Claim State" not in fields:
            if re.search(r"\bblock(?:ed|ing)\b", full_text, re.IGNORECASE):
                fields["Claim State"] = "Blocked"
                flags.pop("Claim State", None)
            elif re.search(r"\baudit(?:ed)?\b", full_text, re.IGNORECASE):
                fields["Claim State"] = "Audit"
                flags.pop("Claim State", None)
        # A blocked claim always names its blocking type somewhere on the
        # page (e.g. "NO_MEMBER_MATCH") -- if the Blocker field ended up
        # empty (the pairing grabbed a button/section title that the
        # rejects above threw out), fill it from the blocking-type code
        # itself. Codes already stored as some other field's value (e.g.
        # Goal = "PROCESS_CLAIM") are never reused as the Blocker.
        if (
            "Blocker" in all_variable_names
            and fields.get("Claim State") == "Blocked"
            and "Blocker" not in fields
        ):
            existing_values = {str(v).strip() for v in fields.values() if isinstance(v, str)}
            for tok in tokens:
                cand = tok.text.strip()
                if not _BLOCKING_CODE_RE.match(cand):
                    continue
                if cand in existing_values or _normalize(cand) == "process claim":
                    continue
                fields["Blocker"] = cand
                flags.pop("Blocker", None)
                break
    # Claim Type is a CLOSED vocabulary: HCFA-Professional, UB - Facility,
    # or NEWBORN. Check which one the page wording carries (in that
    # priority order) and store exactly that -- an existing value already
    # in the vocabulary is kept, anything else the fuzzy pass grabbed is
    # replaced.
    if (
        (on_claim_screen or on_member_claims)
        and "Claim Type" in all_variable_names
        and fields.get("Claim Type") not in ("HCFA-Professional", "UB - Facility", "NEWBORN")
    ):
        if re.search(r"\bHCFA\b|\bprofessional\b", scan_text, re.IGNORECASE):
            fields["Claim Type"] = "HCFA-Professional"
            flags.pop("Claim Type", None)
        elif re.search(r"\bUB\b|\bfacility\b|\binstitut", scan_text, re.IGNORECASE):
            fields["Claim Type"] = "UB - Facility"
            flags.pop("Claim Type", None)
        elif re.search(r"\bnewborn\b", scan_text, re.IGNORECASE):
            fields["Claim Type"] = "NEWBORN"
            flags.pop("Claim Type", None)

    # Page-wide OCR-tolerant scan -- the LAST resort before Claim State is
    # left blank. The state word is somewhere in the formatted text even
    # when the label/value pairing could not read it (garbled label, value
    # split across OCR lines), and this also covers Appian screens, which
    # have no Claim State field of their own but do name the state in the
    # page text. Runs only when nothing above produced a value, so a
    # readable Claim State is never overridden.
    if "Claim State" not in fields:
        page_state, page_match, page_clean = find_claim_state_in_page(
            doc.raw_text or full_text
        )
        if page_state:
            fields["Claim State"] = page_state
            flags.pop("Claim State", None)
            if not page_clean:
                flags["Claim State"] = (
                    f"Recovered as {page_state!r} from a garbled page-wide "
                    f"match ({page_match!r}) -- no readable Claim State "
                    "label/value on this screen; verify"
                )

    # Flag (don't silently accept) a "Reason" value that's actually an
    # unfilled dropdown placeholder ("Select", "-- Select --", ...) --
    # it means no reason was actually chosen in the UI.
    reason_val = fields.get("Reason")
    reason_candidates = reason_val if isinstance(reason_val, list) else ([reason_val] if reason_val else [])
    for rv in reason_candidates:
        if isinstance(rv, str) and _SELECT_PLACEHOLDER_RE.match(rv.strip()):
            flags["Reason"] = f"Placeholder value detected ({rv!r}) -- dropdown appears unselected, needs manual review"
            break

    for fname, fval in list(fields.items()):
        if fname in flags:
            continue  # already flagged (e.g. Reason above)
        candidates = fval if isinstance(fval, list) else [fval]
        for cv in candidates:
            if isinstance(cv, str) and _GENERIC_PLACEHOLDER_RE.match(cv.strip()):
                flags[fname] = (
                    f"Placeholder/unfilled-input value detected ({cv!r}) -- "
                    "field appears empty in the source screen, needs manual review"
                )
                break

    return fields, extra, flags


def extract_all_documents(raw_text: str) -> List[DocumentResult]:
    schema = load_schema()
    all_variable_names = sorted({v for vars_ in schema.values() for v in vars_})

    # Fields that are only ever trusted from a dedicated, shape-validated
    # regex/positional scan (never from generic fuzzy label-adjacency) --
    # see _extract_fields_for_document below. Safe to expose for ANY app,
    # schema'd or not, because each one clears/overrides whatever the
    # fuzzy pass may have grabbed once its own dedicated check has run.
    _UNIVERSAL_ID_FIELDS = ["Claim ID", "OSC ID", "Member ID (OSC)", "Member ID"]

    results: List[DocumentResult] = []
    for doc in parse_documents(raw_text):
        app = classify_document(doc.tokens)
        if app == "VERINT":
            # Explicitly out of scope per the CSV ("N/A" / Knowledge
            # Management app) -- nothing to extract.
            results.append(DocumentResult(doc.index, app, {}, {}, {}))
            continue
        if app not in schema and app != "Unknown":
            # This document was identified with high confidence (a literal
            # app-name match, e.g. "Rosco" / "Elbow") as belonging to a
            # SPECIFIC product that simply isn't one of the schema'd
            # Appian/Cortex screens -- unlike a genuinely "Unknown" screen,
            # where the app itself is undetermined and searching every
            # known label at least has a chance of hitting the right one.
            # Falling back to the full Appian/Cortex label vocabulary here
            # (schema.get(app, all_variable_names)) doesn't broaden
            # coverage -- it fuzzy-matches unrelated labels (e.g. "Member
            # ID") against this app's own unrelated UI text (e.g. a
            # "Member info" nav tab) and produces confident-looking but
            # wrong values. Real screenshots confirmed this: a Rosco
            # screen with a "Member info / Issues / Policy details" nav
            # bar had "Member ID" fuzzy-match onto "Issues" -- a tab name,
            # not a value. Restrict the fuzzy pass to the universal
            # shape-validated ID fields only (still genuinely useful for
            # cross-app scenario linking) until this app gets its own
            # schema entries in _FALLBACK_SCHEMA_ROWS.
            variable_names = _UNIVERSAL_ID_FIELDS
        elif app in schema:
            # Search the app's own labels PLUS its product-family siblings'
            # (all Cortex views together, both Appian screens together) --
            # sibling misclassification otherwise silently blanks every
            # field that isn't in the wrongly-chosen sibling's short list.
            variable_names = _family_vocabulary(app, schema)
        else:  # genuinely "Unknown" -- search the full label vocabulary
            variable_names = all_variable_names
        fields, extra, flags = _extract_fields_for_document(doc, variable_names)
        results.append(DocumentResult(doc.index, app, fields, extra, flags))
    return results


# ===========================================================================
# 8. Convenience: read directly from a .docx OCR-metadata dump
# ===========================================================================


def read_docx_text(path: str) -> str:
    import docx  # local import so this module has no hard dependency on
    # python-docx unless this helper is actually used.

    d = docx.Document(path)
    return "\n".join(p.text for p in d.paragraphs)


def extract_from_docx(path: str) -> List[DocumentResult]:
    return extract_all_documents(read_docx_text(path))


def read_text_file(path: str) -> str:
    """Read a plain-text OCR dump (e.g. formatted_text.txt). Handles
    Windows-style \\r\\n line endings transparently."""
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def extract_from_text_file(path: str) -> List[DocumentResult]:
    return extract_all_documents(read_text_file(path))


def extract_from_file(path: str) -> List[DocumentResult]:
    """Convenience dispatcher: pick the right reader based on the file
    extension so callers don't need to know/care whether the OCR dump
    was saved as a .docx or a plain .txt file."""
    if path.lower().endswith(".docx"):
        return extract_from_docx(path)
    return extract_from_text_file(path)


# ===========================================================================
# 8b. Batch processing across many input files + Excel export
# ===========================================================================


def extract_from_files(paths: List[str]) -> List[Tuple[str, DocumentResult]]:
    """Loop extraction over a list of input files (any mix of .txt/.docx
    OCR dumps). Missing files are skipped (reported, not raised) so one
    bad path in a big batch doesn't abort the whole run. Returns a flat
    list of (source_filename, DocumentResult) -- one entry per document
    found inside each file, tagged with which file it came from."""
    all_results: List[Tuple[str, DocumentResult]] = []
    for path in paths:
        try:
            doc_results = extract_from_file(path)
        except FileNotFoundError:
            print(f"(skipping {path}: not found)")
            continue
        fname = os.path.basename(path)
        for r in doc_results:
            all_results.append((fname, r))
    return all_results


def results_to_excel(all_results: List[Tuple[str, DocumentResult]], output_path: str) -> str:
    """Write batch extraction results to a single .xlsx file with one
    row per document and one column per schema variable (so every
    extracted field lines up as its own Excel column), plus a few
    metadata/QA columns (source file, detected application, any extra
    fields found outside the schema, and any flags such as an unfilled
    'Reason' dropdown). Returns the output_path for convenience."""
    import pandas as pd
    from openpyxl import load_workbook
    from openpyxl.styles import Font

    schema = load_schema()
    all_variable_names = sorted({v for vars_ in schema.values() for v in vars_})

    rows = []
    for source_file, r in all_results:
        row: Dict[str, Any] = {
            "Source File": source_file,
            "Document Index": r.document_index,
            "Detected Application": r.detected_application,
        }
        for var in all_variable_names:
            val = r.fields.get(var, "")
            if isinstance(val, list):
                val = "; ".join(str(v) for v in val)
            row[var] = val
        row["Extra Fields"] = (
            "; ".join(f"{k}={v}" for k, v in r.extra_fields.items()) if r.extra_fields else ""
        )
        row["Flags"] = "; ".join(f"{k}: {v}" for k, v in r.flags.items()) if r.flags else ""
        rows.append(row)

    columns = ["Source File", "Document Index", "Detected Application"] + all_variable_names + [
        "Extra Fields",
        "Flags",
    ]
    df = pd.DataFrame(rows, columns=columns)
    df.to_excel(output_path, index=False, sheet_name="Extracted Fields")

    # Light formatting pass: professional font, bold/frozen header,
    # readable column widths.
    wb = load_workbook(output_path)
    ws = wb.active
    header_font = Font(name="Arial", bold=True)
    body_font = Font(name="Arial")
    for cell in ws[1]:
        cell.font = header_font
    for row_cells in ws.iter_rows(min_row=2):
        for cell in row_cells:
            cell.font = body_font
    ws.freeze_panes = "A2"
    for col_cells in ws.columns:
        longest = max((len(str(c.value)) if c.value is not None else 0) for c in col_cells)
        ws.column_dimensions[col_cells[0].column_letter].width = min(max(longest + 2, 10), 40)
    wb.save(output_path)

    return output_path


# ===========================================================================
# 8c. Scenario / user auto-linking + dashboard-ready multi-sheet workbook
#
# Design intent (per explicit sign-off): a "scenario" is a chain of
# screenshots connected by an EXACT shared identifier (Ticket ID, Claim
# ID, OSC ID, or the literal Appian Member ID) -- never a fuzzy/temporal
# guess. This is deliberately conservative: a screenshot that carries no
# linkable ID of its own (VERINT knowledge-base pages, popups with only a
# Reason/Sub status, an unfilled Member-match lookup) stays its own
# unlinked singleton rather than being silently attached to whichever
# scenario happens to be nearby in the file order. Built as a general
# many-rows -> many-scenarios union-find so the exact same function keeps
# working once multiple users/scenarios are batched together later --
# nothing here assumes "one user, one scenario" beyond the sample data.
# ===========================================================================

ID_LINK_FIELDS: List[str] = ["Ticket ID", "Claim ID", "OSC ID", "Member ID (OSC)", "Member ID"]
# A claim and its ticket are 1:1 (business rule), so these two identify THE
# claim -- rows sharing one are always the same scenario. The member-level
# IDs below identify the MEMBER, and one member can have many claims: a
# shared OSC/Member ID must never merge two different claims' screenshots
# into one scenario (real case: claims EFN1MBBZ and 30347TZ9 both belong to
# OSC75112973 -- linking them made one of the claims vanish from the
# one-row-per-scenario Dataset A).
_CLAIM_LEVEL_LINK_FIELDS = ("Ticket ID", "Claim ID")
_MEMBER_LEVEL_LINK_FIELDS = ("OSC ID", "Member ID (OSC)", "Member ID")
_USER_NAME_FIELDS: List[str] = ["Assignee", "Resolved By", "Updated By"]

# Wider net used ONLY as a fallback when the primary fields above yielded
# nothing -- these can carry service accounts ("svc.automat") so they never
# take precedence, but a real human name in them beats "Unknown".
_USER_NAME_FALLBACK_FIELDS: List[str] = ["Created By", "Submitter"]

# ---------------------------------------------------------------------------
# Known-user reference list (optional Excel).
#
# Point USER_REFERENCE_XLSX (env var, or call load_known_users() directly)
# at a previously validated workbook -- every human name found in its
# "User" / "Name" columns becomes a known user. Two things then happen:
#   1. A row whose User would come out "Unknown" is re-scanned: if any
#      known name appears anywhere in the scenario's extracted values
#      (Assignee, Updated By, ... even OCR-glued like "JessicaBrister"),
#      that name is used instead of "Unknown".
#   2. Glued/misspaced extracted names are canonicalized to the reference
#      spelling ("JessicaBrister" -> "Jessica Brister").
# ---------------------------------------------------------------------------

_KNOWN_USER_NAMES: List[str] = []
_USER_JUNK_NORMS = {"unknown", "", "nan", "none"}

# Capitalized UI/status words that can land in a User/Name column but are
# never part of a person's name -- any word from this set disqualifies the
# whole candidate ("Status Created Closed").
_NON_NAME_WORDS = {
    "status", "created", "closed", "resolved", "open", "denied", "approved",
    "pending", "start", "next", "unknown", "audit", "blocked", "paid",
    "escalated", "withdrawn", "paused", "reopened", "hold", "claim", "ticket",
}


def load_known_users(xlsx_path: str) -> List[str]:
    """Loads human names from the 'User' / 'Name' columns of every sheet of
    a reference workbook into the module-level known-user list."""
    global _KNOWN_USER_NAMES
    try:
        import pandas as pd
        names: List[str] = []
        for _sheet, df in pd.read_excel(xlsx_path, sheet_name=None).items():
            for col in df.columns:
                if str(col).strip().lower() not in ("user", "name"):
                    continue
                for v in df[col].dropna().astype(str):
                    v = re.sub(r"\s+", " ", v.strip())
                    norm = _normalize(v)
                    if norm in _USER_JUNK_NORMS:
                        continue
                    # A real person's name: 2-4 Capitalized words, letters
                    # only ("Hilary Redd", "Jessica Brister"). ALL-CAPS UI
                    # words ("START NEXT"), single status words ("Denied"),
                    # dates, and codes never qualify.
                    if not re.match(
                        r"^[A-Z][a-z][A-Za-z.'-]*(?: [A-Z][a-z][A-Za-z.'-]*){1,3}$", v
                    ):
                        continue
                    if any(w.lower() in _NON_NAME_WORDS for w in v.split()):
                        continue
                    if v not in names:
                        names.append(v)
        _KNOWN_USER_NAMES = names
    except Exception as exc:
        import logging
        logging.getLogger("field_extractor").warning(
            "Could not load known users from %s: %s", xlsx_path, exc
        )
    return _KNOWN_USER_NAMES


_user_ref = os.getenv("USER_REFERENCE_XLSX", "").strip()
if _user_ref:
    load_known_users(_user_ref)


_CAMEL_NAME_RE = re.compile(r"^([A-Z][a-z]+)((?:[A-Z][a-z]+)+)$")


def _canonical_user(raw: str) -> str:
    """Snaps an extracted name to its known-user spelling when it matches
    one ignoring spaces/case ('JessicaBrister' -> 'Jessica Brister'). When
    the name is NOT in the reference list, an OCR-glued CamelCase name is
    still split on its capital letters so the output reads as a real
    name."""
    raw = (raw or "").strip()
    squashed = re.sub(r"[^a-z0-9]", "", raw.lower())
    if not squashed:
        return raw
    for known in _KNOWN_USER_NAMES:
        if re.sub(r"[^a-z0-9]", "", known.lower()) == squashed:
            return known
    m = _CAMEL_NAME_RE.match(raw)
    if m:
        return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", raw)
    return raw


def _find_known_user_in_text(text: str) -> Optional[str]:
    """A known user's name appearing ANYWHERE in the text (spaces optional,
    case-insensitive) -- how the reference names get recovered from values
    the fuzzy pass filed under other fields."""
    squashed = re.sub(r"[^a-z0-9]", "", (text or "").lower())
    if not squashed:
        return None
    for known in _KNOWN_USER_NAMES:
        if re.sub(r"[^a-z0-9]", "", known.lower()) in squashed:
            return known
    return None


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


# An OSC ID's base is its first 8 digits; the optional trailing "-NN"
# member/dependent suffix is kept in the DISPLAYED value ("OSC7823944202")
# but stripped for LINKING, because the same member's OSC is rendered with
# the suffix on some screens and without it on others ("OSC78239442") --
# an exact-value join would wrongly split those screenshots into separate
# scenarios. Merging on the base is safe: member-level links are already
# guarded against joining two different Claim IDs in link_scenarios.
_OSC_LINK_BASE_RE = re.compile(r"^OSC(\d{8})\d{0,4}$", re.IGNORECASE)


def _claim_link_key(v: str) -> str:
    """O/0-insensitive JOIN KEY for a Claim ID. OCR flips O<->0 inside
    these alphanumeric IDs ('POCTXSM4' read as 'P0CTXSM4'), and an exact
    comparison then treats two spellings of the SAME claim as different
    claims -- which both blocks the member-level scenario link (the guard
    sees 'two claims' and refuses) and defeats identity reconciliation.
    Only ever used for comparing/grouping; displayed values are untouched."""
    return (v or "").strip().upper().replace("0", "O")


def _row_id_values(row: Dict[str, Any]) -> Dict[str, str]:
    """Flatten a row's ID-link fields to single string values (a field
    that came back as a list of OCR variants is skipped for linking
    purposes -- ambiguous multi-value fields are not a safe join key).
    OSC-prefixed values are reduced to their 8-digit base and Claim IDs
    to their O/0-insensitive key -- for linking only, never for display."""
    out = {}
    for f in ID_LINK_FIELDS:
        v = row.get(f)
        if isinstance(v, str) and v.strip():
            v = v.strip()
            m = _OSC_LINK_BASE_RE.match(v)
            if m:
                v = f"OSC{m.group(1)}"
            elif f == "Claim ID":
                v = _claim_link_key(v)
            out[f] = v
    return out


def link_scenarios(rows: List[Dict[str, Any]]) -> Tuple[List[int], Dict[str, List[int]]]:
    """Union rows that share an exact, non-empty identifier. Claim-level
    IDs (Ticket ID / Claim ID -- 1:1 with the claim) always link.
    Member-level IDs (OSC / Member ID) only link when unambiguous: a
    member can have MANY claims, so a shared member ID never merges rows
    that carry two DIFFERENT Claim IDs. Returns (scenario_index_per_row,
    value->row-indices index used to build the links) -- scenario_index is
    a 0-based id assigned in first-appearance order (stable/reproducible
    across runs)."""
    n = len(rows)
    uf = _UnionFind(n)
    value_to_rows: Dict[str, List[int]] = {}
    for i, row in enumerate(rows):
        for _field, value in _row_id_values(row).items():
            key = f"{_field}::{value}"
            value_to_rows.setdefault(key, []).append(i)

    def _claim_of(i: int) -> str:
        v = rows[i].get("Claim ID")
        # O/0-insensitive: two OCR spellings of the same claim must not
        # read as "two different claims" and block a member-level link.
        return _claim_link_key(v) if isinstance(v, str) and v.strip() else ""

    for key, idxs in value_to_rows.items():
        field = key.split("::", 1)[0]
        if field in _MEMBER_LEVEL_LINK_FIELDS:
            claims = {_claim_of(i) for i in idxs if _claim_of(i)}
            if len(claims) > 1:
                # Same member, different claims -- these are different
                # scenarios by definition; do not merge (and any claimless
                # row sharing only this member ID stays its own singleton,
                # since which claim it belongs to is genuinely ambiguous).
                continue
        for j in idxs[1:]:
            uf.union(idxs[0], j)

    root_to_scenario: Dict[int, int] = {}
    scenario_per_row: List[int] = []
    for i in range(n):
        root = uf.find(i)
        if root not in root_to_scenario:
            root_to_scenario[root] = len(root_to_scenario)
        scenario_per_row.append(root_to_scenario[root])
    return scenario_per_row, value_to_rows


def _infer_user(group_rows: List[Dict[str, Any]]) -> str:
    from collections import Counter

    counts = Counter()
    for row in group_rows:
        for f in _USER_NAME_FIELDS:
            v = row.get(f)
            if isinstance(v, str) and v.strip():
                counts[_canonical_user(re.sub(r"\s+", " ", v.strip()))] += 1
    if counts:
        return counts.most_common(1)[0][0]

    # Nothing in the primary name fields -- before settling on "Unknown",
    # look for a known user's name (from the reference workbook) anywhere
    # in this scenario's extracted values, then in the wider fallback
    # fields. Only real names ever replace Unknown; a screenshot batch with
    # genuinely no operator name still comes out "Unknown".
    if _KNOWN_USER_NAMES:
        for row in group_rows:
            for v in row.values():
                candidates = v if isinstance(v, list) else [v]
                for cv in candidates:
                    if isinstance(cv, str):
                        hit = _find_known_user_in_text(cv)
                        if hit:
                            return hit
    for row in group_rows:
        for f in _USER_NAME_FALLBACK_FIELDS:
            v = row.get(f)
            if isinstance(v, str) and v.strip():
                cand = _canonical_user(re.sub(r"\s+", " ", v.strip()))
                # Service accounts ("svc.automat") / non-name junk stay out.
                if re.match(r"^[A-Za-z][A-Za-z.'-]*(?: [A-Za-z][A-Za-z.'-]*){1,3}$", cand):
                    return cand
    return "Unknown"


def _cross_fill_ids(group_rows: List[Dict[str, Any]]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """For a linked scenario group, compute the single explicit value for
    each ID-link field (first non-empty value seen) plus a note of any
    genuine conflict (two DIFFERENT values for the same field inside one
    linked group -- should be rare/impossible by construction unless an
    OCR misread produced a spurious link; surfaced rather than silently
    picking one)."""
    resolved: Dict[str, str] = {}
    conflicts: Dict[str, str] = {}
    for f in ID_LINK_FIELDS:
        seen = []
        for row in group_rows:
            v = row.get(f)
            if isinstance(v, str) and v.strip() and v.strip() not in seen:
                seen.append(v.strip())
        if seen:
            resolved[f] = seen[0]
            if len(seen) > 1:
                # Several spellings of the SAME OSC id (with/without the
                # "-NN" member suffix) are expected across screenshots --
                # resolve to the most complete (longest) variant and only
                # report a conflict when the variants genuinely disagree
                # (different 8-digit bases).
                if f in _MEMBER_LEVEL_LINK_FIELDS:
                    bases = set()
                    for v in seen:
                        m = _OSC_LINK_BASE_RE.match(v)
                        bases.add(m.group(1) if m else v)
                    resolved[f] = max(seen, key=len)
                    if len(bases) > 1:
                        conflicts[f] = f"Multiple values linked in one scenario: {', '.join(seen)} -- needs manual review"
                else:
                    conflicts[f] = f"Multiple values linked in one scenario: {', '.join(seen)} -- needs manual review"
    return resolved, conflicts


def _build_raw_rows_and_scenarios(
    all_results: List[Tuple[str, DocumentResult]]
) -> Dict[str, Any]:
    """Shared first pass used by build_dashboard_workbook, build_dataset_a,
    and build_dataset_b: turns raw DocumentResults into per-screenshot row
    dicts, links them into scenarios (exact shared-ID union-find), and
    resolves the explicit cross-fill values for each scenario. Kept in one
    place so all three outputs are always built from the same linking
    logic rather than three separate copies that could drift apart."""
    schema = load_schema()
    all_variable_names = sorted({v for vars_ in schema.values() for v in vars_})

    raw_rows: List[Dict[str, Any]] = []
    for seq, (source_file, r) in enumerate(all_results, start=1):
        row: Dict[str, Any] = {
            "Sequence No.": seq,
            "Source File": source_file,
            "Document Index": r.document_index,
            "Detected Application": r.detected_application,
        }
        for var in all_variable_names:
            val = r.fields.get(var, "")
            if isinstance(val, list):
                val = "; ".join(str(v) for v in val)
            row[var] = val
        row["Extra Fields"] = "; ".join(f"{k}={v}" for k, v in r.extra_fields.items()) if r.extra_fields else ""
        row["Flags"] = "; ".join(f"{k}: {v}" for k, v in r.flags.items()) if r.flags else ""
        raw_rows.append(row)

    # Cross-row repair BEFORE anything is linked or written (section 8c-2):
    # fill the state/attribute gaps and overwrite the identity anomalies
    # (e.g. a lone "88MEMBER" Claim ID among many screenshots of the same
    # ticket) using the other screenshots of the same claim/ticket.
    # raw_rows itself is never mutated, so the Raw Extracted sheet still
    # shows exactly what OCR read.
    reconciled_rows: List[Dict[str, Any]] = (
        gap_fill_rows(raw_rows) if GAP_FILL_ENABLED else [dict(r) for r in raw_rows]
    )
    for _rrow in reconciled_rows:
        _rrow.setdefault("Gap-Fill Notes", "")

    scenario_idx, _links = link_scenarios(reconciled_rows)
    n_scenarios = (max(scenario_idx) + 1) if scenario_idx else 0
    groups: Dict[int, List[int]] = {}
    for i, s in enumerate(scenario_idx):
        groups.setdefault(s, []).append(i)

    scenario_id_label = {s: f"SCN-{s + 1:04d}" for s in range(n_scenarios)}
    user_by_scenario = {s: _infer_user([reconciled_rows[i] for i in idxs]) for s, idxs in groups.items()}
    resolved_by_scenario: Dict[int, Dict[str, str]] = {}
    conflicts_by_scenario: Dict[int, Dict[str, str]] = {}
    for s, idxs in groups.items():
        resolved_by_scenario[s], conflicts_by_scenario[s] = _cross_fill_ids([reconciled_rows[i] for i in idxs])

    for i, row in enumerate(raw_rows):
        s = scenario_idx[i]
        row["Scenario_ID"] = scenario_id_label[s]
        row["User"] = user_by_scenario[s]
        reconciled_rows[i]["Scenario_ID"] = scenario_id_label[s]
        reconciled_rows[i]["User"] = user_by_scenario[s]

    consolidated_rows: List[Dict[str, Any]] = []
    for i, row in enumerate(reconciled_rows):
        s = scenario_idx[i]
        crow = dict(row)
        filled_notes = []
        for f, v in resolved_by_scenario[s].items():
            if not str(crow.get(f, "")).strip() and v:
                crow[f] = v
                filled_notes.append(f)
        conflict_notes = list(conflicts_by_scenario[s].values())
        note_parts = []
        if filled_notes:
            note_parts.append("Cross-filled from scenario " + scenario_id_label[s] + ": " + ", ".join(filled_notes))
        if conflict_notes:
            note_parts.append(" | ".join(conflict_notes))
        crow["Cross-Fill Notes"] = " ; ".join(note_parts)
        consolidated_rows.append(crow)

    return {
        "all_variable_names": all_variable_names,
        "raw_rows": raw_rows,
        "reconciled_rows": reconciled_rows,
        "consolidated_rows": consolidated_rows,
        "scenario_idx": scenario_idx,
        "n_scenarios": n_scenarios,
        "groups": groups,
        "scenario_id_label": scenario_id_label,
        "user_by_scenario": user_by_scenario,
        "resolved_by_scenario": resolved_by_scenario,
        "conflicts_by_scenario": conflicts_by_scenario,
    }


\
from collections import Counter
from typing import Callable

# ===========================================================================
# 8c-2. Cross-row gap fill / identity reconciliation
#
# Runs on the per-screenshot rows AFTER extraction but BEFORE scenario
# linking and before any of the datasets/sheets are built, so every
# downstream output sees the repaired rows.
#
# Two problems it fixes, both visible in the real Dataset B output:
#
#   1. IDENTITY ANOMALIES. Screenshots of one and the same ticket are
#      1:1 with one and the same claim (existing business rule). So when
#      15 screenshots of HOLD-5892183 report Claim ID "3CH5QP7X" and one
#      reports "88MEMBER", the odd one out is an OCR misread, not a
#      second claim -- the majority value wins and the anomaly is
#      overwritten. Same in reverse: a Ticket ID with a digit slip is
#      repaired from the majority Ticket ID of its claim.
#
#   2. BLANK GAPS. A screenshot that happens not to show the status pill
#      (a popup, a scrolled-down view) previously left Claim Status /
#      Ticket Status empty even though the surrounding screenshots of the
#      same ticket clearly show it. Those blanks are filled from the
#      nearest screenshot of the SAME claim/ticket -- previous one first,
#      the following one if there is no previous.
#
# Every single change is recorded in a "Gap-Fill Notes" column, and the
# Raw Extracted sheet is left completely untouched, so nothing here is
# ever silent: the raw sheet always shows what OCR actually read.
# ===========================================================================

# Master switch -- set to False to write the workbook exactly as before.
GAP_FILL_ENABLED = True

# Second, separate switch for the CAPTURE-ORDER link (section 8c-2b below).
# Exact-ID linking alone strands a Cortex screenshot: Cortex screens never
# display a Ticket ID, so one whose Claim ID was misread matches nothing and
# becomes its own scenario, holding the only Claim State in the batch while
# the Appian screenshots of that same ticket hold the only Ticket State.
# Setting this to False restores strict exact-ID-only behaviour.
SEQUENCE_LINK_ENABLED = True

# Fields carried across screenshots by NEAREST NEIGHBOUR (previous, else
# next) rather than by consensus: they legitimately change over the life
# of a ticket, so a blank is filled from the closest screenshot in time,
# not from a majority vote.
_GAP_FILL_STICKY_FIELDS: Tuple[str, ...] = ("Ticket State", "Claim State")

# Fields that describe the claim/ticket itself and therefore cannot differ
# between two screenshots of it -- a blank is filled with the group's
# consensus value.
_GAP_FILL_STABLE_FIELDS: Tuple[str, ...] = (
    "Ticket Name", "Skill", "Code", "Claim Type", "Submission Type", "Blocker",
    "Goal", "Tenant", "Billing Type", "Stay type", "Provider Claim#",
    "Transmission ID", "Received date", "Service Start Date", "Service End Date",
    "Billed", "Allowed", "Pay provider", "Pay member", "Member responsibilty",
    "OSC ID", "Member ID", "Member ID (OSC)", "Member State", "SLA",
    "Created On", "Submitter", "Assignee",
)

# Deliberately NOT gap-filled (a blank here is real information, and
# copying a value across screenshots would fabricate history):
#   Updated On / Updated By / Resolved On / Resolved By / Closed On /
#   Due / Reason / Sub status.

_VALID_TICKET_ID_RE = re.compile(r"^HOLD-\d{5,8}$", re.IGNORECASE)


def _cell(row: Dict[str, Any], field: str) -> str:
    """A row value as a clean single string ("" when blank). A field that
    came out of extraction as several OCR variants joined with "; " is
    deliberately returned as-is -- it fails the ID validators below and is
    therefore treated as an anomaly to be repaired, not as a join key."""
    v = row.get(field, "")
    return v.strip() if isinstance(v, str) else ""


def _row_label(row: Dict[str, Any]) -> str:
    """Human-readable identifier for a row in a note. Every screenshot in
    this pipeline is stored as '<screenshot-folder>/formatted_text.txt', so
    the bare filename does not identify which one -- the sequence number
    does, and it lines up with the Sequence No. column on every sheet."""
    src = row.get("Source File") or "another screenshot"
    seq = row.get("Sequence No.")
    return f"#{seq} ({src})" if seq else str(src)


def _is_plausible_claim_id(value: Any) -> bool:
    """Same shape rules the extractor itself enforces: exactly 8 capital
    alphanumerics, at least one letter, and never a known workflow word
    (as a substring, so '88MEMBER' / 'MEMBER12' are rejected too)."""
    if not isinstance(value, str):
        return False
    s = value.strip().upper()
    if len(s) != 8 or not s.isalnum():
        return False
    if not any(c.isalpha() for c in s):
        return False
    if _contains_negated_word(s):
        return False
    return bool(CLAIM_ID_ALNUM_MIX_RE.match(s) or CLAIM_ID_ALL_LETTERS_RE.match(s))


def _is_plausible_ticket_id(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return bool(_VALID_TICKET_ID_RE.match(value.strip()))


def _group_rows_by_claim_identity(rows: List[Dict[str, Any]]) -> List[List[int]]:
    """Union rows that share an exact Ticket ID or an exact Claim ID.
    Only SHAPE-VALID ids are ever used as a join key -- an anomalous value
    ('88MEMBER') must never be able to glue two unrelated tickets
    together, it is only ever a repair target. Member-level ids (OSC /
    Member ID) are not used at all here: one member can own many claims.
    Returns groups of row indices, each in sequence order."""
    uf = _UnionFind(len(rows))
    first_seen: Dict[str, int] = {}
    for i, row in enumerate(rows):
        keys: List[str] = []
        tid = _cell(row, "Ticket ID")
        if _is_plausible_ticket_id(tid):
            keys.append("TICKET::" + tid.upper())
        cid = _cell(row, "Claim ID")
        if _is_plausible_claim_id(cid):
            # O/0-insensitive key: 'P0CTXSM4' groups (and later
            # reconciles) with 'POCTXSM4' instead of forming its own group.
            keys.append("CLAIM::" + _claim_link_key(cid))
        for key in keys:
            if key in first_seen:
                uf.union(first_seen[key], i)
            else:
                first_seen[key] = i
    groups: Dict[int, List[int]] = {}
    for i in range(len(rows)):
        groups.setdefault(uf.find(i), []).append(i)
    return [sorted(idxs) for idxs in groups.values()]


def _pick_consensus(
    values: List[str], validator: Callable[[Any], bool]
) -> Tuple[Optional[str], bool]:
    """Winning value for an identity field within one group, plus a tie
    flag. Ranking: shape-valid values beat invalid ones, then higher
    occurrence count wins, then first-seen order. A tie between two
    equally-supported valid values is NOT resolved -- it is reported so a
    human decides, rather than a coin flip silently rewriting an id."""
    if not values:
        return None, False
    counts = Counter(values)
    order = {v: n for n, v in enumerate(dict.fromkeys(values))}
    ranked = sorted(
        counts, key=lambda v: (not validator(v), -counts[v], order[v])
    )
    best = ranked[0]
    if len(ranked) > 1:
        runner_up = ranked[1]
        if (validator(best), counts[best]) == (validator(runner_up), counts[runner_up]):
            return best, True
    return best, False


def _reconcile_identity_field(
    rows: List[Dict[str, Any]],
    group: List[int],
    notes: List[List[str]],
    field: str,
    validator: Callable[[Any], bool],
) -> None:
    """Force one identity field (Claim ID / Ticket ID) to a single value
    across every screenshot of the same claim/ticket: fill the blanks,
    overwrite the OCR anomalies, flag a genuine tie."""
    values = [_cell(rows[i], field).upper() for i in group if _cell(rows[i], field)]
    winner, tie = _pick_consensus(values, validator)
    if winner is None:
        return  # nothing anywhere in the group to reconcile against
    distinct = list(dict.fromkeys(values))
    if tie:
        for i in group:
            notes[i].append(
                f"{field} conflict across this ticket/claim group "
                f"({', '.join(distinct)}) -- equally supported, left as "
                "extracted; needs manual review"
            )
        return
    if not validator(winner):
        # Every value in the group is malformed -- there is nothing sound
        # to copy, so leave them alone and say so (never invent a value).
        for i in group:
            cur = _cell(rows[i], field)
            if cur:
                notes[i].append(
                    f"{field} {cur!r} is not a valid {field} and no valid "
                    "value exists on any screenshot of this ticket/claim "
                    "-- needs manual review"
                )
        return
    for i in group:
        cur = _cell(rows[i], field)
        if cur.upper() == winner:
            continue
        support = f"{values.count(winner)} of {len(group)} screenshots"
        if not cur:
            notes[i].append(f"{field} was blank -- filled with {winner} ({support})")
        elif validator(cur):
            notes[i].append(
                f"{field} {cur!r} disagreed with the rest of this ticket/claim "
                f"-- replaced with {winner} ({support}); a claim and its ticket "
                "are 1:1, so the minority value is an OCR variant"
            )
        else:
            notes[i].append(
                f"{field} {cur!r} is not a valid {field} -- replaced with "
                f"{winner} ({support})"
            )
        rows[i][field] = winner


# ===========================================================================
# 8c-2c. Strict shape rules for Ticket Status and Claim Status
#
# Both fields are validated and rewritten to a canonical form BEFORE any
# gap fill runs, so a malformed value can never be copied onto another
# screenshot of the same ticket.
#
# Ticket Status must carry BOTH a priority code (P1/P2/P3/...) AND a
# recognized status word (In Progress / Resolved / ...). OCR routinely
# truncates the word ("P1 In Progre"), drops the closing bracket
# ("P1 Resolved(Suppressed") and glues the next label on the end
# ("P3 Resolved (Suppressed) Assignee") -- all three are repaired to
# "P1 In Progress" / "P1 Resolved (Suppressed)" / "P3 Resolved
# (Suppressed)". A value missing either half is not a status pill at all:
# it is blanked and reported, which also lets it be refilled from another
# screenshot of the same ticket.
#
# Claim Status is only ever Blocked or Audit (existing business rule).
# Anything else is reported, and a claim that ended up with no status on
# ANY of its screenshots is defaulted to Blocked -- applied last, so a
# real Audit captured anywhere in the claim always wins over the default.
# ===========================================================================

_TICKET_STATE_PRIORITY_RE = re.compile(r"\bP\s*(\d)(?!\d)", re.IGNORECASE)
_TICKET_STATE_PAREN_RE = re.compile(r"\(([^)]*)\)?")

# canonical status word -> the OCR spellings/truncations seen for it
_TICKET_STATE_WORDS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("In Progress", ("in progress", "inprogress", "in progres", "in progre",
                     "in prog", "progress", "progres", "progre", "prog")),
    ("Resolved", ("resolved", "resolvd", "resolve", "resohved", "resoved", "resolv")),
    ("Closed", ("closed", "clesed", "close")),
    ("Escalated", ("escalated", "escalate", "escalatd")),
    ("Withdrawn", ("withdrawn", "withdrew", "withdraw")),
    ("Paused", ("paused", "pause")),
    ("Pending", ("pending", "pendng")),
    ("Open", ("open",)),
)

# canonical sub-status (the bracketed part) -> OCR spellings
_TICKET_SUB_STATE_WORDS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("In Review", ("in review", "inreview", "in revew", "in reviw", "review")),
    ("Suppressed", ("suppressed", "supressed", "suppresed", "suppress")),
    ("Withdrawn", ("withdrawn", "withdrew")),
    ("Escalated", ("escalated",)),
    ("Reopened", ("reopened", "re opened")),
    ("On Hold", ("on hold", "onhold")),
)

_CLAIM_STATE_WORDS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("Blocked", ("blocked", "block", "blocking", "blockd", "blocke", "blockng")),
    ("Audit", ("audit", "audited", "audlt", "aud t")),
)

# A claim with no status captured on any of its screenshots falls back to
# this (per business rule: the claim would not be in the queue otherwise).
_CLAIM_STATE_DEFAULT = "Blocked"

# Require the priority code as well as the status word. Every ticket pill in
# the source screens renders as "P<n> <Status>", so demanding both is what
# keeps a bare neighbouring word ("Assignee", "State") from ever being stored
# as a status. Set to False if a screen legitimately shows a status with no
# priority code -- then a lone recognized status word is accepted on its own.
TICKET_STATE_REQUIRE_PRIORITY = True


def _match_canonical_word(
    text: str, vocabulary: Tuple[Tuple[str, Tuple[str, ...]], ...]
) -> Optional[str]:
    """Best canonical word for an OCR fragment. A truncation is accepted as
    a PREFIX of the real word ("in progre" -> "In Progress"), but only from
    4 characters up, so a stray "in"/"re" can never decide a status."""
    norm = _normalize(text)
    if not norm:
        return None
    best: Optional[str] = None
    best_score = 0.0
    for canonical, variants in vocabulary:
        for variant in variants:
            score = 0.0
            if norm == variant:
                score = 1.0
            elif norm.startswith(variant):
                score = 0.95
            elif len(norm) >= 4 and variant.startswith(norm):
                score = 0.9
            else:
                score = difflib.SequenceMatcher(None, norm, variant).ratio()
                if score < 0.82:
                    score = 0.0
            if score > best_score:
                best, best_score = canonical, score
    return best


def _normalize_ticket_state(value: str) -> Tuple[str, str]:
    """Returns (canonical_value, rejection_reason). One of the two is always
    empty: a value either validates to "P<n> Status" / "P<n> Status (Sub)",
    or it is rejected with the reason why."""
    raw = (value or "").strip()
    if not raw:
        return "", ""
    prio = _TICKET_STATE_PRIORITY_RE.search(raw)
    if not prio:
        if TICKET_STATE_REQUIRE_PRIORITY:
            return "", (
                f"{raw!r} carries no P1/P2/P3 priority code, so it is not a "
                "ticket status pill"
            )
        bare = _match_canonical_word(raw, _TICKET_STATE_WORDS)
        if bare:
            return bare, ""
        return "", (
            f"{raw!r} carries no recognized status word (In Progress / "
            "Resolved / Closed / Escalated / ...)"
        )
    rest = raw[prio.end():]
    paren = _TICKET_STATE_PAREN_RE.search(rest)
    main = rest[:paren.start()] if paren else rest
    status = _match_canonical_word(main, _TICKET_STATE_WORDS)
    if not status:
        # The status word occasionally lands inside the bracket instead.
        status = _match_canonical_word(rest, _TICKET_STATE_WORDS)
    if not status:
        return "", (
            f"{raw!r} carries no recognized status word (In Progress / "
            "Resolved / Closed / Escalated / ...)"
        )
    out = f"P{prio.group(1)} {status}"
    if paren:
        inner = (paren.group(1) or "").strip()
        if inner:
            sub = _match_canonical_word(inner, _TICKET_SUB_STATE_WORDS)
            # An unrecognized bracket is kept verbatim rather than thrown
            # away -- but anything glued on AFTER the bracket is dropped.
            out += f" ({sub or inner})"
    return out, ""


def _normalize_claim_state(value: str) -> Tuple[str, str]:
    raw = (value or "").strip()
    if not raw:
        return "", ""
    canonical = _match_canonical_word(raw, _CLAIM_STATE_WORDS)
    if canonical:
        return canonical, ""
    return "", (
        f"{raw!r} is not a recognized Claim Status (only Blocked or Audit "
        "are possible)"
    )


def _normalize_status_fields(
    rows: List[Dict[str, Any]], notes: List[List[str]]
) -> None:
    """Validate/rewrite Ticket State and Claim State on every row before any
    cross-row copying happens. A value that fails its shape rule is blanked
    (and reported), so the gap fill can refill it from a screenshot of the
    same ticket that captured it properly."""
    for i, row in enumerate(rows):
        for field, normalizer in (
            ("Ticket State", _normalize_ticket_state),
            ("Claim State", _normalize_claim_state),
        ):
            if field not in row:
                continue
            current = _cell(row, field)
            if not current:
                continue
            canonical, reason = normalizer(current)
            if canonical and canonical != current:
                row[field] = canonical
                notes[i].append(
                    f"{field} {current!r} rewritten to {canonical!r} "
                    "(strict status-shape rule)"
                )
            elif not canonical:
                row[field] = ""
                notes[i].append(
                    f"{field} rejected: {reason} -- cleared so it can be "
                    "refilled from another screenshot of the same ticket; "
                    "needs manual review"
                )


def _default_claim_state(
    rows: List[Dict[str, Any]], notes: List[List[str]]
) -> None:
    """Last resort for a claim whose status was captured on none of its
    screenshots: default to Blocked. Runs after every fill, so any real
    Blocked/Audit value found anywhere in the claim takes precedence."""
    for i, row in enumerate(rows):
        if "Claim State" not in row or _cell(row, "Claim State"):
            continue
        if not _cell(row, "Claim ID"):
            continue  # no claim on this screenshot -> no claim status to set
        row["Claim State"] = _CLAIM_STATE_DEFAULT
        notes[i].append(
            f"Claim State was blank on every screenshot of this claim -- "
            f"defaulted to {_CLAIM_STATE_DEFAULT!r} per business rule; verify"
        )


def _fill_sticky_fields(
    rows: List[Dict[str, Any]], group: List[int], notes: List[List[str]]
) -> None:
    """Fill a blank Ticket State / Claim State from the nearest screenshot of
    the same claim/ticket: the previous one first, the following one if there
    is no previous. Never overwrites a value that is already there, so a real
    state change (Blocked -> Audit) is preserved exactly as captured."""
    if len(group) < 2:
        return
    for field in _GAP_FILL_STICKY_FIELDS:
        if not any(field in rows[i] for i in group):
            continue
        source: Optional[int] = None
        for i in group:  # forward pass -- carry the last seen value down
            if _cell(rows[i], field):
                source = i
            elif source is not None:
                rows[i][field] = _cell(rows[source], field)
                notes[i].append(
                    f"{field} was blank -- carried forward from screenshot "
                    f"{_row_label(rows[source])} of the same ticket/claim"
                )
        source = None
        for i in reversed(group):  # backward pass -- for leading blanks
            if _cell(rows[i], field):
                source = i
            elif source is not None:
                rows[i][field] = _cell(rows[source], field)
                notes[i].append(
                    f"{field} was blank -- carried back from screenshot "
                    f"{_row_label(rows[source])} of the same ticket/claim"
                )


def _fill_stable_fields(
    rows: List[Dict[str, Any]], group: List[int], notes: List[List[str]]
) -> None:
    """Fill blanks in the fields that cannot differ between two screenshots of
    one claim/ticket, using the group's consensus value."""
    if len(group) < 2:
        return
    for field in _GAP_FILL_STABLE_FIELDS:
        present = [i for i in group if field in rows[i]]
        if not present:
            continue
        values = [_cell(rows[i], field) for i in present if _cell(rows[i], field)]
        if not values:
            continue
        winner, tie = _pick_consensus(values, lambda _v: True)
        if tie or winner is None:
            continue  # ambiguous -- leave the blanks blank
        for i in present:
            if _cell(rows[i], field):
                continue
            rows[i][field] = winner
            notes[i].append(
                f"{field} was blank -- filled with {winner!r} from the "
                "other screenshots of this ticket/claim"
            )


# ===========================================================================
# 8c-2b. Capture-order link for Cortex screenshots with no Ticket ID
#
# Cortex screens never display a Ticket ID (see the schema: every Cortex view
# carries Claim State but no Ticket ID, and every Appian view the reverse), so
# a Cortex screenshot can only reach its ticket through a matching Claim ID.
# When that Claim ID is misread, the screenshot strands: it becomes its own
# scenario holding the only Claim State in the batch, while the Appian
# screenshots of the very same ticket hold the only Ticket State -- each side
# missing exactly what the other has, and no shared key to join them.
#
# The one remaining signal is capture order. A claim-bearing screenshot with
# no Ticket ID that sits inside (or directly beside) one ticket's run of
# screenshots is attached to that ticket. This is EXPLICITLY a positional
# inference, not an ID match, so:
#
#   * it applies only when both sides agree, or only one side exists -- if the
#     nearest ticket screenshot before it and after it belong to DIFFERENT
#     tickets, the link is genuinely a coin flip and is reported instead;
#   * a screenshot with no Claim ID either (a VERINT page, a bare popup) is
#     never attached -- there is nothing to corroborate the position with;
#   * a valid Claim ID is NEVER overwritten across a positional link. If the
#     two sides disagree on the Claim ID, both values are kept and the
#     disagreement is reported: one of them is probably a misread, but
#     position alone cannot say which.
# ===========================================================================


def _sequence_link_orphan_groups(
    rows: List[Dict[str, Any]], notes: List[List[str]]
) -> None:
    """Attach each claim-bearing group that holds no Ticket ID at all to the
    ticket group it sits inside/next to in capture order, then run the same
    gap fills across the combined set. Mutates rows/notes in place."""
    groups = _group_rows_by_claim_identity(rows)
    anchored: List[List[int]] = []
    orphans: List[List[int]] = []
    for g in groups:
        if any(_is_plausible_ticket_id(_cell(rows[i], "Ticket ID")) for i in g):
            anchored.append(g)
        elif any(_is_plausible_claim_id(_cell(rows[i], "Claim ID")) for i in g):
            orphans.append(g)
    if not anchored or not orphans:
        return

    owner: Dict[int, int] = {}
    for gi, g in enumerate(anchored):
        for i in g:
            owner[i] = gi

    def _nearest_anchored(start: int, step: int) -> Optional[int]:
        """Walk outward in capture order to the nearest screenshot that does
        belong to a ticket, skipping the unanchored ones in between."""
        j = start
        while 0 <= j < len(rows):
            if j in owner:
                return j
            j += step
        return None

    for orphan in orphans:
        lo, hi = min(orphan), max(orphan)
        before = _nearest_anchored(lo - 1, -1)
        after = _nearest_anchored(hi + 1, 1)
        gi_before = owner[before] if before is not None else None
        gi_after = owner[after] if after is not None else None

        if gi_before is not None and gi_after is not None and gi_before != gi_after:
            t_before = _ticket_of_group(rows, anchored[gi_before])
            t_after = _ticket_of_group(rows, anchored[gi_after])
            for i in orphan:
                notes[i].append(
                    "Not linked by capture order: this screen shows no Ticket ID "
                    f"and sits between two different tickets ({t_before} before, "
                    f"{t_after} after) -- which one it belongs to is a coin flip; "
                    "needs manual review"
                )
            continue

        gi = gi_before if gi_before is not None else gi_after
        if gi is None:
            continue
        host = anchored[gi]
        ticket = _ticket_of_group(rows, host)
        if not ticket:
            continue

        if gi_before is not None and gi_after is not None:
            evidence = (
                f"sits between screenshots {_row_label(rows[before])} and "
                f"{_row_label(rows[after])} of {ticket}"
            )
        elif gi_before is not None:
            evidence = (
                f"follows screenshot {_row_label(rows[before])} of {ticket}, "
                "with no later ticket screenshot in the batch"
            )
        else:
            evidence = (
                f"precedes screenshot {_row_label(rows[after])} of {ticket}, "
                "with no earlier ticket screenshot in the batch"
            )

        for i in orphan:
            if not _cell(rows[i], "Ticket ID"):
                rows[i]["Ticket ID"] = ticket
                notes[i].append(
                    f"Ticket ID {ticket} attached by CAPTURE ORDER, not by a "
                    f"matching ID ({evidence}) -- Cortex screens never display "
                    "a Ticket ID; verify if the claim looks wrong"
                )

        combined = sorted(host + orphan)
        claims = []
        for i in combined:
            c = _cell(rows[i], "Claim ID").upper()
            if _is_plausible_claim_id(c) and c not in claims:
                claims.append(c)
        if len(claims) > 1:
            for i in combined:
                notes[i].append(
                    "Capture-order link joined screenshots reporting different "
                    f"Claim IDs ({', '.join(claims)}) -- both kept as extracted "
                    "(one is probably an OCR misread, but position cannot say "
                    "which); needs manual review"
                )

        _fill_sticky_fields(rows, combined, notes)
        _fill_stable_fields(rows, combined, notes)


def _ticket_of_group(rows: List[Dict[str, Any]], group: List[int]) -> str:
    """The single Ticket ID of an already-reconciled group ("" if none)."""
    for i in group:
        t = _cell(rows[i], "Ticket ID")
        if _is_plausible_ticket_id(t):
            return t.upper()
    return ""


def gap_fill_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a repaired COPY of the per-screenshot rows (the originals are
    never mutated, so the Raw Extracted sheet keeps showing exactly what
    OCR read). Rows are grouped by exact shared Ticket ID / Claim ID, then
    within each group: identity fields are reconciled to one value, sticky
    state fields are filled from the nearest neighbouring screenshot, and
    stable claim attributes are filled from the group consensus. Finally, if
    SEQUENCE_LINK_ENABLED, a claim-bearing group that holds no Ticket ID at
    all is attached to the ticket it sits inside in capture order (section
    8c-2b). Every change lands in the row's 'Gap-Fill Notes'."""
    out: List[Dict[str, Any]] = [dict(r) for r in rows]
    notes: List[List[str]] = [[] for _ in out]

    # Strict status shapes FIRST, so a malformed pill is never copied onto
    # another screenshot by the fills below (section 8c-2c).
    _normalize_status_fields(out, notes)

    for group in _group_rows_by_claim_identity(out):
        valid_tickets = {
            _cell(out[i], "Ticket ID").upper()
            for i in group
            if _is_plausible_ticket_id(_cell(out[i], "Ticket ID"))
        }
        valid_claims = {
            _cell(out[i], "Claim ID").upper()
            for i in group
            if _is_plausible_claim_id(_cell(out[i], "Claim ID"))
        }
        if len(valid_tickets) > 1 and len(valid_claims) > 1:
            # Several distinct valid tickets AND several distinct valid
            # claims chained into one group -- that is not one claim being
            # misread, it is a bad merge (most likely one shared id is
            # itself an OCR error). Report it and change nothing.
            for i in group:
                notes[i].append(
                    "Gap fill skipped: this group chained "
                    f"{len(valid_tickets)} Ticket IDs and {len(valid_claims)} "
                    "Claim IDs together, which cannot be one claim "
                    "-- needs manual review"
                )
            continue

        # --- 1. identity reconciliation -------------------------------------
        _reconcile_identity_field(out, group, notes, "Claim ID", _is_plausible_claim_id)
        _reconcile_identity_field(out, group, notes, "Ticket ID", _is_plausible_ticket_id)

        if len(group) < 2:
            continue  # a lone screenshot has no neighbour to fill from

        _fill_sticky_fields(out, group, notes)
        _fill_stable_fields(out, group, notes)

    if SEQUENCE_LINK_ENABLED:
        _sequence_link_orphan_groups(out, notes)

    # Blocked-by-default runs last: a real Blocked/Audit found anywhere in
    # the claim has already been copied by the fills above and wins.
    _default_claim_state(out, notes)

    for i, row in enumerate(out):
        row["Gap-Fill Notes"] = " ; ".join(notes[i])
    return out



# ===========================================================================
# 8d. Dataset A -- unique-Claims master table (one row per Claim ID)
#
# Per sign-off: a Claim and its Ticket are 1:1 (each is genuinely unique),
# so unlike the general scenario model this table never needs to list
# multiple tickets under one claim -- a scenario either has one resolved
# Claim ID (and thus exactly one Ticket ID) or it has none and is simply
# excluded from this table (e.g. a lone VERINT page, an unfilled Member-
# match lookup).
# ===========================================================================

_TICKET_DATE_FIELDS = ["Created On", "Updated On", "Due", "Resolved On", "Closed On"]
_CLAIM_DATE_FIELDS = ["Received date", "Service Start Date", "Service End Date"]


def build_dataset_a(all_results: List[Tuple[str, DocumentResult]]) -> "pandas.DataFrame":
    import pandas as pd

    ctx = _build_raw_rows_and_scenarios(all_results)
    raw_rows, groups = ctx["reconciled_rows"], ctx["groups"]
    resolved_by_scenario, scenario_id_label = ctx["resolved_by_scenario"], ctx["scenario_id_label"]
    user_by_scenario = ctx["user_by_scenario"]

    out_rows: List[Dict[str, Any]] = []
    for s, idxs in groups.items():
        claim_id = resolved_by_scenario[s].get("Claim ID", "")
        if not claim_id:
            continue  # Dataset A is Claims-only -- no claim, no row.
        group_rows = [raw_rows[i] for i in idxs]

        def _first_nonempty(field: str) -> str:
            for gr in group_rows:
                v = gr.get(field, "")
                if v:
                    return v
            return ""

        def _last_nonempty(field: str) -> str:
            val = ""
            for gr in group_rows:
                v = gr.get(field, "")
                if v:
                    val = v
            return val

        row = {
            "CLAIM": claim_id,
            "Ticket No": resolved_by_scenario[s].get("Ticket ID", ""),
            "Skill/Code": _first_nonempty("Code") or _first_nonempty("Skill"),
            "Claim State": _last_nonempty("Claim State"),
            "Claim Type": _first_nonempty("Claim Type"),
            "Submission Type": _first_nonempty("Submission Type"),
            "OSC ID": resolved_by_scenario[s].get("OSC ID", ""),
        }
        for f in _TICKET_DATE_FIELDS:
            row[f"Ticket {f}"] = _first_nonempty(f)
        for f in _CLAIM_DATE_FIELDS:
            row[f"Claim {f}"] = _first_nonempty(f)
        # De-glue / canonicalize exactly like the User column: an OCR-glued
        # "MelissaRippon" renders as "Melissa Rippon".
        row["Name"] = _canonical_user(_first_nonempty("Assignee"))
        row["SLA"] = _first_nonempty("SLA")
        row["Scenario_ID"] = scenario_id_label[s]
        row["User"] = user_by_scenario[s]
        out_rows.append(row)

    columns = (
        ["CLAIM", "Ticket No", "Skill/Code", "Claim State", "Claim Type",
         "Submission Type", "OSC ID"]
        + [f"Ticket {f}" for f in _TICKET_DATE_FIELDS]
        + [f"Claim {f}" for f in _CLAIM_DATE_FIELDS]
        + ["Name", "SLA", "Scenario_ID", "User"]
    )
    return pd.DataFrame(out_rows, columns=columns)


# ===========================================================================
# 8e. Dataset B -- per-screenshot Claim/Ticket status & state change log
# ===========================================================================


def build_dataset_b(all_results: List[Tuple[str, DocumentResult]]) -> "pandas.DataFrame":
    import pandas as pd

    ctx = _build_raw_rows_and_scenarios(all_results)
    consolidated_rows = ctx["consolidated_rows"]

    last_ticket_state: Dict[str, str] = {}
    last_claim_state: Dict[str, str] = {}
    out_rows: List[Dict[str, Any]] = []
    for row in consolidated_rows:
        claim_id = row.get("Claim ID", "")
        if not claim_id:
            continue  # Dataset B is Claims-only -- no claim, no row.

        ticket_state = row.get("Ticket State", "") or last_ticket_state.get(claim_id, "")
        claim_state = row.get("Claim State", "") or last_claim_state.get(claim_id, "")
        if row.get("Ticket State", ""):
            last_ticket_state[claim_id] = row["Ticket State"]
        if row.get("Claim State", ""):
            last_claim_state[claim_id] = row["Claim State"]

        out_rows.append({
            "Screenshot": row["Source File"],
            "Claim ID": claim_id,
            "Ticket ID": row.get("Ticket ID", ""),
            "Claim Status": claim_state,
            "Ticket Status": ticket_state,
        })

    columns = ["Screenshot", "Claim ID", "Ticket ID", "Claim Status", "Ticket Status"]
    return pd.DataFrame(out_rows, columns=columns)


def build_dashboard_workbook(all_results: List[Tuple[str, DocumentResult]], output_path: str) -> str:
    """Writes a 6-sheet, multi-user/multi-scenario-ready workbook:
      1. Dataset A            -- one row per unique Claim (the Claims master table).
      2. Dataset B            -- one row per screenshot that carries a Claim, with Ticket/Claim state-change tracking.
      3. Scenario Summary     -- one row per scenario: user, linked IDs, timeline, final states, flags.
      4. Consolidated         -- per-screenshot rows, ID fields cross-filled ONLY within an auto-linked scenario.
      5. Raw Extracted        -- exactly what OCR+schema extraction found, per screenshot, untouched.
      6. Data Dictionary      -- field definitions, linking rule, and known OCR/extraction limitations.
    """
    import pandas as pd
    from openpyxl import load_workbook
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter

    ctx = _build_raw_rows_and_scenarios(all_results)
    all_variable_names = ctx["all_variable_names"]
    raw_rows, consolidated_rows = ctx["raw_rows"], ctx["consolidated_rows"]
    reconciled_rows = ctx["reconciled_rows"]
    groups, n_scenarios = ctx["groups"], ctx["n_scenarios"]
    scenario_id_label, user_by_scenario = ctx["scenario_id_label"], ctx["user_by_scenario"]
    resolved_by_scenario, conflicts_by_scenario = ctx["resolved_by_scenario"], ctx["conflicts_by_scenario"]

    # ---- Scenario summary rows -------------------------------------------------
    summary_rows: List[Dict[str, Any]] = []
    for s in range(n_scenarios):
        idxs = groups[s]
        group_rows = [reconciled_rows[i] for i in idxs]
        resolved = resolved_by_scenario[s]
        ticket_states = [gr.get("Ticket State", "") for gr in group_rows if gr.get("Ticket State", "")]
        claim_states = []
        for gr in group_rows:
            cs = gr.get("Claim State", "")
            if cs and cs not in claim_states:
                claim_states.append(cs)
        all_flags = [gr.get("Flags", "") for gr in group_rows if gr.get("Flags", "")]
        apps_seen = sorted({gr.get("Detected Application", "") for gr in group_rows if gr.get("Detected Application")})
        summary_rows.append({
            "Scenario_ID": scenario_id_label[s],
            "User": user_by_scenario[s],
            "Screenshot Count": len(group_rows),
            "Source Files": "; ".join(gr["Source File"] for gr in group_rows),
            "Applications Involved": "; ".join(apps_seen),
            "Ticket ID": resolved.get("Ticket ID", ""),
            "Claim ID": resolved.get("Claim ID", ""),
            "OSC ID": resolved.get("OSC ID", ""),
            "Member ID (OSC)": resolved.get("Member ID (OSC)", ""),
            "Member ID": resolved.get("Member ID", ""),
            "Ticket State (last seen)": ticket_states[-1] if ticket_states else "",
            "Ticket State History": " -> ".join(dict.fromkeys(ticket_states)) if ticket_states else "",
            "Claim State History": " -> ".join(claim_states) if claim_states else "",
            "Rows With Flags": len(all_flags),
            "Flag Details": " ; ".join(all_flags),
            "ID Conflicts": " ; ".join(conflicts_by_scenario[s].values()),
            "Gap-Fill Details": " ; ".join(
                gr.get("Gap-Fill Notes", "") for gr in group_rows if gr.get("Gap-Fill Notes")
            ),
        })

    # ---- Data dictionary --------------------------------------------------------
    dictionary_rows = [
        {"Field": "Scenario_ID", "Meaning": "Auto-assigned group of screenshots linked by an EXACT shared Ticket ID / Claim ID / OSC ID / Member ID."},
        {"Field": "User", "Meaning": "Most frequent non-blank name across Assignee / Resolved By / Updated By within the scenario's linked rows."},
        {"Field": "Gap-Fill Notes", "Meaning": "Every cross-row repair applied to this screenshot before the datasets were built."},
    ]

    df_raw = pd.DataFrame(raw_rows, columns=["Sequence No.", "Scenario_ID", "User", "Source File", "Document Index", "Detected Application"] + all_variable_names + ["Extra Fields", "Flags"])
    df_consolidated = pd.DataFrame(consolidated_rows, columns=["Sequence No.", "Scenario_ID", "User", "Source File", "Document Index", "Detected Application"] + all_variable_names + ["Extra Fields", "Flags", "Cross-Fill Notes", "Gap-Fill Notes"])
    df_summary = pd.DataFrame(summary_rows)
    df_dict = pd.DataFrame(dictionary_rows)
    df_a = build_dataset_a(all_results)
    df_b = build_dataset_b(all_results)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_a.to_excel(writer, index=False, sheet_name="Dataset A - Claims")
        df_b.to_excel(writer, index=False, sheet_name="Dataset B - State Log")
        df_summary.to_excel(writer, index=False, sheet_name="Scenario Summary")
        df_consolidated.to_excel(writer, index=False, sheet_name="Consolidated")
        df_raw.to_excel(writer, index=False, sheet_name="Raw Extracted")
        df_dict.to_excel(writer, index=False, sheet_name="Data Dictionary")

    # ---- Formatting pass --------------------------------------------------------
    wb = load_workbook(output_path)
    header_font = Font(name="Arial", bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="FF1F4E78", end_color="FF1F4E78", fill_type="solid")
    body_font = Font(name="Arial")
    flag_fill = PatternFill(start_color="FFFFF2CC", end_color="FFFFF2CC", fill_type="solid")

    for ws in wb.worksheets:
        for cell in ws[1]:
            cell.font = header_font
            cell.fill = header_fill
        ws.freeze_panes = "A2"
        header_vals = [c.value for c in ws[1]]
        flag_cols = {i for i, h in enumerate(header_vals, start=1) if h in ("Flags", "Cross-Fill Notes", "Gap-Fill Notes", "ID Conflicts", "Flag Details", "Gap-Fill Details")}
        for row_cells in ws.iter_rows(min_row=2):
            for cell in row_cells:
                cell.font = body_font
                if cell.column in flag_cols and cell.value:
                    cell.fill = flag_fill
        for col_cells in ws.columns:
            longest = max((len(str(c.value)) if c.value is not None else 0) for c in col_cells)
            ws.column_dimensions[get_column_letter(col_cells[0].column)].width = min(max(longest + 2, 10), 45)
    wb.save(output_path)

    return output_path


def _collect_input_paths(raw_paths: List[str]) -> List[str]:
    """Expand CLI arguments into a deterministic list of .txt files.
    Direct file arguments are accepted. Directory arguments are walked
    recursively and only .txt files are included; images and other formats
    are ignored."""
    collected: List[str] = []
    seen = set()

    def _add_path(path: str) -> None:
        if not path:
            return
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                dirs.sort()
                files.sort()
                for name in files:
                    if name.lower().endswith(".txt"):
                        full_path = os.path.join(root, name)
                        if full_path not in seen:
                            seen.add(full_path)
                            collected.append(full_path)
            return
        if os.path.isfile(path):
            if path.lower().endswith(".txt") and path not in seen:
                seen.add(path)
                collected.append(path)
            return
        print(f"(skipping {path}: not found or unsupported)")

    for raw_path in raw_paths:
        _add_path(raw_path)

    def _natural_sort_key(path: str):
        # Natural/human sort: split the FULL path into alternating digit /
        # non-digit chunks and compare digit chunks numerically. The
        # distinguishing sequence number often sits in a PARENT FOLDER name
        # (e.g. "A-1", "A-10", "B-7"), not in the (identical) leaf filename.
        parts = re.split(r"(\d+)", path)
        return [int(p) if p.isdigit() else p.lower() for p in parts]

    return sorted(collected, key=_natural_sort_key)