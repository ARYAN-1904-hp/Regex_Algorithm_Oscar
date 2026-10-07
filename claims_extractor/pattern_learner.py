"""
Self-learning layer on top of text_cleaner + screen_parser.

The hand-written rules in text_cleaner / screen_parser were built from one
agent's batches (Melissa Rippon). This module makes the pipeline adapt when
the agent, the OCR noise or the screens change -- without code edits:

  * Agent / people: the operator of a batch is identified automatically
    from "Welcome <name>", Assignee / Updated By / Resolved By / Closed By,
    the Rosco header ("MR Melissa" = initials + first name) and Gmail.
    OCR variants of one person ("JessicaBrister", "Jesslca Brlster",
    "JESSICA BRISTER") are clustered on an OCR-confusable key and rebuilt
    into one canonical spelling by letter-level voting.
  * Label repair: a run of words that is an OCR-damaged / glued / split
    form of a known screen label ("Servlce start date", "Receiveddate",
    "Bil led") is rewritten to the label the regexes expect.
  * Word repair: a rare word that is an OCR misread (l/1/I for i, 0 for o,
    rn for m) of a word the batch uses often is corrected.
  * New patterns: header variants of known screens are matched fuzzily,
    unknown screens are logged, and new "Label <date/$ amount>" fields are
    discovered and (once seen repeatedly) extracted as "Auto: <Label>".

Everything learned is stored in knowledge_base.json next to main.py and
reused on the next batch. The file is plain JSON so a reviewer can edit it:
set "blocked": true on a bad auto-fix to stop it, or add a fix by hand with
"source": "manual".
"""

from __future__ import annotations

import datetime as dt
import difflib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .text_cleaner import clean_line

KB_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "knowledge_base.json"

# ---------------------------------------------------------------------------
# OCR-confusable folding
# ---------------------------------------------------------------------------
_FOLD_TABLE = str.maketrans({"l": "i", "1": "i", "|": "i", "!": "i", "0": "o"})


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def fold(s: str, keep_spaces: bool = False) -> str:
    """Collapse the characters OCR confuses, so 'Mellssa' == 'Melissa' and
    'C0rtex' == 'Cortex'. Used only as a comparison key, never as output."""
    s = strip_accents(s).lower().translate(_FOLD_TABLE).replace("rn", "m")
    return re.sub(r"[^a-z0-9 ]" if keep_spaces else r"[^a-z0-9]", "", s)


# ---------------------------------------------------------------------------
# Person names (agent-independent)
# ---------------------------------------------------------------------------
# A capitalised name word: unicode letters (José, Núñez), apostrophes and
# hyphens (O'Connor, Mary-Jane), OCR digits inside (Mel1ssa) and initials (R.).
# A leading 0/1 counts when OCR read O/I as a digit ("0liveira").
_CAP_WORD = r"(?:[A-ZÀ-ÖØ-Þ]|[01](?=[a-zà-öø-ÿ]{2}))[^\W_]*(?:['’\-][^\W_]+)*\.?"
_PARTICLES = r"(?:de|del|della|der|den|da|das|do|dos|di|du|la|le|van|von|bin|ibn|al|el|st\.?|y)"
# Single spaces only, so a name never runs onto the next label.
PERSON = rf"{_CAP_WORD}(?: (?:{_PARTICLES} )*{_CAP_WORD}){{0,3}}"
_PARTICLE_SET = set(re.sub(r"[()?:\\.]", " ", _PARTICLES).split()) | {"st."}

# UI words that end a name when OCR puts the next column on the same line.
NAME_STOP = {
    "resolve", "start", "next", "ticket", "change", "closed", "created", "updated",
    "member", "watch", "watchers", "edit", "view", "skill", "resolution", "reason",
    "please", "sla", "due", "submitter", "on", "by", "edt", "est", "cdt", "cst",
    "pdt", "pst", "am", "pm", "description", "queue", "state", "code", "assignee",
    "priority", "status", "details", "open", "new", "save", "cancel", "home",
    "primary", "time", "worked", "inventory", "claims", "claim", "search", "rosco",
    "unknown", "audit", "blocked", "paid", "denied", "pending", "approved",
    "escalated", "withdrawn", "paused", "hold", "in", "progress",
}
_CAMEL_SPLIT_RE = re.compile(r"(?<=[a-zà-öø-ÿ])(?=[A-ZÀ-ÖØ-Þ])")


def clean_person(raw: Optional[str]) -> Optional[str]:
    """Trim a regex-captured name at the first UI word and validate it:
    two+ name words, or one OCR-glued CamelCase word ('JessicaBrister')."""
    if not raw:
        return None
    words: List[str] = []
    for w in raw.replace(",", " ").split():
        if words and w.lower().rstrip(".") in NAME_STOP:
            break
        if not words and w.lower() in NAME_STOP:
            return None
        words.append(w)
    while words and words[-1].lower() in _PARTICLE_SET:
        words.pop()
    if len(words) >= 2:
        return " ".join(words)
    if len(words) == 1 and len(_CAMEL_SPLIT_RE.split(words[0])) >= 2 and len(words[0]) >= 6:
        return words[0]
    return None


def display_name(raw: str) -> str:
    """'JessicaBrister' -> 'Jessica Brister', 'JESSICA BRISTER' -> 'Jessica
    Brister', 'Juan DE LA Cruz' -> 'Juan de la Cruz'."""
    s = _CAMEL_SPLIT_RE.sub(" ", raw.strip())
    out = []
    for i, w in enumerate(s.split()):
        lw = w.lower()
        if i and lw in _PARTICLE_SET:
            out.append(lw)
        elif w.isupper() and len(w) > 2:
            out.append("-".join(p[:1] + p[1:].lower() for p in w.split("-")))
        else:
            out.append(w)
    return " ".join(out)


def person_key(name: str) -> str:
    return fold(display_name(name))


_CHAR_PREF = {c: i for i, c in enumerate("ilI1|!")}  # prefer the real letter on ties


def _vote_spelling(variants: Counter) -> str:
    """Rebuild the most likely true spelling from many OCR sightings: take the
    most common word layout, then vote each character position."""
    disp = Counter()
    for v, n in variants.items():
        weight = n * (0.5 if v.isupper() else 1.0)  # ALL-CAPS loses casing info
        disp[display_name(v)] += weight
    shapes: Dict[Tuple[int, ...], Counter] = defaultdict(Counter)
    for d, n in disp.items():
        shapes[tuple(len(w) for w in d.split())][d] += n
    best_shape = max(shapes, key=lambda s: sum(shapes[s].values()))
    group = shapes[best_shape]
    chars = []
    for i in range(sum(best_shape) + len(best_shape) - 1):
        votes = Counter()
        for d, n in group.items():
            votes[d[i]] += n
        top = max(votes.values())
        tied = [c for c, n in votes.items() if n == top]
        tied.sort(key=lambda c: (c.isdigit(), _CHAR_PREF.get(c, -1) if c in _CHAR_PREF else -1))
        chars.append(tied[0])
    return "".join(chars)


# Where the operator's name shows up, with how strongly it marks the
# logged-in user (Welcome banner / Rosco header) vs. a ticket field.
_AGENT_EVIDENCE = [
    (re.compile(rf"\bWelcome,?[ ]+({PERSON})"), 5, "welcome"),
    (re.compile(rf"\bAssignee[ ]+({PERSON})"), 2, "assignee"),
    (re.compile(rf"\b(?:Updated|Resolved|Closed) By[ ]+({PERSON})"), 2, "ticket"),
    (re.compile(rf"\bHi,?[ ]+({PERSON}),"), 1, "greeting"),
    (re.compile(rf"\(({PERSON})\)"), 1, "calendar"),
]
# Rosco top bar: "rosco Q Search Rosco MR Melissa -" (initials + first name).
_ROSCO_USER_RE = re.compile(rf"^rosco\b.*?\b([A-Z]{{2,3}}) ({_CAP_WORD})\s*-?\s*$", re.I | re.M)


# ---------------------------------------------------------------------------
# Known labels -- the vocabulary screen_parser's regexes are anchored on
# ---------------------------------------------------------------------------
LABEL_PHRASES = [
    # Cortex
    "About this claim", "Latest version", "Claim state", "Blockers", "Denials",
    "Billed", "Allowed", "Pay provider", "Pay member", "Member responsibility",
    "Service start date", "Service end date", "Received date", "Claim type",
    "Stay type", "Billing type", "Place of service", "Diagnosis related group",
    "Transmission ID", "Authorization required", "Is Medicaid reclamation",
    "International Provider", "Vendor source", "Tenant", "Prompt process due date",
    "Payment authority", "Interest accrued", "Excluded from interest",
    "Patient relationship", "Workflow", "Matched Member", "Submitted Info",
    "Billing provider", "Rendering provider", "Save note", "No notes for this claim",
    "EDI file", "Valet file ID", "Member info",
    # Appian
    "Ocelot ID", "In Progress", "Assignee", "Submitter", "Created On", "Updated On",
    "Resolved On", "Closed On", "Updated By", "Resolved By", "Closed By",
    "Created By", "Member ID", "Resolution", "Welcome", "Primary Queue",
    "Claims Home Inventory", "working days",
    # Rosco
    "Viewing information for", "Claims Export CSV", "Medical claims", "Basic info",
]
_LABEL_BY_FOLD: Dict[str, str] = {fold(p): p for p in LABEL_PHRASES}
_LABEL_MAX_WORDS = max(len(p.split()) for p in LABEL_PHRASES) + 1
# Plain alphabetic tokens (OCR may put 0/1/| inside), optional ':' / '?'.
_LABEL_TOKEN_RE = re.compile(r"^([A-Za-z0-9|!'’]*[A-Za-z][A-Za-z0-9|!'’]*)([:?]?)$")

# Screen header markers (classify_screen checks the folded header for these).
SCREEN_MARKERS = ["cortex", "claims home inventory", "rosco", "verint",
                  "knowledge management", "gmail", "looker studio"]
# Markers that open line 1 of a screen, with the spelling the parser's
# header regexes expect -- a damaged one is rewritten to this.
_LEAD_SPELLING = {"cortex": "Cortex", "claims home inventory": "Claims Home Inventory", "rosco": "rosco"}

# "New field" discovery: <Label words> <strict value>.
_DATE = r"\d{1,2}/\d{1,2}/\d{4}"
_MONEY = r"\$[\d,]+\.\d{2}"
_NEW_FIELD_RE = re.compile(
    rf"(?:^|(?<=\s))([A-Z][a-z]+(?: [a-z]+){{0,3}}|[A-Z][a-z]+(?: [A-Z][a-z]+){{0,2}}):?\s+({_MONEY}|{_DATE}(?:\s+\d{{1,2}}:\d{{2}}\s*[AP]M)?)"
)
_AUTO_FIELD_MIN_SEEN = 3
# Generic words that are never a field label on their own, and words a
# real label never ends with (text bleeding in from a neighbouring column).
_GENERIC_LABEL_WORDS = {"name", "group", "date", "dates", "expired", "effective", "total",
                        "from", "to", "through", "since", "until", "client", "oscar", "lines"}
_JUNK_LABEL_TAILS = {"name", "group", "client", "oscar", "lines", "from", "to", "through", "until"}


# Letter pairs OCR swaps beyond the folded ones (e/c, a/o, n/h, f/t ...).
# A fuzzy label match is accepted only through one of these -- 'claims
# Date' must never become 'Claim state' through a d/t swap.
_OCR_SUBS = {frozenset(p) for p in ("ce", "ao", "eo", "nh", "uv", "vy", "gq", "ft", "bh", "s5", "z2", "b8", "b6", "ij")}


def _one_ocr_substitution(a: str, b: str) -> bool:
    if len(a) != len(b):
        return False
    diffs = [(x, y) for x, y in zip(a, b) if x != y]
    return len(diffs) == 1 and frozenset(diffs[0]) in _OCR_SUBS


def _match_label(joined_fold: str) -> Optional[str]:
    hit = _LABEL_BY_FOLD.get(joined_fold)
    if hit or len(joined_fold) < 8:
        return hit
    # One OCR-typical letter swap, only when exactly one label is that close.
    close = [p for f, p in _LABEL_BY_FOLD.items() if _one_ocr_substitution(joined_fold, f)]
    return close[0] if len(close) == 1 else None


_WORD_RE = re.compile(r"(?<![\w$/])([A-Za-z][A-Za-z0-9|]{3,})(?![\w/])")


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------

def _now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class KnowledgeBase:
    """Persistent store of everything the learner has seen. Plain JSON."""

    SECTIONS = ("agents", "people", "label_fixes", "word_fixes", "vocabulary",
                "screen_aliases", "unknown_screens", "discovered_fields", "batches")

    def __init__(self, path: Path, data: Optional[Dict[str, Any]] = None):
        self.path = Path(path)
        self.data: Dict[str, Any] = data or {}
        for s in self.SECTIONS:
            self.data.setdefault(s, [] if s == "batches" else {})

    @classmethod
    def load(cls, path: Optional[str] = None) -> "KnowledgeBase":
        p = Path(path) if path else KB_DEFAULT_PATH
        if p.is_file():
            try:
                return cls(p, json.loads(p.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                bad = p.with_suffix(".corrupt.json")
                p.replace(bad)
        return cls(p)

    def save(self) -> None:
        self.data["updated"] = _now()
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)

    def __getitem__(self, k: str) -> Any:
        return self.data[k]


# ---------------------------------------------------------------------------
# Learner
# ---------------------------------------------------------------------------

class PatternLearner:
    """Use per batch:
        learner = PatternLearner(KnowledgeBase.load())
        learner.learn_batch(cleaned_texts, classify_screen, labels)  # pass 1
        text, fixes = learner.correct_text(cleaned)                    # pass 2,
        learner.postprocess(result, text)                              # per screen
        learner.finish_batch(name); learner.kb.save()
    """

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.agent: Optional[str] = None
        self.agent_evidence: Dict[str, int] = {}
        self.new_this_batch: List[Dict[str, str]] = []
        self._word_fix_map: Dict[str, str] = {}
        self._person_index: Dict[str, str] = {}   # person_key -> canonical
        self._known_labels: List[str] = []
        self.batch_name = ""
        self._repeat = False
        self._rebuild_people_index()
        self._rebuild_word_fixes()

    # ----- bookkeeping -------------------------------------------------
    def _note_new(self, kind: str, pattern: str, value: str) -> None:
        self.new_this_batch.append({"Type": kind, "Pattern": pattern, "Learned": value})

    def _rebuild_people_index(self) -> None:
        self._person_index = {}
        for canon, rec in self.kb["people"].items():
            self._person_index[person_key(canon)] = canon
            for alias in rec.get("aliases", {}):
                self._person_index.setdefault(person_key(alias), canon)

    def _rebuild_word_fixes(self) -> None:
        self._word_fix_map = {
            w: rec["fix"] for w, rec in self.kb["word_fixes"].items() if not rec.get("blocked")
        }

    # ----- pass 1: learn from the whole batch --------------------------
    def learn_batch(self, texts: List[str], classify: Any = None, known_labels: Iterable[str] = (),
                    batch_name: str = "") -> None:
        """Pass 1 over the whole batch. With `classify` (screen_parser.
        classify_screen) new "Label <value>" fields are counted here too, so
        every screen of the batch is treated alike in pass 2. Re-running a
        batch already in the knowledge base applies what was learned but
        does not count its sightings twice."""
        self.batch_name = batch_name
        self._repeat = bool(batch_name) and any(b.get("batch") == batch_name for b in self.kb["batches"])
        self._learn_vocabulary_and_word_fixes(texts)
        self._learn_people(texts)
        if classify:
            self._known_labels = list(known_labels)
            for t in texts:
                t, _ = self.correct_text(t, record=False)
                app = classify(t, self)
                store = self.kb["discovered_fields"].setdefault(app, {})
                for label, value in self._field_hits(store, t):
                    rec = store.get(label)
                    if rec is None:
                        rec = store[label] = {"count": 0, "example": f"{label} {value}",
                                              "first_seen": _now(), "blocked": False}
                    if self._repeat:
                        continue
                    rec["count"] += 1
                    if rec["count"] == _AUTO_FIELD_MIN_SEEN:
                        self._note_new(f"New field ({app})", label, f"e.g. {value}")

    def _learn_vocabulary_and_word_fixes(self, texts: List[str]) -> None:
        freq: Counter = Counter()
        for t in texts:
            for w in _WORD_RE.findall(t):
                if not w.isupper():
                    freq[w.lower()] += 1
        vocab = self.kb["vocabulary"]
        for w, n in freq.items():
            if w.isalpha() and not self._repeat:
                vocab[w] = vocab.get(w, 0) + n
        # Trusted words: label words + words common across all batches.
        trusted = {w.lower() for p in LABEL_PHRASES for w in p.split() if len(w) >= 4}
        trusted |= {w for w, n in vocab.items() if n >= 5}
        by_fold: Dict[str, set] = defaultdict(set)
        for w in trusted:
            by_fold[fold(w)].add(w)
        for w, n in freq.items():
            if w in trusted or w in self.kb["word_fixes"]:
                continue
            targets = by_fold.get(fold(w), set()) - {w}
            if len(targets) != 1:
                continue
            target = next(iter(targets))
            # Only a RARE word next to a COMMON one is a misread -- two
            # real words that fold alike ("fall"/"fail") are both common.
            if n > 2 or vocab.get(target, 0) < max(5, 5 * n) or len(target) != len(w):
                continue
            self.kb["word_fixes"][w] = {"fix": target, "count": 0, "source": "auto",
                                        "first_seen": _now(), "blocked": False}
            self._note_new("OCR word fix", w, target)
        self._rebuild_word_fixes()

    def _learn_people(self, texts: List[str]) -> None:
        sightings: Dict[str, Counter] = defaultdict(Counter)   # key -> raw variants
        weight: Counter = Counter()
        evidence: Dict[str, Counter] = defaultdict(Counter)
        rosco_users: Counter = Counter()
        for t in texts:
            for rx, w, kind in _AGENT_EVIDENCE:
                if kind == "calendar" and "gmail" not in t[:200].lower():
                    continue
                for m in rx.finditer(t):
                    name = clean_person(m.group(1))
                    if not name:
                        continue
                    k = person_key(name)
                    sightings[k][name] += 1
                    weight[k] += w
                    evidence[k][kind] += 1
            for m in _ROSCO_USER_RE.finditer(t):
                rosco_users[(m.group(1).upper(), fold(m.group(2)))] += 1

        # Merge rare fuzzy variants into a frequent cluster (a dropped
        # letter: 'Melisa Rippon' -> 'Melissa Rippon'). Two DIFFERENT people
        # are both frequent, so they never merge.
        keys = sorted(sightings, key=lambda k: -sum(sightings[k].values()))
        merged: Dict[str, str] = {}
        for k in keys:
            n = sum(sightings[k].values())
            for big in keys:
                if big == k or big in merged:
                    continue
                if sum(sightings[big].values()) >= 3 * n and difflib.SequenceMatcher(None, k, big).ratio() >= 0.9:
                    merged[k] = big
                    break
        for k, big in merged.items():
            sightings[big].update(sightings.pop(k))
            weight[big] += weight.pop(k)
            evidence[big].update(evidence.pop(k))

        clusters = {}
        for k, variants in sightings.items():
            canon = self._person_index.get(k) or self._known_fuzzy(k) or _vote_spelling(variants)
            # Rosco header "MR Melissa" confirms the logged-in user.
            parts = canon.split()
            ini = (parts[0][:1] + parts[-1][:1]).upper()
            boost = rosco_users.get((ini, fold(parts[0])), 0)
            if boost:
                evidence[k]["rosco_header"] += boost
                weight[k] += 3 * boost
            clusters[k] = canon
            self._store_person(canon, variants)

        if weight:
            top = max(weight, key=weight.get)
            if weight[top] >= 3:
                self.agent = clusters[top]
                self.agent_evidence = dict(evidence[top])
                rec = self.kb["agents"].setdefault(self.agent, {"first_seen": _now(), "batches": 0})
                if rec["batches"] == 0:
                    self._note_new("Agent", ", ".join(sorted(sightings[top])), self.agent)
                rec["batches"] += 0 if self._repeat else 1
                rec["last_seen"] = _now()
                rec["evidence"] = self.agent_evidence
        self._rebuild_people_index()

    def _known_fuzzy(self, key: str) -> Optional[str]:
        best = max(self._person_index, key=lambda k: difflib.SequenceMatcher(None, k, key).ratio(), default=None)
        if best and difflib.SequenceMatcher(None, best, key).ratio() >= 0.92:
            return self._person_index[best]
        return None

    def _store_person(self, canon: str, variants: Counter) -> None:
        people = self.kb["people"]
        rec = people.get(canon)
        if rec is None:
            rec = people[canon] = {"aliases": {}, "first_seen": _now()}
        for v, n in variants.items():
            if v != canon and v not in rec["aliases"]:
                self._note_new("Name variant", v, canon)
            if v != canon:
                rec["aliases"][v] = rec["aliases"].get(v, 0) + n
        if not self._repeat:
            rec["seen"] = rec.get("seen", 0) + sum(variants.values())
        rec["last_seen"] = _now()

    def canonical_person(self, name: Optional[str]) -> Optional[str]:
        if not name:
            return name
        k = person_key(name)
        return self._person_index.get(k) or self._known_fuzzy(k) or display_name(name)

    # ----- pass 2: per screen ------------------------------------------
    def correct_text(self, text: str, record: bool = True) -> Tuple[str, List[str]]:
        """Apply label repair + learned word fixes. Returns the corrected
        text and the list of corrections made ('Servlce start date ->
        Service start date'). record=False: correct without storing."""
        fixes: List[str] = []
        text = self._repair_header(text, fixes)
        lines = [self._repair_labels(l, fixes, record) for l in text.split("\n")]
        text = "\n".join(lines)
        if self._word_fix_map:
            def sub(m: re.Match) -> str:
                w = m.group(1)
                fix = self._word_fix_map.get(w.lower())
                if not fix:
                    return w
                out = fix.capitalize() if w[:1].isupper() else fix
                fixes.append(f"{w} -> {out}")
                if record:
                    self.kb["word_fixes"][w.lower()]["count"] += 1
                return out
            text = _WORD_RE.sub(sub, text)
        return text, fixes

    def _repair_labels(self, line: str, fixes: List[str], record: bool = True) -> str:
        toks = list(re.finditer(r"\S+", line))
        if not toks:
            return line
        out, last, i = [], 0, 0
        while i < len(toks):
            done = False
            for k in range(min(_LABEL_MAX_WORDS, len(toks) - i), 0, -1):
                span = toks[i:i + k]
                parts = [_LABEL_TOKEN_RE.match(t.group(0)) for t in span]
                if not all(parts) or any(p.group(2) for p in parts[:-1]):
                    continue
                words = [p.group(1) for p in parts]
                label = _match_label(fold("".join(words)))
                if not label:
                    continue
                original = " ".join(words)
                same_text = original == label
                case_only = original.lower() == label.lower()
                # Case-only differences are fixed on capitalised multi-word
                # labels only; "updated by" / "member" in a note is prose.
                if same_text or (case_only and (len(label.split()) == 1 or not original[:1].isupper())):
                    break
                learned = self.kb["label_fixes"].get(original)
                if learned and learned.get("blocked"):
                    break
                if record:
                    if learned is None:
                        self.kb["label_fixes"][original] = {"fix": label, "count": 0, "source": "auto",
                                                            "first_seen": _now(), "blocked": False}
                        self._note_new("Label fix", original, label)
                    self.kb["label_fixes"][original]["count"] += 1
                out.append(line[last:span[0].start()])
                out.append(label + parts[-1].group(2))
                last = span[-1].end()
                fixes.append(f"{original} -> {label}")
                i += k
                done = True
                break
            if not done:
                i += 1
        out.append(line[last:])
        return "".join(out)

    def _repair_header(self, text: str, fixes: List[str]) -> str:
        """'C0rtex - Claim #...', 'rosc0 Q ...', 'Cortx ...' -> the canonical
        header, so the header-anchored regexes (Claim #, search query) work."""
        first, _, rest = text.partition("\n")
        for marker, spelling in _LEAD_SPELLING.items():
            n = len(marker.split())
            m = re.match(rf"([A-Za-z0-9|]+(?: [A-Za-z0-9|]+){{{n - 1}}})(?![A-Za-z0-9|])", first)
            if not m or m.group(1).lower() == marker:
                continue
            if self.match_screen_header(m.group(1).lower()) != marker:
                continue
            fixes.append(f"{m.group(1)} -> {spelling}")
            return clean_line(spelling + first[m.end():]) + ("\n" + rest if rest else "")
        return text

    # ----- screen classification help ----------------------------------
    def match_screen_header(self, head: str) -> Optional[str]:
        """Folded / fuzzy match of a screen header to a known marker, so
        'C0rtex', 'Cortx', 'Clalms Home lnventory' still classify. A fuzzy
        hit is stored as an alias and reused next time."""
        words = fold(head, keep_spaces=True).split()
        if not words:
            return None
        aliases = self.kb["screen_aliases"]
        for n in (3, 2, 1):
            cand = " ".join(words[:n])
            if cand in aliases and not aliases[cand].get("blocked"):
                aliases[cand]["count"] += 1
                return aliases[cand]["marker"]
        for marker in SCREEN_MARKERS:
            mw = fold(marker, keep_spaces=True)
            n = len(mw.split())
            cand = " ".join(words[:n])
            if cand == mw:
                return marker
            if len(mw) >= 5 and difflib.SequenceMatcher(None, cand, mw).ratio() >= 0.8:
                aliases[cand] = {"marker": marker, "count": 1, "source": "auto",
                                 "first_seen": _now(), "blocked": False}
                self._note_new("Screen header", cand, marker)
                return marker
        return None

    def record_unknown_screen(self, text: str, name: str) -> None:
        first = text.split("\n", 1)[0]
        sig = " ".join(re.sub(r"[^a-z ]", " ", first.lower()).split()[:5]) or "(empty)"
        rec = self.kb["unknown_screens"].get(sig)
        if rec is None:
            rec = self.kb["unknown_screens"][sig] = {"count": 0, "example_file": name,
                                                     "example_header": first[:200], "first_seen": _now()}
            self._note_new("Unknown screen", sig, "logged for review")
        rec["count"] += 1
        rec["last_seen"] = _now()

    # ----- post-processing of one extracted screen ---------------------
    _PERSON_FIELDS = ("Assignee", "Updated By", "Resolved By", "Closed By", "Agent")

    def postprocess(self, result: Any, text: str) -> None:
        for bag in (result.fields, result.extra_fields):
            for f in self._PERSON_FIELDS:
                v = bag.get(f)
                if isinstance(v, str) and v:
                    canon = self.canonical_person(v)
                    if canon != v:
                        bag[f] = canon
        if self.agent:
            result.extra_fields["Agent"] = self.agent
        self._discover_fields(result, text)

    def _field_hits(self, store: Dict[str, Any], text: str):
        """(label, value) for every "<new label> <date | $ amount>" on a screen."""
        known = {fold(k) for k in self._known_labels} | set(_LABEL_BY_FOLD)
        by_fold = {fold(k): k for k in store}
        seen = set()
        for m in _NEW_FIELD_RE.finditer(text):
            label, value = m.group(1).strip(), re.sub(r"\s+", " ", m.group(2))
            lf = fold(label)
            # Known label, or a known label with a word glued in front.
            if lf in known or any(lf.endswith(k) for k in known if len(k) >= 6):
                continue
            words = label.lower().split()
            if len(words) == 1 and (words[0] in NAME_STOP or words[0] in _GENERIC_LABEL_WORDS):
                continue
            if words[-1] in _JUNK_LABEL_TAILS:
                continue
            # OCR variants of one label ("Applled to deductible") share a record.
            label = by_fold.setdefault(lf, label)
            if label not in seen:
                seen.add(label)
                yield label, value

    def _discover_fields(self, result: Any, text: str) -> None:
        store = self.kb["discovered_fields"].get(result.detected_application, {})
        for label, value in self._field_hits(store, text):
            rec = store.get(label)
            if rec and rec["count"] >= _AUTO_FIELD_MIN_SEEN and not rec.get("blocked"):
                result.extra_fields.setdefault(f"Auto: {label}", value)

    # ----- end of batch ------------------------------------------------
    def finish_batch(self, screens: int) -> None:
        entry = {"batch": self.batch_name, "processed": _now(), "screens": screens,
                 "agent": self.agent or "", "new_patterns": len(self.new_this_batch)}
        batches = self.kb["batches"]
        old = next((i for i, b in enumerate(batches) if self._repeat and b.get("batch") == self.batch_name), None)
        if old is None:
            batches.append(entry)
        else:
            batches[old] = entry

    def report_rows(self) -> Dict[str, List[Dict[str, Any]]]:
        """Rows for the workbook's 'Learned Patterns' sheet."""
        rows: List[Dict[str, Any]] = []
        for canon, rec in sorted(self.kb["agents"].items()):
            rows.append({"Type": "Agent", "Pattern": canon,
                         "Learned": "current batch" if canon == self.agent else "",
                         "Seen": rec.get("batches", 0), "Status": "active",
                         "Detail": json.dumps(rec.get("evidence", {}))})
        for canon, rec in sorted(self.kb["people"].items()):
            for alias, n in rec.get("aliases", {}).items():
                rows.append({"Type": "Name variant", "Pattern": alias, "Learned": canon,
                             "Seen": n, "Status": "active", "Detail": ""})
        for kind, sec in (("Label fix", "label_fixes"), ("OCR word fix", "word_fixes")):
            for pat, rec in sorted(self.kb[sec].items()):
                rows.append({"Type": kind, "Pattern": pat, "Learned": rec["fix"], "Seen": rec.get("count", 0),
                             "Status": "blocked" if rec.get("blocked") else rec.get("source", "auto"), "Detail": ""})
        for pat, rec in sorted(self.kb["screen_aliases"].items()):
            rows.append({"Type": "Screen header", "Pattern": pat, "Learned": rec["marker"],
                         "Seen": rec.get("count", 0), "Status": "blocked" if rec.get("blocked") else "auto",
                         "Detail": ""})
        for sig, rec in sorted(self.kb["unknown_screens"].items()):
            rows.append({"Type": "Unknown screen", "Pattern": sig, "Learned": "needs a parser",
                         "Seen": rec["count"], "Status": "review",
                         "Detail": f"{rec['example_file']} | {rec['example_header']}"})
        for app, fields in sorted(self.kb["discovered_fields"].items()):
            for label, rec in sorted(fields.items()):
                active = rec["count"] >= _AUTO_FIELD_MIN_SEEN and not rec.get("blocked")
                rows.append({"Type": f"New field ({app})", "Pattern": label,
                             "Learned": f"Auto: {label}" if active else "watching",
                             "Seen": rec["count"], "Status": "blocked" if rec.get("blocked") else
                             ("extracting" if active else "candidate"), "Detail": rec["example"]})
        return {"all": rows, "new": self.new_this_batch}
