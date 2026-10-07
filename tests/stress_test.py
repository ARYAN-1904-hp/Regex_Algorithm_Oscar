"""
Agent-independence / new-pattern stress test.

Re-runs a real batch with the agent's name swapped for other agents (and
OCR-noisy versions of them), with OCR-damaged screen headers and labels,
and checks every extracted field still matches the clean run.

    python tests/stress_test.py "C:\\path\\to\\BATCH_xxx.zip" [--agent "Melissa Rippon"]

Exit code 0 = every case matched.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main as M  # noqa: E402
from claims_extractor.pattern_learner import KnowledgeBase, PatternLearner  # noqa: E402
from claims_extractor.screen_parser import extract_screen, learn_batch  # noqa: E402

SKIP = ("Source File", "Auto-Corrections")


def run(items, transform=lambda s: s, kb_path=None):
    kb_path = kb_path or os.path.join(tempfile.mkdtemp(), "kb.json")
    learner = PatternLearner(KnowledgeBase.load(kb_path))
    raws = [(n, transform(r)) for n, r in items]
    learn_batch(learner, [r for _n, r in raws])
    res, times = [], {}
    for n, raw in raws:
        r, _ = extract_screen(raw, learner=learner, source_name=n)
        res.append((n, r))
        times[n] = M._capture_time(n)
    learner.kb.save()
    return M.build_claim_summary(res, times) + M.build_screens_rows(res, times), learner


def swapper(agent: str, first: str, last: str, variants=None):
    a_first, a_last = agent.split(" ", 1)
    full = f"{first} {last}"
    ini = (first[0] + last.split()[-1][0]).upper()
    a_ini = (a_first[0] + a_last.split()[-1][0]).upper()
    reps = [(agent.upper(), full.upper()), (agent, full), (f"{a_first} {a_last[0]}.", f"{first} {last[0]}."),
            (f"{a_ini} {a_first}", f"{ini} {first}"), (a_first, first), (a_last, last)]
    count = [0]

    def t(s: str) -> str:
        if variants:
            def sub(_m):
                count[0] += 1
                return variants[count[0] % len(variants)]
            s = re.sub(re.escape(agent), sub, s)
        for a, b in reps:
            s = s.replace(a, b)
        return s
    return t


def diff(base, new, back: dict, skip_auto: bool = False):
    out = []
    for b, n in zip(base, new):
        for k in set(b) | set(n):
            if k in SKIP or (skip_auto and k.startswith("Auto: ")):
                continue
            nv = str(n.get(k, ""))
            for a, c in back.items():
                nv = nv.replace(a, c)
            if str(b.get(k, "")) != nv:
                out.append((b.get("Claim ID") or b.get("Seq"), k, str(b.get(k, ""))[:50], nv[:50]))
    if len(base) != len(new):
        out.append(("ROWS", len(base), len(new), ""))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("batch")
    ap.add_argument("--agent", help="Agent name in this batch (default: auto-detected)")
    args = ap.parse_args()
    items = M.read_inputs(args.batch)
    base, learner = run(items)
    agent = args.agent or learner.agent
    print(f"Batch agent: {agent}")
    failures = 0

    def report(label, d):
        nonlocal failures
        failures += bool(d)
        print(f"{'PASS' if not d else 'FAIL'}  {label}  diffs={len(d)} {dict(Counter(x[1] for x in d)) if d else ''}")
        for x in d[:5]:
            print("      ", x)

    cases = [
        ("John", "Smith", None),
        ("Juan", "de la Cruz", None),
        ("José", "Núñez", None),
        ("Mary-Jane", "O'Connor", None),
        ("Jessica", "Brister", ["Jessica Brister", "JessicaBrister", "Jesslca Brlster", "JESSICA BRISTER"]),
        ("Paige", "Turner", ["Paige Turner", "Palge Turner", "Paige  Turner"]),
        ("Liam", "Oliveira", ["Liam Oliveira", "Llam 0liveira", "Liam Oliveira", "LiamOliveira"]),
    ]
    for first, last, variants in cases:
        new, lr = run(items, swapper(agent, first, last, variants))
        full = f"{first} {last}"
        d = diff(base, new, {full: agent})
        if lr.agent != full:
            d.append(("AGENT", "Agent", full, str(lr.agent)))
        report(f"agent '{full}'{' (OCR-noisy)' if variants else ''} -> detected '{lr.agent}'", d)

    def headers(s):
        s = re.sub(r"^Cortex", "C0rtex", s)
        s = re.sub(r"^Claims Home Inventory", "Clalms Home lnventory", s)
        return re.sub(r"^rosco", "rosc0", s)
    report("OCR-damaged headers (C0rtex / Clalms Home lnventory / rosc0)", diff(base, run(items, headers)[0], {}))
    report("never-seen header spelling (Cortx)",
           diff(base, run(items, lambda s: re.sub(r"^Cortex", "Cortx", s))[0], {}))

    damage = [("Service start date", "Servlce start date"), ("Received date", "Receiveddate"),
              ("Claim state", "Clalm state"), ("Billed", "Bllled"), ("Pay provider", "Payprovider"),
              ("Assignee", "Asslgnee"), ("Updated On", "Updated 0n"), ("Created On", "Createdon"),
              ("Stay type", "Stay tvpe"), ("Billing type", "Bllling type"), ("Transmission ID", "Transmlssion lD")]

    def labels(s):
        for a, b in damage:
            s = s.replace(a, b)
        return s
    report("OCR-damaged / glued labels", diff(base, run(items, labels)[0], {}))

    # Learning carries over: batch 1 teaches the name, batch 2 only shows OCR noise.
    kb = os.path.join(tempfile.mkdtemp(), "kb.json")
    run(items, swapper(agent, "Paige", "Turner"), kb)
    new, lr = run(items, swapper(agent, "Paige", "Turner", ["Palge Turner"]), kb)
    # Auto: columns legitimately grow -- batch 2 also extracts fields batch 1 was still watching.
    report("knowledge base reused across batches ('Palge Turner' only)",
           diff(base, new, {"Paige Turner": agent}, skip_auto=True))

    print("ALL PASSED" if not failures else f"{failures} case(s) FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
