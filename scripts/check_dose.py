#!/usr/bin/env python3
"""Dose + survivor audit for one raw.jsonl (post-generation, pre-judge).

Reports the achieved dose (fraction of trigger occurrences actually swapped) and, for
every surviving 'the'/'The' in the model's output, classifies WHY it survived — flagging
any category outside the three known-benign ones so we catch new tokenizations of "the".

  known-benign:
    - subword-merge     : 'the' glued inside a longer token (theory, these) — correct
    - quoted-mention    : token led by punctuation ('"the') — intentional (mention vs use)
    - spaceless-subword : no separator on either side in the token stream (Yi 'userThe')
  anything else -> UNKNOWN (investigate).

Usage: uv run python scripts/check_dose.py <model>.raw.jsonl [hf_model_id_for_verification]
"""
import json
import re
import sys
from collections import Counter

PATH = sys.argv[1]
WORD = re.compile(r"[A-Za-z']+")

swaps = 0
survivors = Counter()          # category -> count
examples = {}                  # category -> (prev, cur, next)
unknown_ex = []

for line in open(PATH):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    steps = r.get("steps") or []
    swaps += (r.get("metrics") or {}).get("num_substitutions") or 0
    for i, s in enumerate(steps):
        if s.get("substituted"):
            continue
        txt = s.get("text", "")
        # does this token contain a standalone 'the' as a word?
        if not re.search(r"\bthe\b", txt, re.IGNORECASE):
            continue
        m = WORD.search(txt)
        lead = txt[:m.start()] if m else ""
        word = m.group(0) if m else ""
        prev = steps[i - 1]["text"] if i > 0 else "<START>"
        nxt = steps[i + 1]["text"] if i + 1 < len(steps) else "<END>"

        if word.lower() != "the":
            cat = "subword-merge"          # 'the' is a substring of a longer alpha run
        elif nxt not in ("<END>",) and nxt[:1].isalpha():
            cat = "subword-merge"          # next token glues on: 'The'+'ft' = 'Theft'
        elif lead and not lead[-1:].isspace():
            cat = "quoted-mention"          # punctuation glued before the word, same token
        elif not prev[-1:].isspace() and prev not in ("<START>",) and not re.search(r"\s", prev[-1:]):
            # neither this token nor the previous provides a separating space in-stream
            cat = "spaceless-subword"
        else:
            cat = "UNKNOWN"
        survivors[cat] += 1
        examples.setdefault(cat, (prev, txt, nxt))
        if cat == "UNKNOWN" and len(unknown_ex) < 25:
            unknown_ex.append((prev, txt, nxt))

missed = sum(survivors.values())
total = swaps + missed
print(f"file: {PATH}")
print(f"swapped occurrences : {swaps:,}")
print(f"surviving 'the'     : {missed:,}")
print(f"DOSE                : {100*swaps/total:.2f}%  (of {total:,} total occurrences)\n")
print("survivors by category:")
for cat, c in survivors.most_common():
    p, cur, n = examples[cat]
    flag = "  <-- INVESTIGATE" if cat == "UNKNOWN" else ""
    print(f"  {c:>7}  {cat:18} e.g. prev={p!r} tok={cur!r} next={n!r}{flag}")

if survivors.get("UNKNOWN"):
    print(f"\n!! {survivors['UNKNOWN']} UNKNOWN survivors — new tokenization of 'the'? samples:")
    for p, cur, n in unknown_ex:
        print(f"     prev={p!r}  tok={cur!r}  next={n!r}")
else:
    print("\nOK: every surviving 'the' falls into a known-benign category.")
