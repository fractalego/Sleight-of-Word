#!/usr/bin/env python3
"""Memory-safe sweep report: streams *.judged.jsonl line-by-line and prints the per-model
JUDGE-LABEL distribution + the entity(Mandela effect)-vs-control breakdown.

Reports the four raw judge labels (ignored / corrected / flagged / derailed) directly
rather than a 'noticed' rollup — 'noticed' conflated 'flagged' with 'derailed', and
derailment is disruption, not necessarily recognition (see LAB_LOG 2026-07-09/10).

Usage: uv run python scripts/report_sweep.py [run_dir]
"""
import glob
import json
import os
import sys
from collections import defaultdict

RUN = sys.argv[1] if len(sys.argv) > 1 else "results/runs/sweep-v2"
LABELS = ("ignored", "corrected", "flagged", "derailed")


class Acc:
    def __init__(self):
        self.n = self.subs = self.think = self.aware_s = self.jn = 0
        self.ds_s = self.de_s = 0.0
        self.ds_n = self.de_n = 0
        self.lab = defaultdict(int)

    def add(self, r):
        self.n += 1
        m = r.get("metrics") or {}
        self.subs += m.get("num_substitutions") or 0
        if (r.get("thinking") or {}).get("detected"):
            self.think += 1
        j = r.get("judge") or {}
        # multi-label (independent booleans) with legacy single-label fallback
        def eff(k):
            v = j.get(k)
            return bool(v) if v is not None else (j.get("label") == k)
        judged = bool(j.get("label")) or any(
            j.get(k) is not None for k in ("flagged", "corrected", "derailed"))
        if judged:
            self.jn += 1
            hit = False
            for k in ("flagged", "corrected", "derailed"):
                if eff(k):
                    self.lab[k] += 1
                    hit = True
            if j.get("aware"):
                self.aware_s += 1
                hit = True
            if not hit:
                self.lab["ignored"] += 1
        for v, s, nattr in ((m.get("delta_surprisal"), "ds_s", "ds_n"),
                            (m.get("delta_entropy"), "de_s", "de_n")):
            if v is not None:
                setattr(self, s, getattr(self, s) + v)
                setattr(self, nattr, getattr(self, nattr) + 1)

    def lp(self, k): return f"{100*self.lab[k]/self.jn:.0f}%" if self.jn else "-"
    def awarep(self): return f"{100*self.aware_s/self.jn:.1f}%" if self.jn else "-"
    def avg(self, s, n): return f"{s/n:+.2f}" if n else "-"


overall = {}
ent = defaultdict(Acc)
ctl = defaultdict(Acc)
files = sorted(glob.glob(os.path.join(RUN, "*.judged.jsonl")))
for f in files:
    model = os.path.basename(f)[:-len(".judged.jsonl")]
    a = Acc()
    for line in open(f, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        a.add(r)
        (ent if r.get("replacement", "").startswith("Mandela") else ctl)[model].add(r)
    overall[model] = a

order = sorted(overall, key=lambda m: (overall[m].lab["flagged"] / overall[m].jn)
               if overall[m].jn else -1, reverse=True)

print(f"\n=== SWEEP REPORT — {len(files)} models, {RUN} ===")
print("judge labels (% of judged) — INDEPENDENT under the panel judge; need not sum to 100. ignored = none apply\n")
h = f"{'model':24}{'swaps':>9}{'think':>7}{'ign':>6}{'corr':>6}{'flag':>6}{'derail':>7}{'aware':>7}{'Δsurp':>8}{'Δent':>8}"
print(h); print("-" * len(h))
for m in order:
    a = overall[m]
    print(f"{m:24}{a.subs:>9}{str(round(100*a.think/a.n))+'%':>7}"
          f"{a.lp('ignored'):>6}{a.lp('corrected'):>6}{a.lp('flagged'):>6}{a.lp('derailed'):>7}"
          f"{a.awarep():>7}{a.avg(a.ds_s,a.ds_n):>8}{a.avg(a.de_s,a.de_n):>8}")

print("\n=== ENTITY ('Mandela effect') vs CONTROL — flagged & derailed ===\n")
h2 = f"{'model':24}{'M-flag':>8}{'C-flag':>8}{'Δ':>5}{'M-der':>8}{'C-der':>8}{'Δ':>6}"
print(h2); print("-" * len(h2))
tef = tcf = ted = tcd = ten = tcn = 0
for m in order:
    e, c = ent[m], ctl[m]
    if not e.jn or not c.jn:
        continue
    ef = 100*e.lab['flagged']/e.jn; cf = 100*c.lab['flagged']/c.jn
    ed = 100*e.lab['derailed']/e.jn; cd = 100*c.lab['derailed']/c.jn
    tef += e.lab['flagged']; tcf += c.lab['flagged']; ted += e.lab['derailed']; tcd += c.lab['derailed']
    ten += e.jn; tcn += c.jn
    print(f"{m:24}{ef:>7.0f}%{cf:>7.0f}%{ef-cf:>+5.0f}{ed:>7.0f}%{cd:>7.0f}%{ed-cd:>+6.0f}")
print("-" * len(h2))
ef, cf, ed, cd = 100*tef/ten, 100*tcf/tcn, 100*ted/ten, 100*tcd/tcn
print(f"{'POOLED':24}{ef:>7.1f}%{cf:>7.1f}%{ef-cf:>+5.1f}{ed:>7.1f}%{cd:>7.1f}%{ed-cd:>+6.1f}")
print()
