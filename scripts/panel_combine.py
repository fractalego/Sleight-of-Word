#!/usr/bin/env python3
"""Combine the 3 panel-judge sidecars into MAJORITY verdicts on the main judged files.

For each trial and each label (flagged / corrected / derailed / aware), the panel verdict
is majority-of-3 (>=2 of the judges true). Writes the majority into each trial's `judge`
field AND keeps the three per-judge verdicts under `panel` for the agreement /
self-preference analysis. Idempotent; requires all 3 sidecars present per target.

Usage: uv run python scripts/panel_combine.py [--run results/runs/sweep-v2]
"""
import argparse
import glob
import json
import os

FIELDS = ("flagged", "corrected", "derailed", "aware")
# A label is assigned when AT LEAST TWO jurors agree (>=2). With two jurors this is
# unanimity; with three it is a majority — the same rule, so interim 2-judge results
# are refined (not redefined) when the third juror lands.


def load_sidecar(run, tag, model):
    path = os.path.join(run, "panel", tag, f"{model}.jsonl")
    d = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            d[r["key"]] = {f: r.get(f) for f in FIELDS}
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/runs/sweep-v2")
    ap.add_argument("--judges", default="gemma26,qwen36,llama70",
                    help="comma-separated judge tags to combine (label true iff >=2 agree)")
    args = ap.parse_args()
    TAGS = tuple(t.strip() for t in args.judges.split(",") if t.strip())

    targets = sorted(glob.glob(os.path.join(args.run, "*.judged.jsonl")))
    print(f"combining judges: {TAGS} (threshold >=2)")
    for tfile in targets:
        model = os.path.basename(tfile)[:-len(".judged.jsonl")]
        side = {tag: load_sidecar(args.run, tag, model) for tag in TAGS}
        counts = {tag: len(side[tag]) for tag in TAGS}
        if min(counts.values()) == 0:
            print(f"[skip] {model}: missing sidecars {counts}")
            continue

        tmp = tfile + ".tmp"
        n = merged = 0
        with open(tfile, encoding="utf-8") as fin, open(tmp, "w", encoding="utf-8") as fout:
            for line in fin:
                if not line.strip():
                    continue
                r = json.loads(line)
                n += 1
                key = f"{r['prompt_id']}::{r['trigger']}->{r['replacement']}"
                votes = {tag: side[tag].get(key) for tag in TAGS}
                if all(votes[t] is not None for t in TAGS):
                    maj = {}
                    for fld in FIELDS:
                        yes = sum(1 for t in TAGS if votes[t].get(fld))
                        maj[fld] = yes >= 2
                    j = r.get("judge") or {}
                    j.update(maj)
                    j["label"] = None  # multi-label now; legacy single label retired
                    j["panel"] = {t: votes[t] for t in TAGS}
                    j["judge_model"] = "panel(>=2 of: " + ",".join(TAGS) + ")"
                    r["judge"] = j
                    merged += 1
                fout.write(json.dumps(r) + "\n")
        os.replace(tmp, tfile)
        print(f"{model}: merged {merged}/{n}  (sidecars {counts})")
    print("done.")


if __name__ == "__main__":
    main()
