#!/usr/bin/env python3
"""Regenerate Table 1 + Table 2 data rows in a paper tex file from the judged files.

Multi-label aware (independent booleans via panel combine; legacy label fallback).
Preserves the yellow \rowcolor{thinkrow} shading for thinking-capable models.

Usage: uv run python scripts/update_paper_tables.py [--tex paper/main-overleaf.tex]
"""
import argparse
import glob
import json
import os
import re

RUN = "results/runs/sweep-v2"
DISP = {
    "qwen2.5-14b": "Qwen2.5-14B", "deepseek-r1-distill-32b": "DeepSeek-R1-Distill-32B",
    "qwen2.5-72b-awq": "Qwen2.5-72B (AWQ)", "qwen2.5-32b-awq": "Qwen2.5-32B (AWQ)",
    "deepseek-r1-distill-7b": "DeepSeek-R1-Distill-7B", "qwen3-32b-awq": "Qwen3 32B (AWQ)",
    "gemma3-27b": "Gemma\\,3 27B", "qwen3.6-27b": "Qwen3.6 27B", "yi-1.5-34b": "Yi-1.5-34B",
    "gemma4-e4b": "Gemma\\,4 E4B", "llama3.3-70b-awq": "Llama\\,3.3 70B (AWQ)",
    "magistral-small-24b": "Magistral-Small-24B", "gemma4-31b": "Gemma\\,4 31B",
    "gemma4-e2b": "Gemma\\,4 E2B", "gpt-oss-20b": "gpt-oss-20B", "phi-4": "Phi-4",
    "gemma4-12b": "Gemma\\,4 12B", "gemma4-26b-a4b": "Gemma\\,4 26B-A4B",
    "mistral-small-24b": "Mistral-Small-24B",
}
THINKING = {"deepseek-r1-distill-32b", "deepseek-r1-distill-7b", "qwen3-32b-awq",
            "gpt-oss-20b", "gemma4-26b-a4b", "gemma4-31b", "gemma4-12b", "gemma4-e2b",
            "gemma4-e4b", "qwen3.6-27b", "magistral-small-24b",
            # phase-4 (2026-10-05): all three think by default under their templates
            "muse-glimmer-30b", "qwen3.8-27b", "nemotron3.5-lightning-30b"}


def eff(j, k):
    v = j.get(k)
    return bool(v) if v is not None else (j.get("label") == k)


def collect():
    stats = {}
    for f in sorted(glob.glob(os.path.join(RUN, "*.judged.jsonl"))):
        m = os.path.basename(f)[:-len(".judged.jsonl")]
        st = dict(n=0, jn=0, flag=0, ign=0, der=0, corr=0, aware=0, unt=0,
                  ds=0.0, dsn=0, de=0.0, den=0,
                  e_flag=0, e_der=0, en=0, s_flag=0, s_der=0, sn=0)
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            j = r.get("judge") or {}
            mt = r.get("metrics") or {}
            st["n"] += 1
            if (mt.get("num_substitutions") or 0) == 0:
                st["unt"] += 1
            if mt.get("delta_surprisal") is not None:
                st["ds"] += mt["delta_surprisal"]; st["dsn"] += 1
            if mt.get("delta_entropy") is not None:
                st["de"] += mt["delta_entropy"]; st["den"] += 1
            judged = bool(j.get("label")) or any(
                j.get(k) is not None for k in ("flagged", "corrected", "derailed"))
            if not judged:
                continue
            st["jn"] += 1
            fl, co, de_ = eff(j, "flagged"), eff(j, "corrected"), eff(j, "derailed")
            aw = bool(j.get("aware"))
            st["flag"] += fl; st["corr"] += co; st["der"] += de_; st["aware"] += aw
            if not (fl or co or de_ or aw):
                st["ign"] += 1
            ent = r["replacement"].startswith("Mandela")
            if ent:
                st["en"] += 1; st["e_flag"] += fl; st["e_der"] += de_
            else:
                st["sn"] += 1; st["s_flag"] += fl; st["s_der"] += de_
        stats[m] = st
    return stats


def pct(a, b, nd=0):
    if not b:
        return 0
    v = round(100 * a / b, nd)
    return int(v) if nd == 0 else v


def rows_t1(stats):
    out = []
    order = sorted(stats, key=lambda m: -pct(stats[m]["flag"], stats[m]["jn"], 2))
    for m in order:
        s = stats[m]
        jn = s["jn"] or 1
        pre = "    \\rowcolor{thinkrow}\n" if m in THINKING else ""
        out.append(
            f"{pre}    {DISP[m]:23} & {pct(s['flag'],jn):>2} & {pct(s['e_flag'],s['en']):>2} & "
            f"{pct(s['ign'],jn):>2} & {pct(s['der'],jn):>2} & {pct(s['corr'],jn)} & "
            f"{pct(s['aware'],jn,1):>3} & ${s['ds']/max(s['dsn'],1):+.2f}$ & "
            f"${s['de']/max(s['den'],1):+.2f}$ \\\\")
    return "\n".join(out), order


def rows_untouched(stats, order):
    out = []
    for m in order:
        s = stats[m]
        pre = "    \\rowcolor{thinkrow}\n" if m in THINKING else ""
        out.append(f"{pre}    {DISP[m]:23} & {pct(s['unt'],s['n'],1):>4} \\\\")
    return "\n".join(out)


def rows_t2(stats, order):
    out = []
    te = dict(f=0, d=0, n=0); ts = dict(f=0, d=0, n=0)
    for m in order:
        s = stats[m]
        ef, sf = pct(s["e_flag"], s["en"]), pct(s["s_flag"], s["sn"])
        ed, sd = pct(s["e_der"], s["en"]), pct(s["s_der"], s["sn"])
        te["f"] += s["e_flag"]; te["d"] += s["e_der"]; te["n"] += s["en"]
        ts["f"] += s["s_flag"]; ts["d"] += s["s_der"]; ts["n"] += s["sn"]
        pre = "    \\rowcolor{thinkrow}\n" if m in THINKING else ""
        out.append(
            f"{pre}    {DISP[m]:23} & {ef} & {sf} & ${ef-sf:+d}$  & {ed} & {sd} & ${ed-sd:+d}$ \\\\")
    pooled = (f"    Pooled (all models)     & {pct(te['f'],te['n'])} & {pct(ts['f'],ts['n'])} & "
              f"${pct(te['f'],te['n'])-pct(ts['f'],ts['n']):+d}$  & {pct(te['d'],te['n'])} & "
              f"{pct(ts['d'],ts['n'])} & ${pct(te['d'],te['n'])-pct(ts['d'],ts['n']):+d}$ \\\\")
    return "\n".join(out), pooled, (te, ts)


def splice(tex, t1_rows, t2_rows, pooled, unt_rows):
    # Table 1 rows: between the header's closing \hline and the final \hline of the tabular
    pat1 = re.compile(
        r"(\\textbf\{\(nats\)\} \\\\\n    \\hline\n).*?(\n    \\hline\n  \\end\{tabular\}\n  \\caption\{Main sweep)",
        re.DOTALL)
    tex, n1 = pat1.subn(lambda mo: mo.group(1) + t1_rows + mo.group(2), tex)
    # Table 2 rows (appendix): between header \hline and the Pooled row's preceding \hline
    pat2 = re.compile(
        r"(\\textbf\{\$\\Delta\$\} \\\\\n    \\hline\n).*?\n    \\hline\n(    Pooled[^\n]*\\\\)",
        re.DOTALL)
    tex, n2 = pat2.subn(lambda mo: mo.group(1) + t2_rows + "\n    \\hline\n" + pooled, tex)
    # Untouched appendix table rows
    pat3 = re.compile(
        r"(\\textbf\{Untouched \(\\%\)\} \\\\\n    \\hline\n).*?(\n    \\hline\n  \\end\{tabular\}\n  \\caption\{Untouched)",
        re.DOTALL)
    tex, n3 = pat3.subn(lambda mo: mo.group(1) + unt_rows + mo.group(2), tex)
    return tex, n1, n2, n3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default="paper/main-overleaf.tex")
    args = ap.parse_args()
    stats = collect()
    t1, order = rows_t1(stats)
    t2, pooled, (te, ts) = rows_t2(stats, order)
    unt = rows_untouched(stats, order)
    tex = open(args.tex, encoding="utf-8").read()
    tex, n1, n2, n3 = splice(tex, t1, t2, pooled, unt)
    if n1 != 1 or n2 != 1 or n3 != 1:
        raise SystemExit(
            f"SPLICE FAILED (t1={n1}, t2={n2}, untouched={n3}) — anchors not found; tex NOT modified")
    open(args.tex, "w", encoding="utf-8").write(tex)
    print(f"updated {args.tex}: table1 ({n1}), table2 ({n2}), untouched ({n3})")
    print(f"pooled entity flag {pct(te['f'],te['n'])}% vs subst {pct(ts['f'],ts['n'])}% | "
          f"derailed {pct(te['d'],te['n'])}% vs {pct(ts['d'],ts['n'])}%")


if __name__ == "__main__":
    main()
