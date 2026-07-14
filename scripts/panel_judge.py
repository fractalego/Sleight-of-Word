#!/usr/bin/env python3
"""One panel judge over a whole run — STREAMING, CHUNKED, RESUMABLE.

Judges every target model's trials with a single judge model, using the multi-label
prompt. Verdicts are appended to a per-judge sidecar in chunks, so stopping (Ctrl-C, a
crash, a kill) loses at most one chunk (~1 min); re-running skips everything already done.

Memory-safe: streams each target file line-by-line (never loads all ~192k records), so it
won't OOM like the all-at-once judge.

Sidecar layout:  <run>/panel/<judge_tag>/<target_model>.jsonl
  one line per judged trial: {key, pid, rep, flagged, corrected, derailed, aware}

Usage:
  uv run python scripts/panel_judge.py --judge-model google/gemma-4-26B-A4B-it \
      --judge-tag gemma26 --quant fp8 [--run results/runs/sweep-v2] [--chunk 1000]
"""
import argparse
import glob
import json
import os

from awareness.judge import SYSTEM, USER_TEMPLATE, _verdict_from

EXPECTED = 10100


def trial_key(pid, trigger, rep):
    return f"{pid}::{trigger}->{rep}"


def done_keys(path):
    keys = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                keys.add(json.loads(line)["key"])
            except Exception:  # noqa: BLE001 - tolerate a torn final line from a kill
                continue
    return keys


def stream_todo(target_file, done):
    """Yield minimal judgeable trials from a target file not already in `done`."""
    for line in open(target_file, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        # only judge trials where a swap happened and there is text
        if not r.get("reaction_text") or not r.get("substitution_indices") or r.get("error"):
            continue
        k = trial_key(r["prompt_id"], r["trigger"], r["replacement"])
        if k in done:
            continue
        yield {"key": k, "pid": r["prompt_id"], "trigger": r["trigger"],
               "rep": r["replacement"], "query": r["query"], "rx": r["reaction_text"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--judge-tag", required=True)
    ap.add_argument("--quant", default=None)
    ap.add_argument("--load-format", default=None)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--run", default="results/runs/sweep-v2")
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--no-think", action="store_true",
                    help="disable a hybrid judge's thinking channel (Qwen3) so it emits JSON directly")
    ap.add_argument("--only", default="", help="comma-separated target models (default all)")
    args = ap.parse_args()

    outdir = os.path.join(args.run, "panel", args.judge_tag)
    os.makedirs(outdir, exist_ok=True)

    targets = sorted(glob.glob(os.path.join(args.run, "*.judged.jsonl")))
    if args.only:
        want = {x.strip() for x in args.only.split(",")}
        targets = [t for t in targets if os.path.basename(t)[:-len(".judged.jsonl")] in want]

    # Which targets still need work? (skip fully-judged so we don't load the model for nothing)
    plan = []
    for t in targets:
        m = os.path.basename(t)[:-len(".judged.jsonl")]
        if sum(1 for _ in open(t, encoding="utf-8")) < EXPECTED:
            print(f"[skip] {m}: target incomplete"); continue
        side = os.path.join(outdir, f"{m}.jsonl")
        have = len(done_keys(side))
        plan.append((m, t, side, have))
    pending = [p for p in plan if p[3] < EXPECTED]
    if not pending:
        print(f"[{args.judge_tag}] all {len(plan)} targets already judged — nothing to do")
        return

    from vllm import LLM, SamplingParams
    lf = args.load_format or ("bitsandbytes" if args.quant == "bitsandbytes" else "auto")
    print(f"[{args.judge_tag}] loading {args.judge_model} ...")
    llm = LLM(model=args.judge_model, quantization=args.quant, load_format=lf, dtype="auto",
              max_model_len=args.max_model_len, gpu_memory_utilization=0.9, enforce_eager=True)
    sp = SamplingParams(temperature=0.0, max_tokens=256)

    for m, tfile, side, have in plan:
        done = done_keys(side)
        if len(done) >= EXPECTED:
            print(f"[{args.judge_tag}] {m}: already complete ({len(done)})"); continue
        print(f"[{args.judge_tag}] {m}: {len(done)} done, judging the rest (chunk={args.chunk})")
        buf = []

        def flush(buf):
            if not buf:
                return
            convs = [[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": USER_TEMPLATE.format(
                          query=t["query"], trigger=t["trigger"],
                          replacement=t["rep"], reaction=t["rx"])}] for t in buf]
            ck = {"chat_template_kwargs": {"enable_thinking": False}} if args.no_think else {}
            outs = llm.chat(convs, sp, **ck)
            with open(side, "a", encoding="utf-8") as f:
                for t, o in zip(buf, outs):
                    v = _verdict_from(o.outputs[0].text, args.judge_tag)
                    f.write(json.dumps({"key": t["key"], "pid": t["pid"], "rep": t["rep"],
                                        "flagged": v.flagged, "corrected": v.corrected,
                                        "derailed": v.derailed, "aware": v.aware}) + "\n")
                f.flush()
                os.fsync(f.fileno())

        for t in stream_todo(tfile, done):
            buf.append(t)
            if len(buf) >= args.chunk:
                flush(buf); buf = []
        flush(buf)
        print(f"[{args.judge_tag}] {m}: done ({len(done_keys(side))})")

    print(f"[{args.judge_tag}] FINISHED")


if __name__ == "__main__":
    main()
