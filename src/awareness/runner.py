"""Phase 1: chat models answer a fixed question; we swap a trigger word in the
assistant's own output and watch what happens next.

Per (model, query) we first do a CLEAN run (greedy, no intervention) to get the model's
baseline surprisal/entropy. Then, per pair, an INTERVENTION run:

  generate greedily -> scan the output for the first whole-word trigger -> keep the
  tokens before it, replace the trigger token with the replacement token(s), then
  REGENERATE the rest from the corrupted context -> repeat for every later occurrence.

This is exactly "the model produced 'the', we substituted 'umbrella', it keeps going":
because a transformer only sees tokens (not whether it produced them), regenerating
from the spliced sequence gives the true surprisal of what the model says next given
the corruption. The tokens after each swap are the signal.

vLLM is imported lazily so the rest of the package works on a machine without a GPU.
"""

from __future__ import annotations

from pathlib import Path

from . import evidence, thinking, tokens
from .config import ExperimentConfig, ModelSpec, Pair, PromptSpec
from .schemas import TokenStep, TrialMetrics, TrialResult


def _thinking_default(tokenizer) -> bool:
    """Does the model's chat template open a thinking/thought channel by default?"""
    try:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": "x"}], add_generation_prompt=True, tokenize=False
        )
        tail = rendered[-80:].lower()
        return any(kw in tail for kw in ("thought", "<think", "analysis", "channel"))
    except Exception:  # noqa: BLE001
        return False


def build_prefix_ids(tokenizer, system_prompt: str, query: str) -> list[int]:
    """Chat-template token ids up to the start of the assistant turn."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": query})
    out = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True
    )
    # Depending on transformers version this is a list[int] or a BatchEncoding dict.
    if hasattr(out, "input_ids"):
        out = out["input_ids"]
    if out and isinstance(out[0], list):  # batched -> single conversation
        out = out[0]
    return [int(t) for t in out]


def _surprisal(logprob_dict, token_id: int):
    if logprob_dict and token_id in logprob_dict:
        return float(-logprob_dict[token_id].logprob)
    return None


def _step(tokenizer, idx: int, token_id: int, logprob_dict) -> TokenStep:
    return TokenStep(
        index=idx,
        token_id=token_id,
        text=tokenizer.decode([token_id]),
        surprisal=_surprisal(logprob_dict, token_id),
        entropy=tokens.entropy_nats(logprob_dict),
    )


def clean_run(llm, SamplingParams, tokenizer, prefix_ids, max_new, k):
    sp = SamplingParams(temperature=0.0, max_tokens=max_new, logprobs=k)
    out = llm.generate([{"prompt_token_ids": prefix_ids}], sp)[0].outputs[0]
    steps = [
        _step(tokenizer, i, tid, lp)
        for i, (tid, lp) in enumerate(zip(out.token_ids, out.logprobs))
    ]
    return steps, out.text


def _first_trigger(tokenizer, assistant_ids, toks, trigger):
    """Index into ``toks`` of the first whole-word ``trigger``, or None.

    Each regeneration round's ``toks`` starts fresh, so a round-initial token's left
    neighbour lives in the already-committed ``assistant_ids``, not in ``toks``. We
    prepend that one token (and offset the index) so ``_boundary_before`` can decide the
    left boundary. Without it, a space-less trigger token (e.g. Yi's ``the``/``The``,
    whose separator only appears in the pair-decode) at a round start is missed — the bug
    that left Yi under-dosed while space-preserving tokenizers were unaffected.
    """
    ctx = ([assistant_ids[-1]] + toks) if assistant_ids else toks
    off = 1 if assistant_ids else 0
    for j in range(len(toks)):
        is_first = (len(assistant_ids) == 0 and j == 0)
        if tokens.is_trigger(tokenizer, ctx, j + off, trigger, is_first):
            return j
    return None


def intervention_run(
    llm, SamplingParams, tokenizer, prefix_ids, max_new, k, pair: Pair
):
    """Greedy decode with every whole-word ``trigger`` swapped for ``replacement``."""
    steps: list[TokenStep] = []
    sub_indices: list[int] = []
    assistant_ids: list[int] = []
    idx = 0

    while len(assistant_ids) < max_new:
        budget = max_new - len(assistant_ids)
        sp = SamplingParams(temperature=0.0, max_tokens=budget, logprobs=k)
        out = llm.generate(
            [{"prompt_token_ids": prefix_ids + assistant_ids}], sp
        )[0].outputs[0]
        toks = list(out.token_ids)
        lps = out.logprobs

        found = _first_trigger(tokenizer, assistant_ids, toks, pair.trigger)

        upto = len(toks) if found is None else found
        for j in range(upto):
            steps.append(_step(tokenizer, idx, toks[j], lps[j]))
            assistant_ids.append(toks[j])
            idx += 1

        if found is None:
            break

        # splice in the replacement, mirroring spacing + capitalization of the trigger
        lead_ws, word, _ = tokens.token_word(tokens.piece(tokenizer, toks[found]))
        capitalize = bool(word) and word[0].isupper()
        rep_ids = tokens.encode_replacement(
            tokenizer, pair.replacement, leading_space=lead_ws, capitalize=capitalize
        )
        for n, rid in enumerate(rep_ids):
            steps.append(
                TokenStep(
                    index=idx, token_id=rid, text=tokenizer.decode([rid]),
                    surprisal=None, entropy=None, substituted=True,
                )
            )
            if n == 0:
                sub_indices.append(idx)
            assistant_ids.append(rid)
            idx += 1
        # loop: regenerate the remainder from the now-corrupted context

    text = tokenizer.decode(assistant_ids)
    return steps, text, sub_indices


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def compute_metrics(clean_steps, steps, sub_indices, window) -> TrialMetrics:
    baseline_s = _mean([s.surprisal for s in clean_steps])
    baseline_e = _mean([s.entropy for s in clean_steps])

    by_index = {s.index: s for s in steps}
    post_s, post_e = [], []
    for s0 in sub_indices:
        count, kk = 0, s0 + 1
        while count < window and kk in by_index:
            st = by_index[kk]
            if not st.substituted and st.surprisal is not None:
                post_s.append(st.surprisal)
                if st.entropy is not None:
                    post_e.append(st.entropy)
                count += 1
            kk += 1
    ps, pe = _mean(post_s), _mean(post_e)
    return TrialMetrics(
        num_substitutions=len(sub_indices),
        baseline_surprisal=baseline_s,
        baseline_entropy=baseline_e,
        post_sub_surprisal=ps,
        post_sub_entropy=pe,
        delta_surprisal=(ps - baseline_s) if (ps is not None and baseline_s is not None) else None,
        delta_entropy=(pe - baseline_e) if (pe is not None and baseline_e is not None) else None,
    )


# --------------------------------------------------------------------------------------
# Batched intervention: same semantics as ``intervention_run`` (above), but many trials
# advance together so vLLM sees a large batch each round instead of one sequence at a
# time. ``_apply_output`` is the per-trial body of that while-loop, factored out so the
# sequential and batched paths can't diverge.
# --------------------------------------------------------------------------------------
class _IxnState:
    """In-progress intervention for one (prompt, pair) trial."""

    __slots__ = ("prompt", "pair", "prefix_ids", "steps", "sub_indices",
                 "assistant_ids", "idx", "done")

    def __init__(self, prompt: PromptSpec, pair: Pair, prefix_ids: list[int]):
        self.prompt = prompt
        self.pair = pair
        self.prefix_ids = prefix_ids
        self.steps: list[TokenStep] = []
        self.sub_indices: list[int] = []
        self.assistant_ids: list[int] = []
        self.idx = 0
        self.done = False


def _apply_output(state: _IxnState, seq, tokenizer, max_new: int) -> None:
    """Consume one generate output for ``state``: keep tokens up to the first trigger,
    splice the replacement, and mark the trial done when no trigger is found or the
    budget is spent. Mirrors the body of ``intervention_run``'s while-loop exactly."""
    pair = state.pair
    toks = list(seq.token_ids)
    lps = seq.logprobs

    found = _first_trigger(tokenizer, state.assistant_ids, toks, pair.trigger)

    upto = len(toks) if found is None else found
    for j in range(upto):
        lp = lps[j] if lps is not None else None
        state.steps.append(_step(tokenizer, state.idx, toks[j], lp))
        state.assistant_ids.append(toks[j])
        state.idx += 1

    if found is None:
        state.done = True
        return

    lead_ws, word, _ = tokens.token_word(tokens.piece(tokenizer, toks[found]))
    capitalize = bool(word) and word[0].isupper()
    rep_ids = tokens.encode_replacement(
        tokenizer, pair.replacement, leading_space=lead_ws, capitalize=capitalize
    )
    for n, rid in enumerate(rep_ids):
        state.steps.append(
            TokenStep(index=state.idx, token_id=rid, text=tokenizer.decode([rid]),
                      surprisal=None, entropy=None, substituted=True)
        )
        if n == 0:
            state.sub_indices.append(state.idx)
        state.assistant_ids.append(rid)
        state.idx += 1

    if len(state.assistant_ids) >= max_new:
        state.done = True


def clean_run_batched(llm, SamplingParams, tokenizer, prefixes, max_new, k):
    """One clean (no-intervention) decode per prefix, all in a single batched call."""
    sp = SamplingParams(temperature=0.0, max_tokens=max_new, logprobs=k)
    reqs = [{"prompt_token_ids": pf} for pf in prefixes]
    outs = llm.generate(reqs, sp)
    res = []
    for out in outs:
        o = out.outputs[0]
        lps = o.logprobs if o.logprobs is not None else [None] * len(o.token_ids)
        steps = [_step(tokenizer, i, tid, lp)
                 for i, (tid, lp) in enumerate(zip(o.token_ids, lps))]
        res.append((steps, o.text))
    return res


def intervention_run_batched(llm, SamplingParams, tokenizer, states, max_new, k, on_done):
    """Advance every state in lockstep rounds. Each round batches all still-active
    trials into one ``llm.generate`` call (per-request max_tokens = remaining budget).
    ``on_done(state)`` is invoked the moment a trial finishes (for checkpointing)."""
    active = [s for s in states if not s.done]
    while active:
        reqs = [{"prompt_token_ids": s.prefix_ids + s.assistant_ids} for s in active]
        sps = [SamplingParams(temperature=0.0,
                              max_tokens=max(1, max_new - len(s.assistant_ids)),
                              logprobs=k)
               for s in active]
        outs = llm.generate(reqs, sps)
        for s, out in zip(active, outs):
            _apply_output(s, out.outputs[0], tokenizer, max_new)
            if s.done:
                on_done(s)
        active = [s for s in active if not s.done]


def _done_keys(path: Path) -> set[tuple[str, str, str]]:
    """(prompt_id, trigger, replacement) of trials already written (tolerant of a
    truncated final line from a crash)."""
    keys: set[tuple[str, str, str]] = set()
    if not path.exists():
        return keys
    with path.open(encoding="utf-8") as f:  # real newlines only (see read_results)
        for line in f:
            if not line.strip():
                continue
            try:
                r = TrialResult.model_validate_json(line)
            except Exception:  # noqa: BLE001 - skip a partial/corrupt line
                continue
            keys.add((r.prompt_id, r.trigger, r.replacement))
    return keys


def _append_result(r: TrialResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(r.model_dump_json() + "\n")
        f.flush()


def run_model(
    spec: ModelSpec,
    experiment: ExperimentConfig,
    prompts: list[PromptSpec],
    pairs: list[Pair],
    out_path: str | Path | None = None,
    overwrite: bool = False,
) -> list[TrialResult]:
    """Run every (prompt, pair) trial for one model.

    Batched: all clean runs in one call, then interventions advance in lockstep rounds.
    If ``out_path`` is given, each trial is appended to it as it completes (crash-safe)
    and, unless ``overwrite``, already-written trials are skipped so an interrupted model
    resumes where it left off.
    """
    from vllm import LLM, SamplingParams  # lazy

    out_path = Path(out_path) if out_path is not None else None
    if out_path is not None and overwrite and out_path.exists():
        out_path.unlink()
    done = _done_keys(out_path) if out_path is not None else set()

    remaining: dict[str, list[Pair]] = {}
    for p in prompts:
        rem = [pr for pr in pairs if (p.id, pr.trigger, pr.replacement) not in done]
        if rem:
            remaining[p.id] = rem
    if not remaining:  # already complete
        return read_results(out_path) if out_path is not None else []

    load_format = spec.load_format
    if spec.quantization == "bitsandbytes" and not load_format:
        load_format = "bitsandbytes"  # vLLM needs this paired with bnb quantization

    llm = LLM(
        model=spec.model,
        quantization=spec.quantization,
        load_format=load_format or "auto",
        dtype=spec.dtype,
        max_model_len=spec.max_model_len,
        gpu_memory_utilization=spec.gpu_memory_utilization,
        trust_remote_code=spec.trust_remote_code,
        # One-shot eval with short outputs: the ~25-min torch.compile + CUDA-graph warmup
        # never amortizes, so eager is a big net win (skips per-model compilation).
        enforce_eager=True,
        **(spec.extra or {}),  # per-model escape hatch (see ModelSpec.extra)
    )
    tokenizer = llm.get_tokenizer()
    k = experiment.top_logprobs
    max_new = experiment.max_new_tokens
    thinking_default = _thinking_default(tokenizer)

    prompt_by_id = {p.id: p for p in prompts}
    prefix_by_id = {
        pid: build_prefix_ids(tokenizer, experiment.system_prompt, prompt_by_id[pid].query)
        for pid in remaining
    }

    # 1) clean runs for every prompt that still has work -- one batched call
    needed = list(remaining)
    clean_by_id: dict[str, tuple[list[TokenStep], str, str | None]] = {}
    try:
        cleans = clean_run_batched(
            llm, SamplingParams, tokenizer, [prefix_by_id[pid] for pid in needed], max_new, k
        )
        for pid, (steps, text) in zip(needed, cleans):
            clean_by_id[pid] = (steps, text, None)
    except Exception as exc:  # noqa: BLE001 - clean failed for all; record on the trials
        err = f"{type(exc).__name__}: {exc}"
        for pid in needed:
            clean_by_id[pid] = ([], "", err)

    # 2) interventions for all remaining trials, advanced in batched rounds
    states = [
        _IxnState(prompt_by_id[pid], pair, prefix_by_id[pid])
        for pid in remaining for pair in remaining[pid]
    ]

    new_results: list[TrialResult] = []

    def _finish(st: _IxnState) -> None:
        cs, ct, cerr = clean_by_id.get(st.prompt.id, ([], "", None))
        reaction = tokenizer.decode(st.assistant_ids)
        metrics = compute_metrics(cs, st.steps, st.sub_indices, experiment.post_window)
        r = TrialResult(
            trial_id=f"{spec.name}::{st.prompt.id}::{st.pair.trigger}->{st.pair.replacement}",
            model=spec.name,
            prompt_id=st.prompt.id,
            query=st.prompt.query,
            trigger=st.pair.trigger,
            replacement=st.pair.replacement,
            clean_text=ct,
            reaction_text=reaction,
            clean_steps=cs,
            steps=st.steps,
            substitution_indices=st.sub_indices,
            metrics=metrics,
            text_evidence=evidence.detect(reaction),
            thinking=thinking.detect(reaction, default_on=thinking_default),
            error=cerr,
        )
        if out_path is not None:
            _append_result(r, out_path)
        new_results.append(r)

    intervention_run_batched(llm, SamplingParams, tokenizer, states, max_new, k, _finish)

    del llm
    _free_gpu()
    if out_path is not None:
        return read_results(out_path)  # full set: prior + newly appended
    return new_results


def _free_gpu() -> None:
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def write_results(results: list[TrialResult], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in results:
            f.write(r.model_dump_json() + "\n")


def read_results(path: str | Path) -> list[TrialResult]:
    # Iterate file lines (splits on real newlines only) rather than str.splitlines(),
    # which also splits on U+2028/NEL etc. — gpt-oss emits U+2028 inside reaction text,
    # and splitting there shears the JSON record in half (2026-07-09).
    with Path(path).open(encoding="utf-8") as f:
        return [TrialResult.model_validate_json(line) for line in f if line.strip()]


def raw_files(path: str | Path) -> list[Path]:
    """Resolve a path to the raw-result files it contains (file -> itself; dir -> *.raw.jsonl)."""
    p = Path(path)
    if p.is_dir():
        return sorted(p.glob("*.raw.jsonl"))
    return [p]


def read_results_path(path: str | Path) -> list[TrialResult]:
    """Read results from a file, or from a run dir (prefers judged, then raw, then any)."""
    p = Path(path)
    if not p.is_dir():
        return read_results(p)
    for pattern in ("**/*.judged.jsonl", "**/*.raw.jsonl", "**/*.jsonl"):
        files = sorted(p.glob(pattern))
        if files:
            out: list[TrialResult] = []
            for f in files:
                out.extend(read_results(f))
            return out
    return []


def append_index(index_path: str | Path, run_id: str, results: list[TrialResult]) -> None:
    """Append one summary row per model to an append-only CSV log of every run."""
    import csv
    from collections import defaultdict

    by_model: dict[str, list[TrialResult]] = defaultdict(list)
    for r in results:
        by_model[r.model].append(r)

    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    new = not index_path.exists()
    with index_path.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["run_id", "model", "n_trials", "text_flag_rate",
                        "judge_flagged_rate", "aware_rate", "mean_delta_surprisal", "errors"])
        for model, rs in by_model.items():
            n = len(rs)
            tf = sum(int(r.text_evidence.flagged) for r in rs) / n if n else 0
            judged = [r for r in rs if r.judge]
            nr = (sum(int(r.judge.has("flagged")) for r in judged) / len(judged)) if judged else ""
            # only verdicts that actually assessed awareness (aware is not None) count
            aware_judged = [r for r in judged if r.judge.aware is not None]
            aw = (sum(int(r.judge.aware) for r in aware_judged) / len(aware_judged)) if aware_judged else ""
            ds_vals = [r.metrics.delta_surprisal for r in rs if r.metrics.delta_surprisal is not None]
            ds = (sum(ds_vals) / len(ds_vals)) if ds_vals else ""
            errs = sum(1 for r in rs if r.error)
            w.writerow([run_id, model, n, round(tf, 3),
                        (round(nr, 3) if nr != "" else ""),
                        (round(aw, 3) if aw != "" else ""),
                        (round(ds, 3) if ds != "" else ""), errs])
