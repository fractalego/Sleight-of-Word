"""Offline sanity checks for the pure logic (no GPU / no vLLM needed).

Run with:  uv run python -m tests.test_logic
"""

import math
from pathlib import Path

from awareness import tokens
from awareness.config import Pair
from awareness.runner import compute_metrics, write_results, read_results
from awareness.schemas import TokenStep, TrialMetrics, TrialResult


class FakeTok:
    """Tiny tokenizer: id <-> string piece, good enough for trigger/encode tests."""

    def __init__(self):
        self.vocab = {
            10: "The", 11: " capital", 12: " of", 13: " France", 14: " is",
            15: " Paris", 16: " the", 17: "ory", 18: " umbrella", 19: " The",
            20: ",", 21: " theory", 22: "\n\n", 23: "   ", 24: '"the',
        }
        self.rev = {v: k for k, v in self.vocab.items()}

    def decode(self, ids):
        return "".join(self.vocab.get(i, "?") for i in ids)

    def encode(self, text, add_special_tokens=False):
        if text in self.rev:
            return [self.rev[text]]
        # split " umbrella" style single words we know; fall back to char ids
        return [self.rev.get(text, 18)]


def test_token_word():
    assert tokens.token_word(" the") == (True, "the", "")
    assert tokens.token_word("The") == (False, "The", "")
    assert tokens.token_word(" the,")[1] == "the"
    assert tokens.token_word(" 100")[1] is None
    print("ok: token_word")


def test_is_trigger():
    tk = FakeTok()
    # " the" followed by " of" -> whole word -> trigger
    toks = [16, 12]
    assert tokens.is_trigger(tk, toks, 0, "the", is_first=False)
    # first token "The" (no leading space) counts as word-initial
    assert tokens.is_trigger(tk, [10, 11], 0, "the", is_first=True)
    # " the"+"ory" == "theory" -> next token continues the word -> NOT a trigger
    assert not tokens.is_trigger(tk, [16, 17], 0, "the", is_first=False)
    # a single " theory" token is not "the"
    assert not tokens.is_trigger(tk, [21], 0, "the", is_first=False)
    # line-initial: '\n\n'+'The' — whitespace lives in the PREVIOUS token
    # (regression: the sweep-v1 dose bug silently skipped every such occurrence)
    assert tokens.is_trigger(tk, [22, 10, 11], 1, "the", is_first=False)
    # bullet indent: '   '+'The' — same shape
    assert tokens.is_trigger(tk, [23, 10, 11], 1, "the", is_first=False)
    # ' the'+'ory' after a newline is still 'theory', not a trigger
    assert not tokens.is_trigger(tk, [22, 16, 17], 1, "the", is_first=False)
    # DESIGN CHOICE: quoted mention '"the' is the model *talking about* the word
    # (mention, not use) — allowed to stand so models can articulate the correct word
    assert not tokens.is_trigger(tk, [14, 24], 1, "the", is_first=False)
    print("ok: is_trigger (whole-word, look-ahead, line-initial, quoted-mention)")


class YiTok:
    """Tokenizer whose single-token decode drops spaces but pair-decode restores them."""

    def __init__(self):
        self.v = {1: "The", 2: "capital", 3: "of", 4: "Paris", 5: "the", 6: "ory", 7: "umbrella"}
        self.cont = {6}  # 'ory' attaches to the previous token (no space)

    def decode(self, ids):
        s = ""
        for k, i in enumerate(ids):
            w = self.v[i]
            if k > 0 and i not in self.cont:
                s += " "
            s += w
        return s

    def encode(self, text, add_special_tokens=False):
        rev = {v: k for k, v in self.v.items()}
        return [rev.get(text.strip(), 7)]


def test_is_trigger_spaceless_tokenizer():
    tk = YiTok()
    # 'The capital' -> first token 'The' is a whole-word trigger (regression: Yi gave 0 subs)
    assert tokens.is_trigger(tk, [1, 2], 0, "the", is_first=True)
    # 'the'+'ory' == 'theory' -> NOT a trigger (pair-decode shows no boundary)
    assert not tokens.is_trigger(tk, [5, 6], 0, "the", is_first=False)
    # mid-sequence 'the' in 'Paris the capital' -> trigger via pair-decoded boundaries
    assert tokens.is_trigger(tk, [4, 5, 2], 1, "the", is_first=False)
    print("ok: is_trigger spaceless tokenizer (Yi fix)")


def test_first_trigger_round_start():
    """Round-initial trigger must be found via left-context from assistant_ids.

    Regression for the 2026-07-06 bug: each regeneration round's `toks` starts fresh, so
    a space-less trigger (Yi 'the'/'The') at a round start was missed because its left
    neighbour lived in assistant_ids, not toks.
    """
    from awareness.runner import _first_trigger
    tk = YiTok()
    # committed 'Paris'; the round emits space-less 'the capital' -> 'the' at j=0 must fire
    assert _first_trigger(tk, [4], [5, 2], "the") == 0
    # no prior context: first-token path still finds it (is_first)
    assert _first_trigger(tk, [], [1, 2], "the") == 0
    # 'the'+'ory' == 'theory' at a round start must NOT match
    assert _first_trigger(tk, [4], [5, 6], "the") is None
    print("ok: _first_trigger round-start left-context (round-initial fix)")


def test_encode_replacement():
    tk = FakeTok()
    # leading space + capitalized (trigger was "The") -> " umbrella" here (FakeTok maps to 18)
    assert tokens.encode_replacement(tk, "umbrella", leading_space=True, capitalize=False) == [18]
    print("ok: encode_replacement")


def test_entropy():
    class LP:
        def __init__(self, v): self.logprob = v
    # near-uniform over 2 -> ~ln2 ; peaked -> ~0
    flat = {1: LP(math.log(0.5)), 2: LP(math.log(0.5))}
    peaked = {1: LP(math.log(0.99)), 2: LP(math.log(0.01))}
    assert abs(tokens.entropy_nats(flat) - math.log(2)) < 1e-6
    assert tokens.entropy_nats(peaked) < 0.1
    print("ok: entropy_nats")


def test_compute_metrics():
    # clean baseline surprisal mean = 1.0 ; after a swap at index 2, next tokens spike
    clean = [TokenStep(index=i, token_id=0, text="x", surprisal=1.0, entropy=0.5) for i in range(5)]
    steps = [
        TokenStep(index=0, token_id=0, text="The", surprisal=2.0, entropy=0.4),
        TokenStep(index=1, token_id=0, text=" capital", surprisal=1.0, entropy=0.5),
        TokenStep(index=2, token_id=0, text=" umbrella", surprisal=None, entropy=None, substituted=True),
        TokenStep(index=3, token_id=0, text=" is", surprisal=5.0, entropy=2.0),
        TokenStep(index=4, token_id=0, text=" weird", surprisal=3.0, entropy=1.5),
    ]
    m = compute_metrics(clean, steps, sub_indices=[2], window=8)
    assert m.num_substitutions == 1
    assert abs(m.baseline_surprisal - 1.0) < 1e-9
    # post window after index 2 = surprisals 5.0, 4? -> (5.0+3.0)/2 = 4.0
    assert abs(m.post_sub_surprisal - 4.0) < 1e-9
    assert abs(m.delta_surprisal - 3.0) < 1e-9
    print("ok: compute_metrics -> delta_surprisal", m.delta_surprisal)


def test_text_evidence():
    from awareness import evidence
    # strong: explicit meta-comment
    e = evidence.detect("Wait, 'umbrella' doesn't make sense here — did you mean 'the'?")
    assert e.flagged is True
    assert "doesnt_make_sense" in e.markers or "did_you_mean" in e.markers
    # the real-run phrases the old detector missed
    assert evidence.detect("I mean, that's a funny answer!").flagged   # i_mean + anomaly_adj
    assert evidence.detect("Oh, let me correct myself: ...").flagged   # let_me_correct
    assert evidence.detect("I apologize, umbrella is not a city.").flagged  # apology + wrong_category
    # substitution-awareness ("aware"): names the mechanism, not just the oddity
    sub = evidence.detect("I can't say that word — it keeps getting replaced.")
    assert sub.flagged and sub.aware  # aware is a strict subset of flagged
    assert "cant_say" in evidence.detect("It seems I cannot say it.").markers
    # flagged but NOT aware: calls the word strange without naming a substitution
    odd = evidence.detect("Wait, 'umbrella' doesn't make sense here.")
    assert odd.flagged is True and odd.aware is False
    # weak only: hesitation but no strong marker -> not flagged, not aware
    # ordinary prose must NOT read as substitution-awareness (regression 2026-07-09:
    # "changing with altitude" fired being_replaced -> false aware)
    e3 = evidence.detect("people talk about boiling points changing with altitude.")
    assert not e3.aware, e3
    assert evidence.detect("my word keeps being replaced with something else").aware
    e2 = evidence.detect("Hmm, umbrella capital of France is Paris.")
    assert e2.flagged is False and e2.aware is False
    assert any(m.startswith("weak:") for m in e2.markers)
    # clean continuation -> nothing
    e3 = evidence.detect("Banana Republic")
    assert e3.flagged is False and e3.aware is False and e3.markers == []
    print("ok: text_evidence (expanded, with aware)")


class _FakeSP:
    def __init__(self, temperature=0.0, max_tokens=16, logprobs=None):
        self.temperature, self.max_tokens, self.logprobs = temperature, max_tokens, logprobs


class _LP:
    def __init__(self, lp):
        self.logprob = lp


class _FakeSeq:
    def __init__(self, toks):
        self.token_ids = list(toks)
        # vLLM always returns a logprob dict per token when logprobs is requested
        self.logprobs = [{t: _LP(-0.5)} for t in toks]
        self.text = "".join(str(t) for t in toks)


class _FakeOut:
    def __init__(self, toks):
        self.outputs = [_FakeSeq(toks)]


class _FakeLLM:
    """Deterministic greedy stand-in: continuation depends only on how many assistant
    tokens precede it (so a splice that lengthens the context shifts what comes next),
    exactly like a real model regenerating from a corrupted prefix."""

    def __init__(self, prefix_len, canned):
        self.prefix_len, self.canned = prefix_len, canned

    def generate(self, requests, sampling_params):
        sps = sampling_params if isinstance(sampling_params, list) \
            else [sampling_params] * len(requests)
        outs = []
        for req, sp in zip(requests, sps):
            alen = len(req["prompt_token_ids"]) - self.prefix_len
            outs.append(_FakeOut(self.canned[alen: alen + sp.max_tokens]))
        return outs


def test_batched_matches_sequential():
    from awareness.runner import intervention_run, intervention_run_batched, _IxnState
    from awareness.config import PromptSpec

    tk = FakeTok()
    prefix = [10, 11, 12]  # arbitrary; FakeLLM only uses its length
    canned = [10, 11, 16, 12, 13, 14, 15]  # "The capital the of France is Paris" (2 triggers)
    pair = Pair(trigger="the", replacement="umbrella")
    max_new, k = 50, 5
    llm = _FakeLLM(len(prefix), canned)

    seq_steps, seq_text, seq_sub = intervention_run(
        llm, _FakeSP, tk, prefix, max_new, k, pair
    )

    # batch three identical trials; each must reproduce the sequential result
    states = [_IxnState(PromptSpec(id=f"p{i}", query="q?"), pair, prefix) for i in range(3)]
    finished = []
    intervention_run_batched(llm, _FakeSP, tk, states, max_new, k, on_done=finished.append)

    assert len(finished) == 3
    for st in states:
        assert [s.token_id for s in st.steps] == [s.token_id for s in seq_steps]
        assert st.sub_indices == seq_sub
        assert tk.decode(st.assistant_ids) == seq_text
    assert seq_sub == [0, 2]  # spliced at the two triggers
    print("ok: batched intervention == sequential")


def test_checkpoint_resume_keys():
    from awareness.runner import _done_keys
    import tempfile, os

    rs = [
        TrialResult(trial_id="t1", model="m", prompt_id="q001", query="?",
                    trigger="the", replacement="Mandela"),
        TrialResult(trial_id="t2", model="m", prompt_id="q001", query="?",
                    trigger="the", replacement="umbrella"),
    ]
    d = tempfile.mkdtemp()
    p = Path(d) / "m.raw.jsonl"
    with p.open("w") as f:
        for r in rs:
            f.write(r.model_dump_json() + "\n")
        f.write('{"trial_id": "t3", "model": "m"  <<corrupt')  # simulate crash mid-write
    keys = _done_keys(p)
    assert keys == {("q001", "the", "Mandela"), ("q001", "the", "umbrella")}  # corrupt line skipped
    print("ok: checkpoint resume keys (tolerates truncated line)")


def test_jsonl_roundtrip():
    r = TrialResult(
        trial_id="t", model="m", prompt_id="p", query="q?",
        trigger="the", replacement="umbrella",
        steps=[TokenStep(index=0, token_id=1, text="The", surprisal=None, substituted=False),
               TokenStep(index=1, token_id=2, text=" umbrella", surprisal=None, substituted=True)],
        substitution_indices=[1],
        metrics=TrialMetrics(num_substitutions=1, baseline_surprisal=1.0,
                             post_sub_surprisal=4.0, delta_surprisal=3.0),
    )
    path = Path("/tmp") / "roundtrip.jsonl"
    write_results([r], path)
    back = read_results(path)
    assert len(back) == 1 and back[0].steps[0].surprisal is None
    assert back[0].metrics.delta_surprisal == 3.0
    print("ok: jsonl_roundtrip")


if __name__ == "__main__":
    test_token_word()
    test_is_trigger()
    test_is_trigger_spaceless_tokenizer()
    test_first_trigger_round_start()
    test_encode_replacement()
    test_entropy()
    test_compute_metrics()
    test_text_evidence()
    test_batched_matches_sequential()
    test_checkpoint_resume_keys()
    test_jsonl_roundtrip()
    print("\nall offline logic tests passed")
