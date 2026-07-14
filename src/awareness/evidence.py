"""Judge-independent textual evidence that a model *noticed* the swapped word.

Scans the produced text for self-correction / meta-comment markers -- hard textual
evidence that the model remarked on the oddity, as opposed to merely being internally
perturbed (which surprisal/entropy capture). Heuristic and deliberately separate from
the LLM judge so the two signals can be compared.

Three strong families (any -> ``flagged``):
  * correction / apology      ("I mean", "let me correct myself", "I apologize")
  * anomaly-calling           ("doesn't make sense", "that's a funny/strange word", "typo")
  * substitution-awareness    ("I can't say ...", "keeps getting replaced", "every time I")
plus WEAK hesitation markers ("wait", "hmm", "actually") that are recorded but, alone,
do not set ``flagged``.

Note: because every trigger word is swapped, the model's correction phrasing can't
contain the trigger itself (e.g. "the") -- so the patterns here avoid relying on it.
"""

from __future__ import annotations

import re

from .schemas import TextEvidence

# --- correction / apology / explicit self-repair ---------------------------------
_CORRECTION = {
    "did_you_mean": r"\b(?:did|do) you mean\b|\byou (?:probably |might |may )?mean[t]?\b|\bperhaps you mean[t]?\b",
    "i_mean": r"\bI mean\b",
    "i_meant": r"\bI meant\b",
    "let_me_correct": r"\blet me correct\b|\bcorrect(?:ing)? myself\b|\bto correct that\b|\bcorrection\b",
    "apology": r"\bI apologi[sz]e\b|\b(?:my )?apologies\b|\bI'?m sorry\b|\bsorry,\b",
    "should_be": r"\bshould (?:be|say|read)\b|\bmeant to say\b|\bsupposed to (?:be|say)\b",
    # Narrow: reasoning models say "I think it/that..." as normal narration. Only count
    # "I think there's a mistake / I think you mean"-style references to the anomaly.
    "i_think_there": r"\bI (?:think|believe|suspect) (?:there'?s|there is|there'?re|you mean)\b",
    "let_me_try": r"\blet me try again\b|\bstart over\b",
}

# --- anomaly-calling: naming the word as wrong / strange / nonsense ---------------
_ANOMALY = {
    "typo": r"\btypos?\b",
    "misspell": r"\bmis-?spell",
    "doesnt_make_sense": r"\b(?:doesn'?t|does not|don'?t|not) make(?:s)? sense\b",
    "seems_wrong": r"\bseems? (?:to be )?(?:wrong|off|incorrect|odd|strange|out of place|a typo|a mistake|an error|nonsensical)\b",
    "doesnt_seem_right": r"\b(?:doesn'?t|does not) (?:seem|sound|look) (?:right|correct)\b",
    "a_mistake": r"\b(?:a|an|some|any) (?:mistake|error)\b|\bthere(?:'s| is|'?s been| seems to be) (?:a|an) (?:mistake|error|typo)\b",
    "not_a_word": r"\b(?:isn'?t|is not|not) a (?:real )?word\b",
    "nonsense": r"\b(?:nonsense|gibberish|nonsensical)\b",
    "anomaly_adj": r"\b(?:strange|weird|bizarre|funny|random|unusual|peculiar|nonsensical|absurd|silly)\b",
    "joke": r"\b(?:joke|joking|kidding|playful|prank)\b",
    "something_wrong": r"\bsomething (?:is|seems|'s|feels) (?:wrong|off|not right|strange|weird|amiss)\b",
    "confused": r"\b(?:confus(?:ed|ing)|puzzl(?:ed|ing)|baffl(?:ed|ing))\b",
    "wrong_category": r"\bnot a (?:city|capital|country|place|location)\b|\bis a (?:fruit|type of fruit)\b",
    "fourth_wall": r"\bhere to help\b|\bhappy to help\b|\bI'?m an AI\b",
}

# --- substitution-awareness: the model realizing a word was replaced ---------------
# HARD-WON LESSON (2026-07-09 audit): the swap inserts OBJECT words, so the model's
# answer often becomes prose ABOUT that object — and object-prose contains this
# vocabulary innocently ("clipboard data ... pasted or replaced by new", "wagons were
# replaced by motorized vehicles", "cells can't produce enough energy"). Every pattern
# here must therefore be anchored to the WORD/TEXT itself or to a first-person act,
# never bare verb phrases.
_SUBSTITUTION = {
    # "can't say/write/spell" only counts aimed at a word-like object (or a quote)
    "cant_say": r"\b(?:can'?t|cannot|can not|unable to|won'?t let me|couldn'?t) (?:seem to )?"
                r"(?:say|write|type|spell|output) (?:the |that |this |my )?(?:word|term|name|phrase|it)\b"
                r"|\b(?:can'?t|cannot|unable to) (?:say|write|type|spell) ['\"“]",
    # replace/swap/substitute anchored to the word/text/self, or the "keeps..." form
    # subject must be word-like: bare "it was replaced by ..." is historical prose
    # ("the empire ... it was replaced by the republic"), not substitution-awareness
    "being_replaced": r"\bshould (?:be|have been) (?:replaced|swapped|substituted)\b"
                      r"|\b(?:word|text|term|answer|response) (?:was|is|got|has been) (?:replaced|swapped|substituted)\b"
                      r"|\bI (?:replaced|substituted|swapped)\b"
                      r"|\bkeeps? (?:getting |being )?(?:replaced|changed|swapped|turning)\b",
    "every_time": r"\b(?:every|each) time I (?:say|try|write|type|use)\b",
    "autocorrect": r"\bauto-?correct\b",  # genuine: models blame the anomaly on autocorrect
    "keeps_coming_out": r"\bkeeps? coming out (?:as|wrong)\b|\bcomes out as ['\"“]",
}

_WEAK = {
    "wait": r"\bwait\b",
    "hmm": r"\bhmm+\b",
    "actually": r"\bactually\b",
    "hold_on": r"\bhold on\b|\bhold up\b",
    "i_notice": r"\bI notice\b",
    "oh": r"\boh,\b",
    "seriously": r"\bseriously\b",
}


def _compile(d):
    return {k: re.compile(v, re.IGNORECASE) for k, v in d.items()}


# The substitution family is tracked on its own: matching it means the model named the
# *mechanism* (a word was replaced), which is the stronger ``aware`` signal -- a strict
# subset of the strong markers that set ``flagged``.
_SUBSTITUTION_C = _compile(_SUBSTITUTION)
_STRONG_C = {**_compile(_CORRECTION), **_compile(_ANOMALY), **_SUBSTITUTION_C}
_WEAK_C = _compile(_WEAK)


def detect(text: str) -> TextEvidence:
    if not text:
        return TextEvidence()
    markers: list[str] = []
    first_pos: int | None = None
    for name, pat in _STRONG_C.items():
        m = pat.search(text)
        if m:
            markers.append(name)
            first_pos = m.start() if first_pos is None else min(first_pos, m.start())
    strong = bool(markers)
    aware = any(name in _SUBSTITUTION_C for name in markers)
    for name, pat in _WEAK_C.items():
        m = pat.search(text)
        if m:
            markers.append(f"weak:{name}")
            first_pos = m.start() if first_pos is None else min(first_pos, m.start())

    snippet = ""
    if first_pos is not None:
        a = max(0, first_pos - 30)
        snippet = text[a : first_pos + 50].strip()
    return TextEvidence(flagged=strong, aware=aware, markers=markers, snippet=snippet)
