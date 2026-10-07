"""Detect whether a model's output actually contains reasoning / "thinking".

Two kinds of signal:
  * explicit markers -- <think> tags, or channel tokens (Gemma 4 / harmony-style
    '<|channel>thought', '<channel|>', 'analysis'/'final' channels)
  * a chain-of-thought opener at the very start ("Okay, so I need to figure out...",
    "Let me think...") -- needed because reasoning models often emit the opening
    <think> in the *prompt*, so only the monologue survives in the output.

This records what *appeared in the answer*; whether a model's template enables thinking
by default is tracked separately (``default_on``, set at generation time).
"""

from __future__ import annotations

import re

from .schemas import ThinkingInfo

_MARKERS = {
    "think_tag": re.compile(r"</?think\b", re.IGNORECASE),
    "channel": re.compile(r"[<|]\s*channel|channel\s*[|>]|<\|?(?:thought|analysis|final)\|?>", re.IGNORECASE),
    "analysis_channel": re.compile(r"\b(?:analysis|thought) channel\b", re.IGNORECASE),
    # Muse Glimmer reasons in a recipient-addressed channel: ' to=self<|message|>...<|eom|>'
    # before answering in '<|start|>assistant to=user<|message|>' (added 2026-10-05).
    "self_channel": re.compile(r"\bto=self\s*<\|message\|>"),
}

_OPENER = re.compile(
    r"^\s*(?:okay|ok|alright|hmm+|well|let me think|let'?s think|let me work|"
    r"first,|so,? i need|so,? the|i need to (?:figure|work) out|"
    r"i should (?:figure|think|start)|to answer this)",
    re.IGNORECASE,
)


def detect(text: str, default_on: bool = False) -> ThinkingInfo:
    if not text:
        return ThinkingInfo(default_on=default_on)
    markers = [name for name, pat in _MARKERS.items() if pat.search(text)]
    if _OPENER.match(text):
        markers.append("cot_opener")
    return ThinkingInfo(detected=bool(markers), markers=markers, default_on=default_on)
