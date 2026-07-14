"""Token-level helpers for the generate-and-intervene loop.

The trigger match is whole-word and case-insensitive, decided at the token level with
one token of look-ahead so that e.g. " the" matches but " theory" (tokenized
" the"+"ory") does not.
"""

from __future__ import annotations

import math
import re

_LEADING_WORD = re.compile(r"^(\s*)([A-Za-z']+)(.*)$", re.DOTALL)


def piece(tokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id])


def token_word(text: str) -> tuple[bool, str | None, str]:
    """Split a decoded token into (has_leading_space, leading_alpha_word, trailing).

    Returns word=None if the token doesn't start with (optional space then) letters.
    DESIGN CHOICE (mention vs. use): a token led by punctuation (e.g. '"the') therefore
    never matches. This is intentional — quoted occurrences are the model *mentioning*
    the word (e.g. while reasoning "X must be a typo for \"the\""), and swapping mentions
    would make it impossible for a model to ever articulate the correct word, suppressing
    the `aware` signal. Quoted/mentioned uses of the trigger word are allowed to stand.
    """
    m = _LEADING_WORD.match(text)
    if not m:
        return (bool(text) and text[0].isspace(), None, text)
    return (bool(m.group(1)), m.group(2), m.group(3))


def _is_letter(ch: str) -> bool:
    return ch[:1].isalpha() or ch[:1] == "'"


def _boundary_after(tokenizer, toks: list[int], j: int) -> bool:
    """Is there a word boundary right after token j?

    Uses a PAIR decode so it works for tokenizers whose single-token decode drops the
    leading space (e.g. Yi: decode([t])='capital' but decode([prev,t])=' capital').
    """
    if j + 1 >= len(toks):
        return True
    single = piece(tokenizer, toks[j])
    pair = tokenizer.decode([toks[j], toks[j + 1]])
    suffix = pair[len(single):] if pair.startswith(single) else None
    if suffix is None:
        return True  # decode normalized things; assume a boundary rather than silently miss
    return not _is_letter(suffix)


def _boundary_before(tokenizer, toks: list[int], j: int) -> bool:
    """Is token j word-initial (a boundary right before it)?"""
    if j == 0:
        return False  # caller decides via is_first
    prev = piece(tokenizer, toks[j - 1])
    pair = tokenizer.decode([toks[j - 1], toks[j]])
    sep = pair[len(prev):] if pair.startswith(prev) else None
    if sep is None:
        return True
    if not _is_letter(sep):
        return True  # this token carries its own separator (leading space etc.)
    # This token's contribution starts directly with a letter. It is still word-initial
    # when the PREVIOUS token ends in a non-letter — e.g. '\n\n'+'The' (line start) or
    # '   '+'The' (bullet indent), where the whitespace lives in the previous token.
    # Missing this dropped every line/bullet-initial trigger (the sweep-v1 dose bug).
    # Only a letter-final previous token (' the'+'ory' = 'theory') glues the words.
    return not _is_letter(prev[-1:]) if prev else True


def is_trigger(tokenizer, toks: list[int], j: int, trigger: str, is_first: bool) -> bool:
    """Is token j a standalone whole-word occurrence of ``trigger``?

    Word boundaries are decided with pair-decodes, so this works whether the tokenizer's
    single-token decode keeps the leading space (Qwen/Llama: ' the') or strips it (Yi:
    'the'). The earlier single-token-only check silently missed every trigger on Yi.
    """
    lead_ws, word, trailing = token_word(piece(tokenizer, toks[j]))
    if word is None or word.lower() != trigger.lower():
        return False
    if trailing[:1].isalpha() or trailing[:1] == "'":  # more letters inside this token
        return False
    # word-initial: a space in this token, the very first token, or a boundary before it
    if not (lead_ws or is_first or _boundary_before(tokenizer, toks, j)):
        return False
    # word-final: a boundary after this token (pair-decode, space-agnostic)
    if not _boundary_after(tokenizer, toks, j):
        return False
    return True


def encode_replacement(
    tokenizer, replacement: str, leading_space: bool, capitalize: bool
) -> list[int]:
    """Encode the replacement, mirroring the trigger's spacing and capitalization."""
    word = replacement
    if capitalize and word:
        word = word[0].upper() + word[1:]
    text = (" " if leading_space else "") + word
    return tokenizer.encode(text, add_special_tokens=False)


def entropy_nats(logprob_dict) -> float | None:
    """Entropy (nats) over the returned top-k candidates, renormalized.

    This is a top-k approximation of the true next-token entropy: it under-counts mass
    in the tail, which matters most when the distribution is very flat. The exact
    per-token surprisal (-ln p of the chosen token) is the primary, exact signal; this
    entropy is a supplementary uncertainty proxy.
    """
    if not logprob_dict:
        return None
    lps = [lp.logprob for lp in logprob_dict.values()]
    m = max(lps)
    ps = [math.exp(x - m) for x in lps]
    z = sum(ps)
    probs = [p / z for p in ps]
    return float(-sum(p * math.log(p) for p in probs if p > 0.0))
