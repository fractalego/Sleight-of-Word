"""Pydantic records that flow through the pipeline and get serialized to JSON.

One ``TrialResult`` per (model, query, pair). Phase 1 (the runner) fills everything
except ``judge``; phase 2 (the judge) fills ``judge`` in place.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

JudgeLabel = Literal["ignored", "corrected", "flagged", "derailed", "unknown"]


class TokenStep(BaseModel):
    """One decoded assistant token. surprisal/entropy are None for injected tokens."""

    index: int  # position within the assistant turn (0-based)
    token_id: int
    text: str
    surprisal: Optional[float] = None  # -ln p of this token (nats); None if substituted
    entropy: Optional[float] = None  # entropy over the top-k next-token dist (nats, approx)
    substituted: bool = False  # True for the injected replacement token(s)


class TrialMetrics(BaseModel):
    num_substitutions: int = 0
    # baselines: mean over the clean (no-intervention) run
    baseline_surprisal: Optional[float] = None
    baseline_entropy: Optional[float] = None
    # post-substitution: mean over the window of tokens right after each swap
    post_sub_surprisal: Optional[float] = None
    post_sub_entropy: Optional[float] = None
    # the headline "noticing" signals
    delta_surprisal: Optional[float] = None  # post_sub - baseline
    delta_entropy: Optional[float] = None


class TextEvidence(BaseModel):
    """Judge-independent textual signs the model remarked on the swap."""

    flagged: bool = False  # True on a strong self-correction / meta marker
    aware: bool = False  # True when it names the *substitution* itself (a word was replaced)
    markers: list[str] = Field(default_factory=list)  # matched marker names ("weak:" tagged)
    snippet: str = ""  # text around the first match


class ThinkingInfo(BaseModel):
    """Whether reasoning/thinking actually appeared in the output."""

    detected: bool = False  # thinking text appeared in the answer
    markers: list[str] = Field(default_factory=list)  # which signals fired
    default_on: bool = False  # model's chat template opens a thinking channel by default


class JudgeVerdict(BaseModel):
    # MULTI-LABEL (2026-07-10): a reply can be several of these at once, so they are
    # independent booleans, not one exclusive label. E.g. a reply can flag the odd word
    # AND still give the right answer (corrected) AND code-switch (derailed).
    #   flagged   -- explicitly remarked a word was strange / wrong / out of place
    #   corrected -- still delivered the correct answer despite the swap
    #   derailed  -- became incoherent, repetitive, off-topic, or fixated on the odd word
    #   aware     -- (strict) explicitly stated a word was SUBSTITUTED (names the mechanism)
    # "ignored" is the derived absence of all four (see .ignored). All Optional so
    # verdicts produced before a field existed read back as None ("not assessed").
    flagged: Optional[bool] = None
    corrected: Optional[bool] = None
    derailed: Optional[bool] = None
    aware: Optional[bool] = None
    # single-label field retained for reading legacy (pre-2026-07-10) judged files.
    label: Optional[JudgeLabel] = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: str = ""
    judge_model: str = ""

    def has(self, kind: str) -> bool:
        """Effective boolean for 'flagged'/'corrected'/'derailed', back-compatible with
        the legacy single ``label`` field (pre-2026-07-10 judged files)."""
        v = getattr(self, kind, None)
        if v is not None:
            return bool(v)
        return self.label == kind

    @property
    def ignored(self) -> bool:
        return not (self.has("flagged") or self.has("corrected")
                    or self.has("derailed") or bool(self.aware))


class TrialResult(BaseModel):
    # identity
    trial_id: str
    model: str
    prompt_id: str
    query: str  # the fixed user question
    trigger: str
    replacement: str

    # phase-1 measurements
    clean_text: str = ""  # assistant answer with no intervention
    reaction_text: str = ""  # assistant answer with every trigger swapped (judge reads this)
    clean_steps: list[TokenStep] = Field(default_factory=list)
    steps: list[TokenStep] = Field(default_factory=list)  # the intervention run
    substitution_indices: list[int] = Field(default_factory=list)  # step indices of swaps
    metrics: TrialMetrics = Field(default_factory=TrialMetrics)
    text_evidence: TextEvidence = Field(default_factory=TextEvidence)
    thinking: ThinkingInfo = Field(default_factory=ThinkingInfo)
    error: Optional[str] = None

    # phase-2, filled by the judge
    judge: Optional[JudgeVerdict] = None
