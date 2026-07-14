"""Config objects loaded from YAML (see ``configs/``)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class ModelSpec(BaseModel):
    """One target chat model and the vLLM kwargs needed to fit it on a 48 GB A6000."""

    name: str  # short label used in results/tables
    model: str  # HF repo id or local path
    quantization: Optional[str] = None  # e.g. "awq", "gptq", "fp8", "bitsandbytes"
    load_format: Optional[str] = None  # auto-set to "bitsandbytes" when quant is bnb
    dtype: str = "auto"
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.90
    trust_remote_code: bool = False
    # extra kwargs passed straight through to vllm.LLM(...) — e.g.
    # {limit_mm_per_prompt: {image: 0}} to disable an unused vision tower whose
    # processor crashes at profiling (Magistral-Small-2509, 2026-07-08)
    extra: dict = Field(default_factory=dict)


class PromptSpec(BaseModel):
    """A fixed user question. Never altered -- the swap happens in the answer."""

    id: str
    query: str


class Pair(BaseModel):
    """A trigger word the assistant might emit and the word we swap it for.

    Match is whole-word and case-insensitive; every occurrence within the token
    budget is replaced. Each pair is run as its own trial.
    """

    trigger: str
    replacement: str


class JudgeConfig(BaseModel):
    # Gemma 4 26B-A4B: MoE, 25.2B total / 3.8B active. Apache 2.0, not gated.
    model: str = "google/gemma-4-26B-A4B-it"
    quantization: Optional[str] = "fp8"  # ~25 GB; fits the A6000 alone in phase 2
    dtype: str = "auto"
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.90
    trust_remote_code: bool = False
    temperature: float = 0.0
    max_tokens: int = 512


class ExperimentConfig(BaseModel):
    prompts: list[PromptSpec]
    pairs: list[Pair] = Field(
        default_factory=lambda: [Pair(trigger="the", replacement="umbrella")]
    )
    system_prompt: str = ""

    # assistant decoding (always greedy / temperature 0 for reproducibility)
    max_new_tokens: int = 256
    top_logprobs: int = 20  # top-k kept per step; used for the entropy estimate
    post_window: int = 8  # tokens after each substitution to average for the signal

    judge: JudgeConfig = JudgeConfig()


def load_models(path: str | Path) -> list[ModelSpec]:
    data = yaml.safe_load(Path(path).read_text())
    return [ModelSpec(**m) for m in data["models"]]


def load_experiment(path: str | Path) -> ExperimentConfig:
    data = yaml.safe_load(Path(path).read_text())
    return ExperimentConfig(**data)
