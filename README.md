# Sleight of Word

A testing harness that measures whether open-source chat LLMs **notice when a word in
their own answer gets swapped out mid-generation**, and how they react.

We call the technique **"sleight of word"**, covertly swapping a word in the model's own
output. The framing nods to the *Mandela effect* (false memory of one's own past). We cite
it as inspiration in the write-up, not as the technique's name.

## The idea

The user asks a fixed question, e.g. `what is the capital of France?`, **never altered.**
The assistant answers **token by token** (greedy, up to a configurable budget). Whenever
the assistant emits a configurable **trigger word** (`the`), we **replace that token with
a random word** (`umbrella`) in its own running output, and let it keep generating from the
now-corrupted context. The swap is content-triggered (fires when the model *says* `the`),
whole-word, case-insensitive, on **every** occurrence.

We then ask, two ways, whether the model noticed:

1. **Surprisal + entropy (quantitative).** For every token the model produces *after* a
   swap, we record its surprisal (`-ln p` of the chosen token, exact) and the entropy of
   its next-token distribution (top-k estimate). We compare against a **paired clean run**
   (same question, no swap). A spike right after the swap = the model is disrupted by the
   odd word in its own context.
2. **Jury of judges (qualitative).** A jury of **three LLM judges from distinct
   lineages** (Gemma 4 26B-A4B at fp8, Qwen3.6-27B, Llama 3.3 70B AWQ) reads each
   corrupted answer. A label is assigned when **at least two jurors agree**. Each judge
   assigns four **independent** boolean labels, `flagged` (remarks a word is out of
   place), `corrected` (still delivers the right answer), `derailed` (incoherence or
   fixation on the odd word), and the strict `aware` (explicitly states a word was
   substituted). A reply with none of these counts as `ignored`. Using three lineages
   counters the self-preference bias of any single judge.

## Why this is faithful and still efficient

The swap lives in the sequence the model *produces*, not in the prompt. The query is
never touched. We implement the loop by **generate → find the first trigger → splice in
the replacement → regenerate the rest from the corrupted context → repeat**. Because a
transformer only sees tokens (not whether it generated them), regenerating from the
spliced sequence yields exactly the surprisal of what the model says next given the
corruption, identical to a literal token-by-token intercept, but far fewer forward
passes (one regenerate per trigger occurrence).

## Hardware

Sized for a single **48 GB A6000**. Targets run smallest→largest. Models up to ~22B run at bf16, larger
ones via 4-bit (AWQ) or bitsandbytes. The judges run as a **separate phase**, one at a
time after the targets are freed, so each gets the whole GPU.

This box has the CUDA driver but no toolkit (`nvcc`), so the package forces vLLM's native
sampler (`VLLM_USE_FLASHINFER_SAMPLER=0`, set in `awareness/__init__.py`) to avoid
runtime kernel JIT.

## Usage

```bash
uv sync                       # installs vLLM + deps (large, one-time)

# Phase 1, chat models answer, trigger word swapped in their replies, measured
uv run awareness generate --models configs/models.yaml --experiment configs/experiment.yaml
# (subset:  --only qwen2.5-0.5b,llama3.1-8b)

# Phase 2, each judge reads every model's reactions (run once per judge, resumable)
uv run python scripts/panel_judge.py --judge-model google/gemma-4-26B-A4B-it \
    --judge-tag gemma26 --run results/runs/<run> --quant fp8
# ... repeat with the other two judges (see scripts/panel_judge.py --help) ...

# Combine, majority vote (label true iff >=2 jurors agree), written into the judged files
uv run python scripts/panel_combine.py --run results/runs/<run>

# Report + interactive viewer, per-model label distribution, mean Δsurprisal, Δentropy
uv run python scripts/report_sweep.py
uv run python scripts/make_viewer.py     # writes index.html + one page per model
```

## Layout

| path | role |
|------|------|
| `configs/models.yaml` | target zoo + vLLM kwargs per model |
| `configs/experiment.yaml` | fixed questions, trigger→replacement pairs, judge config |
| `configs/leaderboard.yaml` | the full benchmark: 100 questions × 101 substituted words (10,100 trials/model) |
| `src/awareness/tokens.py` | whole-word trigger detection, replacement encoding, entropy |
| `src/awareness/runner.py` | phase 1: generate-and-intervene loop + metrics |
| `src/awareness/judge.py` | judge prompt + multi-label verdict parsing |
| `scripts/panel_judge.py` | phase 2: one juror reads all models' reactions (chunked, resumable) |
| `scripts/panel_combine.py` | merge the jurors' sidecars into majority verdicts |
| `scripts/make_viewer.py` | self-contained HTML leaderboard (index + per-model pages) |
| `src/awareness/report.py` | aggregate to a table |
| `tests/test_logic.py` | offline checks (no GPU) |

## Config knobs

- **pairs**, a list of `{trigger, replacement}`. Each pair is its own trial. Default
  `the → umbrella`.
- **max_new_tokens** (256), **post_window** (tokens after a swap to average),
  **top_logprobs** (k for the entropy estimate).
- **system_prompt**, optional. Empty by default so models behave naturally.

## Notes / open choices

- **Entropy is a top-k estimate** (renormalized over the returned candidates). It
  under-counts tail mass when the distribution is very flat. The exact per-token
  surprisal is the primary signal. Entropy is supplementary.
- **Greedy decoding** for reproducibility. If a model never emits the trigger within the
  budget, that trial has no swap (recorded, not judged).
