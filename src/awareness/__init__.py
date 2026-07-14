"""Sleight of Word: do LLMs notice a covertly swapped word in their own output?"""

import os

# This box has the CUDA driver but no CUDA toolkit (nvcc), so FlashInfer can't JIT its
# sampling kernel. Force vLLM's native PyTorch sampler, which needs no compilation.
# Set before any vLLM import; vLLM's engine subprocesses inherit the environment.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

__version__ = "0.1.0"
