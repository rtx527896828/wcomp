# NOTES

Running lab notebook. Every experiment: hypothesis, success criterion, result (also failures). Dates are
2026. Protocol and contracts: DESIGN.md.

## 2026-11-07 Setup

- Environment: RTX 5080 16 GB (≈10 GB free), 61.6 GB RAM, torch 2.11.0+cu128, transformers 5.22.2 (uv, Python 3.12).
- Models: Qwen3.5-0.8B (primary; the task said 0.6B, which does not exist), Qwen3.5-2B, MiniCPM5-2B (Llama
  architecture, as an architecture-independent check).
- Qwen3.5 runs on reference PyTorch kernels (no fla / causal_conv1d on Windows): 0.2 s per 2048-token forward on
  the 0.8B after a 15 s warm-up.

## Phase 1: evaluation framework and baselines

(filled in by the build and run steps)
