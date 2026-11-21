# Validation record

Validation date: 2026-10-12 (Asia/Shanghai). Revision: 0.4.0 local CUDA migration.

## Completed in the controlled build environment

- All Python files passed `compileall`.
- Eighteen offline/synthetic tests pass when pytest uses a writable project-local temp directory.
- Tests cover future missing exit quotes, date-local normalization, split purging, historical industry intervals, degree-preserving rewiring, anchored seed scheduling, self-loop topology, historical stamp duty, dataset-specific keys, paired block bootstrap/Holm output, suspended sale carry-forward, one-price upper-limit buy failure, one-price lower-limit sell failure, evaluation exports, and the synthetic factor-to-Ridge path.
- Both Local and Colab notebooks pass nbformat validation, Python parsing for every code cell, and in-memory HTML rendering.
- Embedded Colab source archive decodes successfully.
- Delivered text/Notebook files were scanned for fragments of the credential pasted in chat; none were found.

## Environment actually visible to this build

The sandbox reported Windows build 26200 and Python 3.11.15, with roughly 66.7 GB free on C: and 94.4 GB on D:. `nvidia-smi` was not available and the sandbox Python did not contain PyTorch/PyG. This is not evidence about the user’s RTX 4070 Laptop GPU.

## Not executed here

- NVIDIA driver, actual GPU model/VRAM, CUDA PyTorch, PyG CUDA kernels, AMP, and peak VRAM on the target laptop.
- Live TuShare download, real-data revised SMOKE_TEST, checkpoint-resume timing, or FULL_RUN.
- The requested `research_results_bundle (1).zip` was not present among accessible attachments; the migration used the latest available project source and the supplied review text. An exact bundle-to-bundle diff remains unavailable.

Therefore Stage A/B source work and Stage C offline validation are complete, but the target-laptop GPU/real-data preflight remains required. No new empirical result or superiority claim is made.
