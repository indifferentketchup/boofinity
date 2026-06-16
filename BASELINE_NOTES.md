# Baseline venv notes (batch 1)

Date: 2026-06-12. Branch `ik-main` at tag `0.0.77` (commit 4355fef).

> **Historical record.** This file documents how the frozen `.venv-baseline` / `baseline-freeze.txt`
> reference was built at batch 1. The fork has since advanced (current package version `0.1.0`, batch 2
> dependency modernization and the `boofinity` rebrand merged, plus a `.venv-batch2` for newer dep
> resolutions). The facts below describe batch 1 as captured and are not updated in place. For current
> commands and state see `CLAUDE.md` (local-only dev guide); for deployment see [`DEPLOY.md`](./DEPLOY.md).

## How the venv was built

```
python3 -m venv .venv-baseline                                   # Python 3.12.3
.venv-baseline/bin/pip install -U pip                            # pip 26.1.2
.venv-baseline/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv-baseline/bin/pip install -e "libs/boofinity[torch,server,logging]"
.venv-baseline/bin/pip freeze > baseline-freeze.txt
```

The editable install succeeded with no source or metadata fixes. `import boofinity`
reports 0.0.77.

## Deviations from a plain `pip install -e libs/boofinity[all]`

1. **torch from the CPU wheel index, not PyPI.** This box has no CUDA and the task
   forbids requiring it. Upstream's own pyproject declares the
   `https://download.pytorch.org/whl/cpu` source ("used for monkey-patching cpu only"),
   so this is the upstream-sanctioned CPU path, not a fork-specific hack.
   Resolved: `torch==2.12.0+cpu`.
2. **Extras `[torch,server,logging]` instead of `[all]`.** The deployment target
   (bge-m3 plus a reranker served over HTTP via the torch engine) only exercises these.
   `[all]` would add ctranslate2, optimum/onnxruntime, colpali-engine, timm,
   torchvision, soundfile, diskcache; those backends are recon subjects this batch,
   not runtime requirements, and keeping them out gives a cleaner import-time baseline
   for the torch-only path.

## Key resolved versions (full list in baseline-freeze.txt)

| package | resolved |
|---------|----------|
| torch | 2.12.0+cpu |
| transformers | 4.57.6 |
| sentence-transformers | 3.4.1 |
| numpy | 1.26.4 (pyproject pins <2 "for onnx") |
| fastapi | 0.136.3 |
| pydantic | 2.13.4 |
| uvicorn | 0.32.1 (pyproject caps ^0.32.0) |

Because upstream pins are loose, "current pinned deps" already resolve to versions at
or past the batch-2 modernization targets (torch>=2.5, transformers>=4.57). The parity
baseline captured against this venv therefore already validates the modern-dep stack
on the torch path.
