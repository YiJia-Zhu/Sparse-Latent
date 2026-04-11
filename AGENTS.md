# Repository Guidelines

## Project Structure & Module Organization
This repository is a lightweight research workspace, not a packaged library. Keep executable code in `scripts/`, structured data in `data/`, local benchmark corpora in `huggingface_datasets/`, and generated experiment outputs in `pilot_results/` and `refine-logs/`. Root-level Markdown files capture literature notes and planning; treat them as generated reports or working docs, not importable modules.

## Build, Test, and Development Commands
Use `python3` directly from the repo root.

- `python3 scripts/build_llm_latent_lit_notes.py`: fetches arXiv metadata and regenerates `data/latent_llm_papers.json` plus `LLM_LATENT_LITERATURE_NOTES.md`.
- `python3 scripts/build_core_latent_deep_review.py`: rebuilds `CORE_LATENT_LLM_DEEP_REVIEW.md` from the JSON dataset.
- `python3 scripts/pilot_negative_control_analysis.py --model-path /path/to/model --output-dir pilot_results/tmp_run`: runs the pilot analysis and writes CSV/JSON summaries.

Prefer writing new outputs to a fresh subdirectory under `pilot_results/` instead of overwriting prior runs.

## Coding Style & Naming Conventions
Follow existing Python conventions: 4-space indentation, type hints where practical, `Path` for filesystem paths, and small helper functions over monolithic scripts. Use `snake_case` for functions, variables, and file names; use `UPPER_CASE` for module-level constants such as `ROOT` or `DATA`. Keep scripts runnable via `if __name__ == "__main__":`.

No formatter or linter config is checked in. Match current PEP 8 style and keep imports grouped: standard library, third-party, local.

## Testing Guidelines
There is no formal `tests/` suite yet. Validate changes by running the affected script with a small, reproducible configuration, then inspect the generated artifacts. For analysis code, prefer smoke tests such as `--num-samples 8 --top-k 4` and confirm `summary.json`, `coordinate_scores.csv`, or Markdown outputs are regenerated without errors.

## Commit & Pull Request Guidelines
This snapshot does not include `.git` history, so no repository-specific commit convention can be inferred. Use short imperative commit subjects, for example `Add ARC pilot summary export`. In pull requests, include the goal, touched paths, exact commands run, and a brief note on generated artifacts. Attach screenshots only when Markdown rendering or report formatting materially changes.

## Data & Configuration Tips
Scripts assume local assets under `huggingface_datasets/` and `huggingface_models/`. Avoid hard-coding alternate absolute paths in new code; expose them as CLI flags with sensible defaults instead.

## GPU 环境

- 这台机器有直接 GPU 访问（不需要 SSH）
- GPU：2x NVIDIA GeForce RTX 4090 49140MiB (可以使用nvidia-smi查看空余，目前id 0、2卡空余)
- 实验环境：`SIMCoT`（Python 3.11.4 + PyTorch）
- 激活前任何 Python 命令：`conda activate SIMCoT`（uv, conda 等）
- 代码目录：`/storage/zyj_data/latent_idea`