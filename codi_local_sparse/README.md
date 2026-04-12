# Local CODI Official Sparse Copy

This directory is the local working copy for the official-CODI-based sparse distillation experiments.

Scope:

- `src/model.py`: official CODI path with sparse distill support
- `train.py`: local training entry
- `evaluate_local_codi.py`: local evaluation entry
- `run_sparse_codi_official.sh`: single-run launcher
- `run_sparse_codi_official_suite.sh`: 3-way suite launcher

Rule:

- Future edits and runs should use this local copy under the current repository
- Paths in launcher scripts are resolved relative to the repository root via `env.sh`

Current selector source:

- `pilot_results/round2_selected_neg_llama1b_gpu4/summary.json`

Default assets:

- base model: `huggingface_models/Llama-3.2-1B-Instruct`
- pretrained CODI: `huggingface_models/CODI-llama3.2-1b-Instruct`
- train dataset default: `load_dataset("zen-E/GSM8k-Aug")`
- optional local mirrors:
  `huggingface_datasets/zen-E__GSM8k-Aug/gsm8k_aug_train.json`
  `huggingface_datasets/zen-E__GSM8k-Aug/gsm8k_test.json`
  `huggingface_datasets/zen-E__GSM8k-Aug-NL/gsm8k_aug_nl.json`
- GSM8K test: `huggingface_datasets/gsm8k/main/test-00000-of-00001.parquet`

Training semantics used in this local copy:

- `pretrained CODI` is a comparison baseline and is evaluated directly from `huggingface_models/CODI-llama3.2-1b-Instruct`
- `full_state / sparse_no_neg / sparse_neg` are training runs that start from the plain base LLaMA checkpoint by default
- `restore_from` is reserved for explicit resume / warm-start requests and is not enabled by default in the launchers

Environment:

- default conda env name is controlled by `CONDA_ENV_NAME`
- current default is `SIMCoT`
