#!/usr/bin/env python3
"""
Summarize one or more selective invariance experiment runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def load_summary(path: Path) -> Dict[str, object]:
    summary_path = path / "summary.json" if path.is_dir() else path
    data = json.loads(summary_path.read_text())
    metrics = data.get("eval_metrics", {})
    return {
        "run": str(summary_path.parent),
        "variant": data.get("variant"),
        "model_path": data.get("model_path"),
        "train_pairs": data.get("train_pairs"),
        "scoring_pairs": data.get("scoring_pairs"),
        "eval_pairs": data.get("eval_pairs"),
        "mean_accuracy": metrics.get("mean_accuracy"),
        "rewrite_consistency": metrics.get("rewrite_consistency"),
        "accuracy_conditioned_rewrite_consistency": metrics.get("accuracy_conditioned_rewrite_consistency"),
        "main_accuracy": metrics.get("main_accuracy"),
        "socratic_accuracy": metrics.get("socratic_accuracy"),
        "mean_prompt_tokens": metrics.get("mean_prompt_tokens"),
        "mean_prompt_tokens_main": metrics.get("mean_prompt_tokens_main"),
        "mean_prompt_tokens_socratic": metrics.get("mean_prompt_tokens_socratic"),
        "mean_generated_tokens": metrics.get("mean_generated_tokens"),
        "mean_generated_tokens_main": metrics.get("mean_generated_tokens_main"),
        "mean_generated_tokens_socratic": metrics.get("mean_generated_tokens_socratic"),
        "orbit_variance_gap": data.get("diagnostics", {}).get("orbit_variance_gap"),
        "gate_enabled": data.get("gate", {}).get("enabled"),
        "gate_center": data.get("gate", {}).get("center"),
        "gate_temperature": data.get("gate", {}).get("temperature"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize selective invariance runs")
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    rows: List[Dict[str, object]] = [load_summary(path) for path in args.runs]
    df = pd.DataFrame(rows)
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
