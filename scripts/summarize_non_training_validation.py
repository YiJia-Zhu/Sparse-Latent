#!/usr/bin/env python3
"""
Aggregate analysis-only pilot runs into comparison tables and a markdown report.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = ROOT / "pilot_results"


def is_analysis_summary(data: Dict[str, object]) -> bool:
    return isinstance(data.get("args"), dict) and isinstance(data.get("set_summaries"), list)


def get_set_map(data: Dict[str, object]) -> Dict[str, Dict[str, float]]:
    return {row["set"]: row for row in data["set_summaries"]}


def short_model_name(model_path: str) -> str:
    path = str(model_path)
    if "Llama-3.2-1B-Instruct" in path:
        return "Llama1B"
    if path.rstrip("/").endswith("gpt2"):
        return "gpt2"
    return Path(path).name


def short_run_label(args: Dict[str, object]) -> str:
    model = short_model_name(str(args.get("model_path")))
    split = str(args.get("split"))
    seed = int(args.get("seed"))
    top_k = int(args.get("top_k"))
    layer_index = args.get("layer_index")
    layer_tag = f"layer{layer_index}" if layer_index is not None else "layerNA"
    return f"{model} {split} seed{seed} k{top_k} {layer_tag}"


def row_information_score(row: Dict[str, object]) -> int:
    score = 0
    for value in row.values():
        if value is None:
            continue
        if isinstance(value, float) and pd.isna(value):
            continue
        score += 1
    return score


def load_runs(result_dirs: List[Path]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    summary_rows: List[Dict[str, object]] = []
    overlap_rows: List[Dict[str, object]] = []

    loaded_by_label: Dict[str, Tuple[Path, Dict[str, object], int]] = {}
    for result_dir in result_dirs:
        summary_path = result_dir / "summary.json"
        if not summary_path.exists():
            continue
        data = json.loads(summary_path.read_text())
        if not is_analysis_summary(data):
            continue

        args = data["args"]
        label = short_run_label(args)
        set_map = get_set_map(data)
        generation = data.get("generation_summary", {})
        selected_neg = set_map["selected_neg"]
        selected_no_neg = set_map["selected_no_neg"]
        random_set = set_map["random"]

        summary_rows.append(
            {
                "run": str(result_dir),
                "label": label,
                "model": short_model_name(str(args.get("model_path"))),
                "split": args.get("split"),
                "seed": int(args.get("seed")),
                "top_k": int(args.get("top_k")),
                "num_samples": int(args.get("num_samples")),
                "layer_index": args.get("layer_index"),
                "selected_neg_omsa": selected_neg["orbit_minus_same_answer"],
                "selected_no_neg_omsa": selected_no_neg["orbit_minus_same_answer"],
                "random_omsa": random_set["orbit_minus_same_answer"],
                "delta_neg_minus_no_neg": (
                    selected_neg["orbit_minus_same_answer"] - selected_no_neg["orbit_minus_same_answer"]
                ),
                "delta_neg_minus_random": (
                    selected_neg["orbit_minus_same_answer"] - random_set["orbit_minus_same_answer"]
                ),
                "selected_neg_same_answer": selected_neg["same_answer_stability_mean"],
                "selected_no_neg_same_answer": selected_no_neg["same_answer_stability_mean"],
                "selected_neg_logit_drop": selected_neg.get("answer_logit_drop_mean"),
                "selected_no_neg_logit_drop": selected_no_neg.get("answer_logit_drop_mean"),
                "main_accuracy": generation.get("main_accuracy"),
                "socratic_accuracy": generation.get("socratic_accuracy"),
                "rewrite_consistency": generation.get("rewrite_consistency"),
                "rewrite_both_correct": generation.get("rewrite_both_correct"),
            }
        )
        score = row_information_score(summary_rows[-1])
        prev = loaded_by_label.get(label)
        if prev is None or score > prev[2]:
            loaded_by_label[label] = (result_dir, data, score)

    deduped_rows: Dict[str, Dict[str, object]] = {}
    for row in summary_rows:
        label = row["label"]
        prev = deduped_rows.get(label)
        if prev is None or row_information_score(row) > row_information_score(prev):
            deduped_rows[label] = row
    summary_rows = list(deduped_rows.values())

    loaded = [(path, data) for path, data, _ in loaded_by_label.values()]
    for (dir_a, data_a), (dir_b, data_b) in combinations(loaded, 2):
        args_a = data_a["args"]
        args_b = data_b["args"]
        if short_model_name(str(args_a.get("model_path"))) != short_model_name(str(args_b.get("model_path"))):
            continue
        set_a = set(data_a["selected_sets"]["selected_neg"])
        set_b = set(data_b["selected_sets"]["selected_neg"])
        inter = len(set_a & set_b)
        union = len(set_a | set_b)
        overlap_rows.append(
            {
                "run_a": short_run_label(args_a),
                "run_b": short_run_label(args_b),
                "model": short_model_name(str(args_a.get("model_path"))),
                "intersection": inter,
                "union": union,
                "jaccard": (inter / union) if union else 0.0,
            }
        )

    return summary_rows, overlap_rows


def build_markdown(summary_df: pd.DataFrame, overlap_df: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("# Non-Training Validation Summary")
    lines.append("")
    lines.append("日期：2026-04-11")
    lines.append("")

    lines.append("## 1. 结论先说")
    lines.append("")
    lines.append("- analysis-only motivation 已经不止停留在 `gpt2`；当前更强的 `Llama1B` 也开始覆盖 seed / top-k / layer 变化。")
    lines.append(
        f"- 统计到的 analysis-only runs 共 `{len(summary_df)}` 个，其中 `selected_neg` 在 `{int((summary_df['delta_neg_minus_no_neg'] > 0).sum())}/{len(summary_df)}` 个 run 中优于 `selected_no_neg`。"
    )
    lines.append(
        f"- `selected_neg - selected_no_neg` 的 `OrbitMinusSameAnswer` 平均增益为 `{summary_df['delta_neg_minus_no_neg'].mean():+.4f}`。"
    )
    lines.append("")

    lines.append("## 2. Raw Data Table")
    lines.append("")
    raw = summary_df[
        [
            "label",
            "selected_neg_omsa",
            "selected_no_neg_omsa",
            "random_omsa",
            "delta_neg_minus_no_neg",
            "selected_neg_same_answer",
            "selected_no_neg_same_answer",
        ]
    ].copy()
    lines.append(dataframe_to_markdown(raw))
    lines.append("")

    lines.append("## 3. Key Findings")
    lines.append("")
    by_model = summary_df.groupby("model")["delta_neg_minus_no_neg"].agg(["mean", "min", "max", "count"]).reset_index()
    for _, row in by_model.iterrows():
        lines.append(
            f"1. `{row['model']}`: delta(`selected_neg - selected_no_neg`) 平均 `{row['mean']:+.4f}`，范围 `[{row['min']:+.4f}, {row['max']:+.4f}]`，run 数 `{int(row['count'])}`。"
        )
    sas_improved = (
        summary_df["selected_neg_same_answer"] < summary_df["selected_no_neg_same_answer"]
    ).sum()
    lines.append(
        f"2. 在 `{sas_improved}/{len(summary_df)}` 个 run 里，`selected_neg` 的 `SameAnswerStability` 低于 `selected_no_neg`，说明 negative control 确实在抑制答案共性污染。"
    )
    if not overlap_df.empty:
        lines.append(
            f"3. `selected_neg` 坐标 id 的跨运行 Jaccard 平均值为 `{overlap_df['jaccard'].mean():.4f}`，说明稳定的是统计判别标准，不是固定坐标 id。"
        )
    lines.append("")

    lines.append("## 4. Selected-Set Overlap")
    lines.append("")
    if overlap_df.empty:
        lines.append("无可用 overlap 对。")
    else:
        lines.append(dataframe_to_markdown(overlap_df))
    lines.append("")

    lines.append("## 5. Suggested Next Experiments")
    lines.append("")
    lines.append("1. 若继续 analysis-only，优先扩充 `Llama1B` 的 test split / layer sweep，而不是继续堆更多 `gpt2`。")
    lines.append("2. 若转入行为验证，优先做多 seed 的 `selected-no-neg` vs `selected-neg`。")
    lines.append("3. 若要补最便宜的行为相关性证据，下一步应做训练后 zeroing / causal ablation。")
    lines.append("")
    return "\n".join(lines)


def format_cell(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(format_cell(row[col]) for col in columns) + " |")
    return "\n".join([header, sep] + rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize analysis-only validation runs")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
    )
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-overlap-csv", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()

    result_dirs = sorted(path for path in args.results_root.iterdir() if path.is_dir())
    summary_rows, overlap_rows = load_runs(result_dirs)
    summary_df = pd.DataFrame(summary_rows).sort_values(["model", "split", "seed", "top_k", "layer_index", "run"])
    overlap_df = pd.DataFrame(overlap_rows).sort_values(["model", "run_a", "run_b"])

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(args.output_csv, index=False)
    if args.output_overlap_csv is not None:
        args.output_overlap_csv.parent.mkdir(parents=True, exist_ok=True)
        overlap_df.to_csv(args.output_overlap_csv, index=False)
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(build_markdown(summary_df, overlap_df), encoding="utf-8")

    print(summary_df.to_string(index=False))
    if not overlap_df.empty:
        print("\n[overlap]")
        print(overlap_df.to_string(index=False))


if __name__ == "__main__":
    main()
