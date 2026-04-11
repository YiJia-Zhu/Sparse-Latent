#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


PREFERRED_TAG_GROUPS = [
    ("loss", ["loss", "train/loss"]),
    ("ce_loss", ["ce_loss", "train/ce_loss"]),
    ("distill_loss", ["distill_loss", "train/distill_loss"]),
    ("ref_ce_loss", ["ref_ce_loss", "train/ref_ce_loss"]),
]


def load_scalars_from_events(event_dir: Path) -> Dict[str, List[Tuple[int, float]]]:
    if not event_dir.exists():
        return {}
    accumulator = EventAccumulator(str(event_dir))
    accumulator.Reload()
    tags = set(accumulator.Tags().get("scalars", []))
    result: Dict[str, List[Tuple[int, float]]] = {}
    for out_name, candidates in PREFERRED_TAG_GROUPS:
        for tag in candidates:
            if tag in tags:
                result[out_name] = [(item.step, item.value) for item in accumulator.Scalars(tag)]
                break
    return result


def load_scalars_from_trainer_state(trainer_state_path: Path) -> Dict[str, List[Tuple[int, float]]]:
    if not trainer_state_path.exists():
        return {}
    data = json.loads(trainer_state_path.read_text())
    scalars: Dict[str, List[Tuple[int, float]]] = {name: [] for name, _ in PREFERRED_TAG_GROUPS}
    for row in data.get("log_history", []):
        step = row.get("step")
        if step is None:
            continue
        for name, _ in PREFERRED_TAG_GROUPS:
            if name in row and isinstance(row[name], (int, float)):
                scalars[name].append((int(step), float(row[name])))
    return {k: v for k, v in scalars.items() if v}


def choose_scalars(event_dir: Path, trainer_state_path: Path) -> Dict[str, List[Tuple[int, float]]]:
    event_scalars = load_scalars_from_events(event_dir)
    if event_scalars:
        return event_scalars
    return load_scalars_from_trainer_state(trainer_state_path)


def plot_scalars(scalars: Dict[str, List[Tuple[int, float]]], output_path: Path, title: str) -> None:
    plt.figure(figsize=(10, 6))
    plotted = 0
    for name, points in scalars.items():
        if not points:
            continue
        xs = [step for step, _ in points]
        ys = [value for _, value in points]
        plt.plot(xs, ys, label=name)
        plotted += 1
    if plotted == 0:
        raise ValueError("No scalar series found for plotting.")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export TensorBoard/trainer_state loss plot")
    parser.add_argument("--event-dir", type=Path, required=True)
    parser.add_argument("--trainer-state", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--title", type=str, default="Training Loss")
    args = parser.parse_args()

    scalars = choose_scalars(args.event_dir, args.trainer_state)
    if not scalars:
        raise SystemExit(f"No scalar data found in {args.event_dir} or {args.trainer_state}")
    plot_scalars(scalars, args.output_path, args.title)
    summary = {
        "event_dir": str(args.event_dir),
        "trainer_state": str(args.trainer_state),
        "output_path": str(args.output_path),
        "series": sorted(scalars.keys()),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
