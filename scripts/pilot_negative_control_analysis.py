#!/usr/bin/env python3
"""
Few-shot pilot for negative-controlled orbit coordinate selection.

This script is intentionally analysis-first. It does not train the proposed
regularizer yet. Instead, it checks whether the core signal exists:

1. Orbit-stable coordinates across equivalent rationales
2. Output-necessary coordinates for the correct answer token
3. Same-answer-generic coordinates that should be filtered out

It uses local Hugging Face model checkpoints and local GSM8K `main` / `socratic`
splits as a cheap orbit proxy.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = ROOT / "huggingface_datasets"
DEFAULT_MODEL_PATH = ROOT / "huggingface_models" / "gpt2"
DEFAULT_OUTPUT_DIR = ROOT / "pilot_results" / "orbit_negative_control"


def normalize_answer(answer: str) -> str:
    answer = answer.strip()
    answer = answer.replace(",", "").replace("$", "")
    try:
        value = float(answer)
        if value == int(value):
            return str(int(value))
        return f"{value:.6f}".rstrip("0").rstrip(".")
    except ValueError:
        return answer.lower()


def split_gsm8k_answer(answer_text: str) -> Tuple[str, str]:
    match = re.search(r"^(.*?)(?:\n)?####\s*(.+?)\s*$", answer_text, flags=re.S)
    if match:
        rationale = match.group(1).strip()
        final_answer = normalize_answer(match.group(2).strip())
        return rationale, final_answer

    lines = answer_text.strip().splitlines()
    rationale = "\n".join(lines[:-1]).strip() if len(lines) > 1 else answer_text.strip()
    numbers = re.findall(r"-?\d+\.?\d*", answer_text)
    final_answer = normalize_answer(numbers[-1]) if numbers else ""
    return rationale, final_answer


def build_prompt(question: str, rationale: str) -> str:
    return (
        "Solve the following math problem.\n\n"
        f"Question: {question}\n\n"
        f"Solution: {rationale}\n\n"
        "Final answer:"
    )


def extract_last_number(text: str) -> str:
    numbers = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    if not numbers:
        return text.strip()
    return normalize_answer(numbers[-1])


@dataclass
class OrbitPair:
    question: str
    rationale_main: str
    rationale_socratic: str
    answer: str


@dataclass
class ExampleFeatures:
    answer: str
    question: str
    prompt_main: str
    prompt_socratic: str
    hidden_main: np.ndarray
    hidden_socratic: np.ndarray
    target_token_id: int
    generation_main: Optional[str] = None
    generation_socratic: Optional[str] = None
    pred_main: Optional[str] = None
    pred_socratic: Optional[str] = None


def load_gsm8k_orbit_pairs(
    dataset_root: Path,
    split: str,
    max_pairs: int,
    seed: int,
) -> List[OrbitPair]:
    main_path = dataset_root / "gsm8k" / "main" / f"{split}-00000-of-00001.parquet"
    soc_path = dataset_root / "gsm8k" / "socratic" / f"{split}-00000-of-00001.parquet"

    main_df = pd.read_parquet(main_path)
    soc_df = pd.read_parquet(soc_path)

    if len(main_df) != len(soc_df):
        raise ValueError("gsm8k main/socratic splits have different lengths")

    pairs: List[OrbitPair] = []
    for (_, main_row), (_, soc_row) in zip(main_df.iterrows(), soc_df.iterrows()):
        if main_row["question"] != soc_row["question"]:
            continue

        rationale_main, answer_main = split_gsm8k_answer(main_row["answer"])
        rationale_soc, answer_soc = split_gsm8k_answer(soc_row["answer"])

        if not answer_main or answer_main != answer_soc:
            continue

        pairs.append(
            OrbitPair(
                question=main_row["question"],
                rationale_main=rationale_main,
                rationale_socratic=rationale_soc,
                answer=answer_main,
            )
        )

    rng = random.Random(seed)
    rng.shuffle(pairs)
    return pairs[:max_pairs]


def choose_device(requested: Optional[str]) -> str:
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_local_model(model_path: Path, device: str):
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = None
    if device == "cuda":
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    if getattr(model, "generation_config", None) is not None:
        # Some instruct checkpoints carry sampling-only defaults that trigger
        # noisy warnings under greedy decoding.
        for attr in ("temperature", "top_p", "top_k", "typical_p"):
            if hasattr(model.generation_config, attr):
                setattr(model.generation_config, attr, None)
    model.to(device)
    model.eval()
    return tokenizer, model


def get_target_token_id(tokenizer, answer: str) -> int:
    candidates = [
        " " + answer,
        answer,
    ]
    for candidate in candidates:
        token_ids = tokenizer.encode(candidate, add_special_tokens=False)
        if token_ids:
            return int(token_ids[0])
    raise ValueError(f"Could not tokenize answer: {answer!r}")


@torch.no_grad()
def collect_features(
    model,
    tokenizer,
    pairs: Sequence[OrbitPair],
    device: str,
    max_length: int,
    layer_index: int,
    do_generation_eval: bool,
    generation_tokens: int,
) -> List[ExampleFeatures]:
    features: List[ExampleFeatures] = []

    for pair in pairs:
        prompt_main = build_prompt(pair.question, pair.rationale_main)
        prompt_soc = build_prompt(pair.question, pair.rationale_socratic)
        target_token_id = get_target_token_id(tokenizer, pair.answer)

        hidden_vectors = {}
        generations: Dict[str, Optional[str]] = {"main": None, "soc": None}
        preds: Dict[str, Optional[str]] = {"main": None, "soc": None}

        for tag, prompt in [("main", prompt_main), ("soc", prompt_soc)]:
            enc = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            outputs = model(**enc, output_hidden_states=True, use_cache=False, return_dict=True)
            hidden_state = outputs.hidden_states[layer_index]
            hidden_vectors[tag] = hidden_state[0, -1, :].float().cpu().numpy()

            if do_generation_eval:
                gen_ids = model.generate(
                    **enc,
                    max_new_tokens=generation_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                full_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                generated_suffix = full_text[len(prompt):].strip()
                generations[tag] = generated_suffix
                preds[tag] = extract_last_number(generated_suffix)

        features.append(
            ExampleFeatures(
                answer=pair.answer,
                question=pair.question,
                prompt_main=prompt_main,
                prompt_socratic=prompt_soc,
                hidden_main=hidden_vectors["main"],
                hidden_socratic=hidden_vectors["soc"],
                target_token_id=target_token_id,
                generation_main=generations["main"],
                generation_socratic=generations["soc"],
                pred_main=preds["main"],
                pred_socratic=preds["soc"],
            )
        )

    return features


def zscore_basis(examples: Sequence[ExampleFeatures]) -> Tuple[np.ndarray, np.ndarray]:
    stacked = np.stack(
        [ex.hidden_main for ex in examples] + [ex.hidden_socratic for ex in examples],
        axis=0,
    )
    mu = stacked.mean(axis=0)
    sigma = stacked.std(axis=0)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    return mu, sigma


def closeness(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.abs(a - b))


def compute_same_answer_pairs(
    examples: Sequence[ExampleFeatures],
    max_pairs_per_answer: int,
    seed: int,
) -> List[Tuple[int, int]]:
    answer_to_indices: Dict[str, List[int]] = {}
    for idx, ex in enumerate(examples):
        answer_to_indices.setdefault(ex.answer, []).append(idx)

    rng = random.Random(seed)
    pairs: List[Tuple[int, int]] = []
    for indices in answer_to_indices.values():
        if len(indices) < 2:
            continue
        local_pairs = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                local_pairs.append((indices[i], indices[j]))
        rng.shuffle(local_pairs)
        pairs.extend(local_pairs[:max_pairs_per_answer])
    return pairs


def compute_coordinate_statistics(
    model,
    examples: Sequence[ExampleFeatures],
    mu: np.ndarray,
    sigma: np.ndarray,
    max_pairs_per_answer: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    z_main = np.stack([(ex.hidden_main - mu) / sigma for ex in examples], axis=0)
    z_soc = np.stack([(ex.hidden_socratic - mu) / sigma for ex in examples], axis=0)
    z_mean = 0.5 * (z_main + z_soc)

    orbit_stability = closeness(z_main, z_soc).mean(axis=0)

    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise ValueError("Model does not expose output embeddings")
    weight = lm_head.weight.detach().float().cpu().numpy()

    necessity_terms = []
    for ex in examples:
        for hidden in (ex.hidden_main, ex.hidden_socratic):
            contrib = hidden * weight[ex.target_token_id]
            necessity_terms.append(np.maximum(contrib, 0.0))
    output_necessity = np.stack(necessity_terms, axis=0).mean(axis=0)

    same_pairs = compute_same_answer_pairs(examples, max_pairs_per_answer=max_pairs_per_answer, seed=seed)
    if same_pairs:
        sas_terms = []
        for i, j in same_pairs:
            sas_terms.append(closeness(z_mean[i], z_mean[j]))
        same_answer_stability = np.stack(sas_terms, axis=0).mean(axis=0)
    else:
        same_answer_stability = np.zeros_like(orbit_stability)

    high_activation = np.abs(z_mean).mean(axis=0)

    return {
        "orbit_stability": orbit_stability,
        "output_necessity": output_necessity,
        "same_answer_stability": same_answer_stability,
        "high_activation": high_activation,
        "z_main": z_main,
        "z_soc": z_soc,
    }


def rank_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(values))
    return ranks.astype(np.float32)


def select_coordinate_sets(
    stats: Dict[str, np.ndarray],
    top_k: int,
    seed: int,
    same_answer_filter_quantile: float,
) -> Dict[str, np.ndarray]:
    orbit_stability = stats["orbit_stability"]
    output_necessity = stats["output_necessity"]
    same_answer_stability = stats["same_answer_stability"]
    high_activation = stats["high_activation"]

    no_neg_score = -rank_desc(orbit_stability) - rank_desc(output_necessity)
    selected_no_neg = np.argsort(-no_neg_score)[:top_k]

    threshold = float(np.quantile(same_answer_stability, same_answer_filter_quantile))
    allowed = np.where(same_answer_stability <= threshold)[0]
    if len(allowed) < top_k:
        allowed = np.arange(len(same_answer_stability))
    filtered_score = -rank_desc(orbit_stability) - rank_desc(output_necessity)
    filtered_order = allowed[np.argsort(-filtered_score[allowed])]
    selected_neg = filtered_order[:top_k]

    high_activation_set = np.argsort(-high_activation)[:top_k]

    rng = np.random.default_rng(seed)
    random_set = np.sort(rng.choice(len(orbit_stability), size=top_k, replace=False))

    return {
        "selected_neg": selected_neg,
        "selected_no_neg": selected_no_neg,
        "high_activation": high_activation_set,
        "random": random_set,
        "same_answer_threshold": np.array([threshold], dtype=np.float32),
    }


def summarize_coordinate_set(
    name: str,
    indices: np.ndarray,
    stats: Dict[str, np.ndarray],
    answer_logit_drop: float,
) -> Dict[str, float]:
    orbit = stats["orbit_stability"][indices]
    necessity = stats["output_necessity"][indices]
    sas = stats["same_answer_stability"][indices]
    high_act = stats["high_activation"][indices]
    return {
        "set": name,
        "num_coords": int(len(indices)),
        "orbit_stability_mean": float(np.mean(orbit)),
        "output_necessity_mean": float(np.mean(necessity)),
        "same_answer_stability_mean": float(np.mean(sas)),
        "high_activation_mean": float(np.mean(high_act)),
        "orbit_minus_same_answer": float(np.mean(orbit - sas)),
        "answer_logit_drop_mean": float(answer_logit_drop),
    }


def compute_answer_logit_drop_for_set(
    model,
    examples: Sequence[ExampleFeatures],
    indices: np.ndarray,
) -> float:
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise ValueError("Model does not expose output embeddings")
    weight = lm_head.weight.detach().float().cpu().numpy()

    drops: List[float] = []
    for ex in examples:
        target_weight = weight[ex.target_token_id]
        for hidden in (ex.hidden_main, ex.hidden_socratic):
            selected_contrib = float(np.sum(hidden[indices] * target_weight[indices]))
            drops.append(selected_contrib)
    return float(np.mean(drops)) if drops else 0.0


def evaluate_generation_consistency(examples: Sequence[ExampleFeatures]) -> Dict[str, float]:
    valid = [ex for ex in examples if ex.pred_main is not None and ex.pred_socratic is not None]
    if not valid:
        return {}

    main_acc = np.mean([ex.pred_main == ex.answer for ex in valid])
    soc_acc = np.mean([ex.pred_socratic == ex.answer for ex in valid])
    rewrite_consistency = np.mean([ex.pred_main == ex.pred_socratic for ex in valid])
    rewrite_both_correct = np.mean(
        [(ex.pred_main == ex.answer) and (ex.pred_socratic == ex.answer) for ex in valid]
    )
    return {
        "num_eval_examples": len(valid),
        "main_accuracy": float(main_acc),
        "socratic_accuracy": float(soc_acc),
        "rewrite_consistency": float(rewrite_consistency),
        "rewrite_both_correct": float(rewrite_both_correct),
    }


def save_outputs(
    output_dir: Path,
    args: argparse.Namespace,
    coordinate_stats: Dict[str, np.ndarray],
    selected_sets: Dict[str, np.ndarray],
    set_summaries: List[Dict[str, float]],
    generation_summary: Dict[str, float],
    examples: Sequence[ExampleFeatures],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    coord_df = pd.DataFrame(
        {
            "coord": np.arange(len(coordinate_stats["orbit_stability"])),
            "orbit_stability": coordinate_stats["orbit_stability"],
            "output_necessity": coordinate_stats["output_necessity"],
            "same_answer_stability": coordinate_stats["same_answer_stability"],
            "high_activation": coordinate_stats["high_activation"],
        }
    ).sort_values(["orbit_stability", "output_necessity"], ascending=False)
    coord_df.to_csv(output_dir / "coordinate_scores.csv", index=False)

    pd.DataFrame(set_summaries).to_csv(output_dir / "set_summaries.csv", index=False)

    preview = []
    for ex in examples[: min(20, len(examples))]:
        preview.append(
            {
                "question": ex.question,
                "answer": ex.answer,
                "pred_main": ex.pred_main,
                "pred_socratic": ex.pred_socratic,
                "generation_main": ex.generation_main,
                "generation_socratic": ex.generation_socratic,
            }
        )

    summary = {
        "args": {
            k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()
        },
        "generation_summary": generation_summary,
        "set_summaries": set_summaries,
        "selected_sets": {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in selected_sets.items()
        },
        "preview_examples": preview,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Few-shot pilot for negative-controlled orbit analysis")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--num-samples", type=int, default=96)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--layer-index", type=int, default=-1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--same-answer-filter-quantile", type=float, default=0.75)
    parser.add_argument("--max-pairs-per-answer", type=int, default=8)
    parser.add_argument("--do-generation-eval", action="store_true")
    parser.add_argument("--generation-tokens", type=int, default=16)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = choose_device(args.device)
    print(f"[pilot] device={device}")
    print(f"[pilot] model={args.model_path}")

    pairs = load_gsm8k_orbit_pairs(
        dataset_root=args.dataset_root,
        split=args.split,
        max_pairs=args.num_samples,
        seed=args.seed,
    )
    print(f"[pilot] loaded {len(pairs)} paired orbit examples")

    tokenizer, model = load_local_model(args.model_path, device=device)
    print(
        f"[pilot] model_ready hidden={getattr(model.config, 'hidden_size', 'NA')} "
        f"layers={getattr(model.config, 'num_hidden_layers', 'NA')} "
        f"layer_index={args.layer_index}"
    )

    features = collect_features(
        model=model,
        tokenizer=tokenizer,
        pairs=pairs,
        device=device,
        max_length=args.max_length,
        layer_index=args.layer_index,
        do_generation_eval=args.do_generation_eval,
        generation_tokens=args.generation_tokens,
    )
    print(f"[pilot] extracted hidden features for {len(features)} examples")

    mu, sigma = zscore_basis(features)
    stats = compute_coordinate_statistics(
        model=model,
        examples=features,
        mu=mu,
        sigma=sigma,
        max_pairs_per_answer=args.max_pairs_per_answer,
        seed=args.seed,
    )

    selected_sets = select_coordinate_sets(
        stats=stats,
        top_k=min(args.top_k, len(stats["orbit_stability"])),
        seed=args.seed,
        same_answer_filter_quantile=args.same_answer_filter_quantile,
    )

    summaries = []
    for name in ["selected_neg", "selected_no_neg", "high_activation", "random"]:
        answer_logit_drop = compute_answer_logit_drop_for_set(
            model=model,
            examples=features,
            indices=selected_sets[name],
        )
        summary = summarize_coordinate_set(
            name,
            selected_sets[name],
            stats,
            answer_logit_drop=answer_logit_drop,
        )
        summaries.append(summary)
        print(f"[pilot] {name}: {json.dumps(summary, ensure_ascii=False)}")

    generation_summary = evaluate_generation_consistency(features)
    if generation_summary:
        print(f"[pilot] generation: {json.dumps(generation_summary, ensure_ascii=False)}")

    save_outputs(
        output_dir=args.output_dir,
        args=args,
        coordinate_stats=stats,
        selected_sets=selected_sets,
        set_summaries=summaries,
        generation_summary=generation_summary,
        examples=features,
    )
    print(f"[pilot] saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
