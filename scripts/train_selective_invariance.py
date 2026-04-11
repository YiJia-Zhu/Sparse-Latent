#!/usr/bin/env python3
"""
Minimal training/evaluation entry for selective invariance experiments.

This script upgrades the current analysis-only pilot into a small trainable
experiment package. It supports the core variants from EXPERIMENT_PLAN.md:

- baseline
- no-orbit
- aug-no-reg
- full-state
- selected-no-neg
- selected-neg
- random

The implementation is intentionally minimal:
- local checkpoints only
- LoRA on the last N transformer blocks
- GSM8K `main`/`socratic` paired rationales as orbit pairs
- answer-level evaluation against dataset ground truth
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

from pilot_negative_control_analysis import (
    OrbitPair,
    build_prompt,
    choose_device,
    collect_features,
    compute_answer_logit_drop_for_set,
    compute_coordinate_statistics,
    evaluate_generation_consistency,
    extract_last_number,
    load_gsm8k_orbit_pairs,
    load_local_model,
    select_coordinate_sets,
    summarize_coordinate_set,
    zscore_basis,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = ROOT / "huggingface_datasets"
DEFAULT_MODEL_PATH = ROOT / "huggingface_models" / "Llama-3.2-1B-Instruct"


VARIANTS = {
    "baseline",
    "no-orbit",
    "aug-no-reg",
    "full-state",
    "full-state-gated",
    "selected-no-neg",
    "selected-no-neg-gated",
    "selected-neg",
    "selected-neg-gated",
    "random",
}


@dataclass
class EvalPrediction:
    question: str
    answer: str
    prompt_main: str
    prompt_socratic: str
    prompt_tokens_main: int
    prompt_tokens_socratic: int
    pred_main: str
    pred_socratic: str
    generation_main: str
    generation_socratic: str
    generated_tokens_main: int
    generated_tokens_socratic: int


class OrbitTrainingDataset(Dataset):
    def __init__(self, pairs: Sequence[OrbitPair]):
        self.pairs = list(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        pair = self.pairs[idx]
        answer_text = pair.answer
        return {
            "question": pair.question,
            "answer": answer_text,
            "prompt_main": build_prompt(pair.question, pair.rationale_main),
            "prompt_socratic": build_prompt(pair.question, pair.rationale_socratic),
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_num_layers(model) -> int:
    config = model.config
    if hasattr(config, "num_hidden_layers"):
        return int(config.num_hidden_layers)
    if hasattr(config, "n_layer"):
        return int(config.n_layer)
    raise ValueError("Unable to infer transformer depth")


def get_model_type(model) -> str:
    return str(getattr(model.config, "model_type", "")).lower()


def build_lora_model(
    model,
    rank: int,
    alpha: int,
    dropout: float,
    last_n_layers: int,
):
    model_type = get_model_type(model)
    num_layers = get_num_layers(model)
    layer_ids = list(range(max(0, num_layers - last_n_layers), num_layers))

    if "llama" in model_type or "qwen" in model_type or "mistral" in model_type:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        layers_pattern = "layers"
    elif "gpt2" in model_type:
        target_modules = ["c_attn", "c_proj", "c_fc"]
        layers_pattern = "h"
    else:
        raise ValueError(f"Unsupported model_type for LoRA auto-config: {model_type}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=target_modules,
        layers_to_transform=layer_ids,
        layers_pattern=layers_pattern,
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model


def tokenize_texts(
    tokenizer,
    prompts: Sequence[str],
    answers: Sequence[str],
    max_length: int,
    device: str,
) -> Tuple[Dict[str, torch.Tensor], List[int]]:
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_id_rows: List[List[int]] = []
    label_rows: List[List[int]] = []
    prompt_lengths: List[int] = []

    for prompt, answer in zip(prompts, answers):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        answer_ids = tokenizer.encode(f" {answer}", add_special_tokens=False)
        if not answer_ids:
            answer_ids = tokenizer.encode(answer, add_special_tokens=False)
        if not answer_ids:
            raise ValueError(f"Failed to tokenize answer text: {answer!r}")

        if len(answer_ids) >= max_length:
            answer_ids = answer_ids[: max(1, max_length - 1)]

        max_prompt_len = max_length - len(answer_ids)
        if max_prompt_len < 1:
            max_prompt_len = 1
        prompt_ids = prompt_ids[:max_prompt_len]

        input_ids = prompt_ids + answer_ids
        labels = ([-100] * len(prompt_ids)) + answer_ids
        prompt_lengths.append(len(prompt_ids))
        input_id_rows.append(input_ids)
        label_rows.append(labels)

    max_batch_len = max(len(row) for row in input_id_rows)
    padded_inputs = []
    padded_masks = []
    padded_labels = []

    for input_ids, labels in zip(input_id_rows, label_rows):
        pad_len = max_batch_len - len(input_ids)
        padded_inputs.append(input_ids + ([pad_token_id] * pad_len))
        padded_masks.append(([1] * len(input_ids)) + ([0] * pad_len))
        padded_labels.append(labels + ([-100] * pad_len))

    batch = {
        "input_ids": torch.tensor(padded_inputs, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(padded_masks, dtype=torch.long, device=device),
        "labels": torch.tensor(padded_labels, dtype=torch.long, device=device),
    }
    return batch, prompt_lengths


def collate_orbit_batch(tokenizer, max_length: int, device: str):
    def _inner(batch: List[Dict[str, str]]) -> Dict[str, object]:
        prompts_main = [item["prompt_main"] for item in batch]
        prompts_soc = [item["prompt_socratic"] for item in batch]
        answers = [item["answer"] for item in batch]

        main_batch, main_prompt_lengths = tokenize_texts(
            tokenizer=tokenizer,
            prompts=prompts_main,
            answers=answers,
            max_length=max_length,
            device=device,
        )
        soc_batch, soc_prompt_lengths = tokenize_texts(
            tokenizer=tokenizer,
            prompts=prompts_soc,
            answers=answers,
            max_length=max_length,
            device=device,
        )

        return {
            "main": main_batch,
            "socratic": soc_batch,
            "main_prompt_lengths": main_prompt_lengths,
            "socratic_prompt_lengths": soc_prompt_lengths,
            "questions": [item["question"] for item in batch],
            "answers": answers,
            "prompts_main": prompts_main,
            "prompts_socratic": prompts_soc,
        }

    return _inner


def gather_prompt_hidden(outputs, prompt_lengths: Sequence[int], layer_index: int) -> torch.Tensor:
    hidden = outputs.hidden_states[layer_index]
    rows = []
    for i, prompt_len in enumerate(prompt_lengths):
        idx = max(0, min(prompt_len - 1, hidden.shape[1] - 1))
        rows.append(hidden[i, idx, :])
    return torch.stack(rows, dim=0)


def build_mask_tensor(
    variant: str,
    hidden_dim: int,
    selected_sets: Dict[str, np.ndarray],
    device: str,
) -> Optional[torch.Tensor]:
    base_variant = get_base_variant(variant)
    if base_variant == "full-state":
        mask = torch.ones(hidden_dim, dtype=torch.bool, device=device)
        return mask
    if base_variant == "selected-no-neg":
        mask = torch.zeros(hidden_dim, dtype=torch.bool, device=device)
        mask[torch.tensor(selected_sets["selected_no_neg"], dtype=torch.long, device=device)] = True
        return mask
    if base_variant == "selected-neg":
        mask = torch.zeros(hidden_dim, dtype=torch.bool, device=device)
        mask[torch.tensor(selected_sets["selected_neg"], dtype=torch.long, device=device)] = True
        return mask
    if base_variant == "random":
        mask = torch.zeros(hidden_dim, dtype=torch.bool, device=device)
        mask[torch.tensor(selected_sets["random"], dtype=torch.long, device=device)] = True
        return mask
    return None


def get_base_variant(variant: str) -> str:
    if variant.endswith("-gated"):
        return variant[: -len("-gated")]
    return variant


def is_gated_variant(variant: str) -> bool:
    return variant.endswith("-gated")


def compute_invariance_losses_per_example(
    z_main: torch.Tensor,
    z_soc: torch.Tensor,
    variant: str,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    base_variant = get_base_variant(variant)
    if base_variant in {"baseline", "no-orbit", "aug-no-reg"}:
        return z_main.new_zeros((z_main.shape[0],))
    if base_variant == "full-state":
        return ((z_main - z_soc) ** 2).mean(dim=-1)
    if mask is None:
        return z_main.new_zeros((z_main.shape[0],))
    return ((z_main[:, mask] - z_soc[:, mask]) ** 2).mean(dim=-1)


def compute_gate_weights(
    nll_main: torch.Tensor,
    nll_soc: torch.Tensor,
    gate_center: float,
    gate_temperature: float,
) -> torch.Tensor:
    mean_nll = 0.5 * (nll_main.detach() + nll_soc.detach())
    centered = mean_nll - gate_center
    weights = torch.sigmoid(centered / max(gate_temperature, 1e-6))
    return weights


def compute_answer_nll_per_sample(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    token_losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).view(shift_labels.shape)
    valid = (shift_labels != -100).float()
    denom = valid.sum(dim=-1).clamp_min(1.0)
    return (token_losses * valid).sum(dim=-1) / denom


def normalize_hidden(hidden: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return (hidden - mu) / sigma


def train_one_epoch(
    model,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    scaler,
    device: str,
    layer_index: int,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    variant: str,
    mask: Optional[torch.Tensor],
    lambda_orbit: float,
    gate_center: float,
    gate_temperature: float,
    grad_accum_steps: int,
    mixed_precision: bool,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_answer_loss = 0.0
    total_orbit_loss = 0.0
    total_gate_weight = 0.0
    step_count = 0
    optimizer.zero_grad(set_to_none=True)

    autocast_dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16

    for step_idx, batch in enumerate(dataloader):
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=(device == "cuda" and mixed_precision)):
            main_outputs = model(
                **batch["main"],
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            answer_loss = main_outputs.loss
            nll_main = compute_answer_nll_per_sample(main_outputs.logits, batch["main"]["labels"])
            nll_soc = nll_main

            orbit_loss = answer_loss.new_zeros(())
            gate_weight_mean = answer_loss.new_tensor(1.0)
            if get_base_variant(variant) != "no-orbit":
                soc_outputs = model(
                    **batch["socratic"],
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                answer_loss = 0.5 * (answer_loss + soc_outputs.loss)
                nll_soc = compute_answer_nll_per_sample(soc_outputs.logits, batch["socratic"]["labels"])

                z_main = normalize_hidden(
                    gather_prompt_hidden(main_outputs, batch["main_prompt_lengths"], layer_index),
                    mu=mu,
                    sigma=sigma,
                )
                z_soc = normalize_hidden(
                    gather_prompt_hidden(soc_outputs, batch["socratic_prompt_lengths"], layer_index),
                    mu=mu,
                    sigma=sigma,
                )
                orbit_losses = compute_invariance_losses_per_example(z_main, z_soc, variant=variant, mask=mask)
                if is_gated_variant(variant):
                    gate_weights = compute_gate_weights(
                        nll_main=nll_main,
                        nll_soc=nll_soc,
                        gate_center=gate_center,
                        gate_temperature=gate_temperature,
                    )
                    gate_weight_mean = gate_weights.mean()
                    orbit_loss = (orbit_losses * gate_weights).mean()
                else:
                    orbit_loss = orbit_losses.mean()

            loss = answer_loss + lambda_orbit * orbit_loss
            loss = loss / grad_accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step_idx + 1) % grad_accum_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.detach().item()) * grad_accum_steps
        total_answer_loss += float(answer_loss.detach().item())
        total_orbit_loss += float(orbit_loss.detach().item())
        total_gate_weight += float(gate_weight_mean.detach().item())
        step_count += 1

    if step_count > 0 and step_count % grad_accum_steps != 0:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    return {
        "train_loss": total_loss / max(1, step_count),
        "train_answer_loss": total_answer_loss / max(1, step_count),
        "train_orbit_loss": total_orbit_loss / max(1, step_count),
        "train_gate_weight_mean": total_gate_weight / max(1, step_count),
    }


@torch.no_grad()
def generate_answer(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_length: int,
    max_new_tokens: int,
) -> Tuple[str, str, int, int]:
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    prompt_token_count = int(enc["attention_mask"][0].sum().item())
    gen_ids = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated_token_count = max(0, int(gen_ids.shape[1] - enc["input_ids"].shape[1]))
    text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    suffix = text[len(prompt):].strip()
    return suffix, extract_last_number(suffix), prompt_token_count, generated_token_count


@torch.no_grad()
def evaluate_pairs(
    model,
    tokenizer,
    pairs: Sequence[OrbitPair],
    device: str,
    max_length: int,
    generation_tokens: int,
) -> Tuple[Dict[str, float], List[EvalPrediction]]:
    model.eval()
    predictions: List[EvalPrediction] = []

    for pair in pairs:
        prompt_main = build_prompt(pair.question, pair.rationale_main)
        prompt_soc = build_prompt(pair.question, pair.rationale_socratic)
        gen_main, pred_main, prompt_tokens_main, generated_tokens_main = generate_answer(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_main,
            device=device,
            max_length=max_length,
            max_new_tokens=generation_tokens,
        )
        gen_soc, pred_soc, prompt_tokens_socratic, generated_tokens_socratic = generate_answer(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_soc,
            device=device,
            max_length=max_length,
            max_new_tokens=generation_tokens,
        )
        predictions.append(
            EvalPrediction(
                question=pair.question,
                answer=pair.answer,
                prompt_main=prompt_main,
                prompt_socratic=prompt_soc,
                prompt_tokens_main=prompt_tokens_main,
                prompt_tokens_socratic=prompt_tokens_socratic,
                pred_main=pred_main,
                pred_socratic=pred_soc,
                generation_main=gen_main,
                generation_socratic=gen_soc,
                generated_tokens_main=generated_tokens_main,
                generated_tokens_socratic=generated_tokens_socratic,
            )
        )

    main_acc = np.mean([pred.pred_main == pred.answer for pred in predictions]) if predictions else 0.0
    soc_acc = np.mean([pred.pred_socratic == pred.answer for pred in predictions]) if predictions else 0.0
    rewrite_consistency = np.mean([pred.pred_main == pred.pred_socratic for pred in predictions]) if predictions else 0.0
    both_correct = np.mean(
        [(pred.pred_main == pred.answer) and (pred.pred_socratic == pred.answer) for pred in predictions]
    ) if predictions else 0.0
    pair_accuracy = np.mean(
        [(pred.pred_main == pred.answer) or (pred.pred_socratic == pred.answer) for pred in predictions]
    ) if predictions else 0.0
    mean_generated_tokens_main = np.mean([pred.generated_tokens_main for pred in predictions]) if predictions else 0.0
    mean_generated_tokens_socratic = (
        np.mean([pred.generated_tokens_socratic for pred in predictions]) if predictions else 0.0
    )
    mean_prompt_tokens_main = np.mean([pred.prompt_tokens_main for pred in predictions]) if predictions else 0.0
    mean_prompt_tokens_socratic = (
        np.mean([pred.prompt_tokens_socratic for pred in predictions]) if predictions else 0.0
    )

    metrics = {
        "num_eval_examples": len(predictions),
        "main_accuracy": float(main_acc),
        "socratic_accuracy": float(soc_acc),
        "mean_accuracy": float(0.5 * (main_acc + soc_acc)),
        "rewrite_consistency": float(rewrite_consistency),
        "accuracy_conditioned_rewrite_consistency": float(both_correct),
        "pair_accuracy_at_least_one_correct": float(pair_accuracy),
        "mean_prompt_tokens_main": float(mean_prompt_tokens_main),
        "mean_prompt_tokens_socratic": float(mean_prompt_tokens_socratic),
        "mean_prompt_tokens": float(0.5 * (mean_prompt_tokens_main + mean_prompt_tokens_socratic)),
        "mean_generated_tokens_main": float(mean_generated_tokens_main),
        "mean_generated_tokens_socratic": float(mean_generated_tokens_socratic),
        "mean_generated_tokens": float(0.5 * (mean_generated_tokens_main + mean_generated_tokens_socratic)),
    }
    return metrics, predictions


@torch.no_grad()
def compute_diagnostics(
    model,
    tokenizer,
    pairs: Sequence[OrbitPair],
    device: str,
    max_length: int,
    layer_index: int,
    mu: np.ndarray,
    sigma: np.ndarray,
    selected_sets: Dict[str, np.ndarray],
    max_pairs_per_answer: int,
    seed: int,
    variant: str,
) -> Dict[str, object]:
    model.eval()
    features = collect_features(
        model=model,
        tokenizer=tokenizer,
        pairs=pairs,
        device=device,
        max_length=max_length,
        layer_index=layer_index,
        do_generation_eval=False,
        generation_tokens=8,
    )
    stats = compute_coordinate_statistics(
        model=model,
        examples=features,
        mu=mu,
        sigma=sigma,
        max_pairs_per_answer=max_pairs_per_answer,
        seed=seed,
    )

    summaries: List[Dict[str, float]] = []
    for name in ["selected_neg", "selected_no_neg", "high_activation", "random"]:
        indices = selected_sets[name]
        summary = summarize_coordinate_set(
            name=name,
            indices=indices,
            stats=stats,
            answer_logit_drop=compute_answer_logit_drop_for_set(model, features, indices),
        )
        summaries.append(summary)

    ovg = None
    base_variant = get_base_variant(variant)
    if base_variant == "selected-neg":
        ovg = float(summaries[0]["orbit_minus_same_answer"])
    elif base_variant == "selected-no-neg":
        ovg = float(summaries[1]["orbit_minus_same_answer"])
    elif base_variant == "random":
        ovg = float(summaries[3]["orbit_minus_same_answer"])
    elif base_variant == "full-state":
        ovg = float(np.mean(stats["orbit_stability"] - stats["same_answer_stability"]))

    generation_summary = evaluate_generation_consistency(features)
    return {
        "diagnostic_set_summaries": summaries,
        "orbit_variance_gap": ovg,
        "feature_generation_summary": generation_summary,
    }


def save_predictions(predictions: Sequence[EvalPrediction], path: Path) -> None:
    rows = [asdict(pred) for pred in predictions]
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))


def run_experiment(args: argparse.Namespace) -> Dict[str, object]:
    if args.variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {args.variant}")

    seed_everything(args.seed)
    device = choose_device(args.device)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    scoring_pairs = load_gsm8k_orbit_pairs(
        dataset_root=args.dataset_root,
        split="train",
        max_pairs=args.max_scoring_samples,
        seed=args.seed,
    )
    train_pairs = load_gsm8k_orbit_pairs(
        dataset_root=args.dataset_root,
        split="train",
        max_pairs=args.max_train_samples,
        seed=args.seed + 1,
    )
    eval_pairs = load_gsm8k_orbit_pairs(
        dataset_root=args.dataset_root,
        split=args.eval_split,
        max_pairs=args.max_eval_samples,
        seed=args.seed,
    )

    tokenizer, model = load_local_model(args.model_path, device=device)
    tokenizer.padding_side = "right"
    model.config.use_cache = False

    selector_features = collect_features(
        model=model,
        tokenizer=tokenizer,
        pairs=scoring_pairs,
        device=device,
        max_length=args.max_length,
        layer_index=args.layer_index,
        do_generation_eval=False,
        generation_tokens=8,
    )
    mu_np, sigma_np = zscore_basis(selector_features)
    stats = compute_coordinate_statistics(
        model=model,
        examples=selector_features,
        mu=mu_np,
        sigma=sigma_np,
        max_pairs_per_answer=args.max_pairs_per_answer,
        seed=args.seed,
    )
    selected_sets = select_coordinate_sets(
        stats=stats,
        top_k=min(args.top_k, len(stats["orbit_stability"])),
        seed=args.seed,
        same_answer_filter_quantile=args.same_answer_filter_quantile,
    )

    selector_rows = []
    for name in ["selected_neg", "selected_no_neg", "high_activation", "random"]:
        selector_rows.append(
            summarize_coordinate_set(
                name=name,
                indices=selected_sets[name],
                stats=stats,
                answer_logit_drop=compute_answer_logit_drop_for_set(model, selector_features, selected_sets[name]),
            )
        )
    pd.DataFrame(selector_rows).to_csv(output_dir / "selector_summary.csv", index=False)

    if args.variant != "baseline" and args.train_epochs > 0:
        model = build_lora_model(
            model=model,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            last_n_layers=args.last_n_layers,
        )
        model.config.use_cache = False
        model.train()

        dataset = OrbitTrainingDataset(train_pairs)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_orbit_batch(
                tokenizer=tokenizer,
                max_length=args.max_length,
                device=device,
            ),
        )

        trainable_params = [param for param in model.parameters() if param.requires_grad]
        optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
        total_steps = math.ceil(len(dataloader) / max(1, args.grad_accum_steps)) * args.train_epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=int(total_steps * args.warmup_ratio),
            num_training_steps=max(1, total_steps),
        )
        scaler = torch.amp.GradScaler("cuda") if device == "cuda" and args.mixed_precision else None
        mu_t = torch.tensor(mu_np, device=device, dtype=torch.float32)
        sigma_t = torch.tensor(sigma_np, device=device, dtype=torch.float32)
        mask = build_mask_tensor(
            variant=args.variant,
            hidden_dim=len(mu_np),
            selected_sets=selected_sets,
            device=device,
        )

        train_history: List[Dict[str, float]] = []
        for epoch in range(args.train_epochs):
            epoch_metrics = train_one_epoch(
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                layer_index=args.layer_index,
                mu=mu_t,
                sigma=sigma_t,
                variant=args.variant,
                mask=mask,
                lambda_orbit=args.lambda_orbit,
                gate_center=args.gate_center,
                gate_temperature=args.gate_temperature,
                grad_accum_steps=args.grad_accum_steps,
                mixed_precision=args.mixed_precision,
            )
            epoch_metrics["epoch"] = epoch + 1
            train_history.append(epoch_metrics)
            print(f"[train] epoch={epoch + 1} {json.dumps(epoch_metrics)}")

        pd.DataFrame(train_history).to_csv(output_dir / "train_history.csv", index=False)

        if hasattr(model, "save_pretrained"):
            model.save_pretrained(output_dir / "adapter")
            tokenizer.save_pretrained(output_dir / "adapter")
    else:
        train_history = []

    eval_metrics, predictions = evaluate_pairs(
        model=model,
        tokenizer=tokenizer,
        pairs=eval_pairs,
        device=device,
        max_length=args.max_length,
        generation_tokens=args.generation_tokens,
    )
    diagnostics = compute_diagnostics(
        model=model,
        tokenizer=tokenizer,
        pairs=eval_pairs,
        device=device,
        max_length=args.max_length,
        layer_index=args.layer_index,
        mu=mu_np,
        sigma=sigma_np,
        selected_sets=selected_sets,
        max_pairs_per_answer=args.max_pairs_per_answer,
        seed=args.seed,
        variant=args.variant,
    )

    save_predictions(predictions, output_dir / "predictions.json")

    summary = {
        "variant": args.variant,
        "model_path": str(args.model_path),
        "device": device,
        "num_layers": get_num_layers(model),
        "train_pairs": len(train_pairs),
        "scoring_pairs": len(scoring_pairs),
        "eval_pairs": len(eval_pairs),
        "selector_summary": selector_rows,
        "selected_sets": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in selected_sets.items()},
        "eval_metrics": eval_metrics,
        "diagnostics": diagnostics,
        "gate": {
            "enabled": is_gated_variant(args.variant),
            "center": args.gate_center,
            "temperature": args.gate_temperature,
        },
        "train_history": train_history,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/evaluate selective invariance variants")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
    )
    parser.add_argument("--variant", type=str, default="selected-neg", choices=sorted(VARIANTS))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--max-scoring-samples", type=int, default=128)
    parser.add_argument("--max-train-samples", type=int, default=512)
    parser.add_argument("--max-eval-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--train-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--lambda-orbit", type=float, default=0.1)
    parser.add_argument("--gate-center", type=float, default=1.5)
    parser.add_argument("--gate-temperature", type=float, default=0.5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--last-n-layers", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--layer-index", type=int, default=-1)
    parser.add_argument("--same-answer-filter-quantile", type=float, default=0.75)
    parser.add_argument("--max-pairs-per-answer", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--generation-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--mixed-precision", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    start = time.time()
    summary = run_experiment(args)
    elapsed = time.time() - start
    print(
        json.dumps(
            {
                "variant": summary["variant"],
                "elapsed_seconds": round(elapsed, 2),
                "eval_metrics": summary["eval_metrics"],
                "orbit_variance_gap": summary["diagnostics"]["orbit_variance_gap"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
