#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from peft import LoraConfig, TaskType

from src.model import CODI, DataArguments, ModelArguments, TrainingArguments


def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(",", "")
    pred = [s for s in re.findall(r"-?\d+\.?\d*", sentence)]
    if not pred:
        return float("inf")
    return float(pred[-1])


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_lora_config(model_name_or_path: str, lora_r: int, lora_alpha: int) -> LoraConfig:
    lower = model_name_or_path.lower()
    if any(name in lower for name in ["llama", "mistral", "falcon", "qwen"]):
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    elif "phi" in lower:
        target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
    elif "gpt2" in lower:
        target_modules = ["c_attn", "c_proj", "c_fc"]
    else:
        raise ValueError(f"Unsupported model for LoRA config: {model_name_or_path}")
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules=target_modules,
        init_lora_weights=True,
    )


def load_examples(local_data_path: Path, max_samples: int | None) -> List[Dict[str, object]]:
    df = pd.read_parquet(local_data_path)
    if max_samples is not None and max_samples > 0:
        df = df.iloc[:max_samples]
    rows: List[Dict[str, object]] = []
    for record in df.to_dict(orient="records"):
        answer_text = str(record["answer"])
        final_answer = answer_text.split("####")[-1].strip()
        try:
            gold = float(final_answer.replace(",", ""))
        except ValueError:
            gold = float("inf")
        rows.append(
            {
                "question": str(record["question"]).strip().replace("  ", " "),
                "gold_answer": gold,
            }
        )
    return rows


def load_model_and_tokenizer(args: argparse.Namespace, device: str):
    model_args = ModelArguments(
        model_name_or_path=str(args.model_path),
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_init=True,
        train=False,
        ckpt_dir=str(args.ckpt_dir),
    )
    data_args = DataArguments(
        data_name="gsm8k-local",
        local_data_path=str(args.local_test_path),
        batch_size=args.batch_size,
    )
    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "tmp_unused"),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        bf16=(device == "cuda"),
        model_max_length=args.model_max_length,
        use_lora=True,
        num_latent=args.num_latent,
        use_prj=args.use_prj,
        prj_dim=args.prj_dim,
        prj_no_ln=False,
        prj_dropout=0.0,
        remove_eos=args.remove_eos,
        greedy=args.greedy,
        inf_latent_iterations=args.inf_latent_iterations,
        inf_num_iterations=args.inf_num_iterations,
        report_to=[],
        logging_steps=1,
    )

    lora_config = build_lora_config(str(args.model_path), args.lora_r, args.lora_alpha)
    model = CODI(model_args, training_args, lora_config)

    ckpt_path = args.ckpt_dir / "pytorch_model.bin"
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model.codi.tie_weights()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(args.model_path),
        model_max_length=args.model_max_length,
        padding_side="left",
        use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.pad_token_id = model.pad_token_id
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")

    model = model.to(device)
    if device == "cuda":
        model = model.to(torch.bfloat16)
    model.eval()
    return model, tokenizer, model_args, data_args, training_args


def build_question_batches(
    questions: Sequence[str],
    tokenizer,
    model,
    batch_size: int,
    remove_eos: bool,
    device: str,
) -> List[Dict[str, torch.Tensor]]:
    batches: List[Dict[str, torch.Tensor]] = []
    for start in range(0, len(questions), batch_size):
        question_batch = questions[start : start + batch_size]
        enc = tokenizer(question_batch, return_tensors="pt", padding="longest")
        if remove_eos:
            bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).expand(enc["input_ids"].size(0), 1)
        else:
            bot_tensor = torch.tensor([tokenizer.eos_token_id, model.bot_id], dtype=torch.long).expand(
                enc["input_ids"].size(0), 2
            )
        enc["input_ids"] = torch.cat((enc["input_ids"], bot_tensor), dim=1)
        enc["attention_mask"] = torch.cat((enc["attention_mask"], torch.ones_like(bot_tensor)), dim=1)
        batches.append({k: v.to(device) for k, v in enc.items()})
    return batches


def sample_next_token(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    logits = logits / temperature
    if top_k > 1:
        top_k_values, _ = torch.topk(logits, top_k, dim=-1)
        min_top_k_value = top_k_values[:, -1].unsqueeze(-1)
        logits = logits.clone()
        logits[logits < min_top_k_value] = -float("inf")

    if top_p < 1.0:
        sorted_logit, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logit, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        if sorted_indices_to_remove.any():
            sorted_indices_to_remove = sorted_indices_to_remove.roll(1, dims=-1)
            sorted_indices_to_remove[:, 0] = False
        logits = logits.clone()
        for b in range(logits.size(0)):
            logits[b, sorted_indices[b, sorted_indices_to_remove[b]]] = -float("inf")

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def run_single_pass(
    model,
    tokenizer,
    training_args,
    question_batches: Sequence[Dict[str, torch.Tensor]],
    gold_answers: Sequence[float],
    questions: Sequence[str],
    args: argparse.Namespace,
    pass_index: int,
) -> Dict[str, object]:
    predictions: List[Dict[str, object]] = []
    gen_lengths: List[int] = []

    offset = 0
    for batch in question_batches:
        batch_questions = questions[offset : offset + batch["input_ids"].size(0)]
        batch_gold = gold_answers[offset : offset + batch["input_ids"].size(0)]
        offset += batch["input_ids"].size(0)

        with torch.no_grad():
            outputs = model.codi(
                input_ids=batch["input_ids"],
                use_cache=True,
                output_hidden_states=True,
                attention_mask=batch["attention_mask"],
            )
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

            for _ in range(training_args.inf_latent_iterations):
                outputs = model.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values,
                )
                past_key_values = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                if training_args.use_prj:
                    latent_embd = model.prj(latent_embd)

            if training_args.remove_eos:
                eot_tokens = torch.tensor([model.eot_id], dtype=torch.long, device=batch["input_ids"].device)
            else:
                eot_tokens = torch.tensor(
                    [model.eot_id, tokenizer.eos_token_id], dtype=torch.long, device=batch["input_ids"].device
                )
            eot_emb = model.get_embd(model.codi, model.model_name)(eot_tokens).unsqueeze(0)
            eot_emb = eot_emb.expand(batch["input_ids"].size(0), -1, -1)

            output = eot_emb
            finished = torch.zeros(batch["input_ids"].size(0), dtype=torch.bool, device=batch["input_ids"].device)
            pred_tokens = [[] for _ in range(batch["input_ids"].size(0))]
            for _ in range(args.max_new_tokens):
                out = model.codi(
                    inputs_embeds=output,
                    output_hidden_states=False,
                    attention_mask=None,
                    use_cache=True,
                    output_attentions=False,
                    past_key_values=past_key_values,
                )
                past_key_values = out.past_key_values
                logits = out.logits[:, -1, : model.codi.config.vocab_size - 1]
                if training_args.greedy:
                    next_token_ids = torch.argmax(logits, dim=-1).squeeze(-1)
                else:
                    next_token_ids = sample_next_token(
                        logits=logits,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p,
                    )

                for b in range(batch["input_ids"].size(0)):
                    if not finished[b]:
                        pred_tokens[b].append(next_token_ids[b].item())
                        if next_token_ids[b] == tokenizer.eos_token_id:
                            finished[b] = True
                if finished.all():
                    break
                output = model.get_embd(model.codi, model.model_name)(next_token_ids).unsqueeze(1).to(
                    batch["input_ids"].device
                )

        for q, gold, toks in zip(batch_questions, batch_gold, pred_tokens):
            decoded = tokenizer.decode(toks, skip_special_tokens=True)
            pred = extract_answer_number(decoded)
            gen_lengths.append(len(toks))
            predictions.append(
                {
                    "question": q,
                    "gold_answer": gold,
                    "prediction": pred,
                    "correct": pred == gold,
                    "generated_text": decoded,
                    "generated_tokens": len(toks),
                    "pass_index": pass_index,
                }
            )

    accuracy = sum(1 for row in predictions if row["correct"]) / max(1, len(predictions))
    mean_generated_tokens = sum(gen_lengths) / max(1, len(gen_lengths))
    return {
        "pass_index": pass_index,
        "accuracy": accuracy,
        "mean_generated_tokens": mean_generated_tokens,
        "predictions": predictions,
    }


def run_eval(args: argparse.Namespace) -> Dict[str, object]:
    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    model, tokenizer, model_args, data_args, training_args = load_model_and_tokenizer(args, device)

    examples = load_examples(args.local_test_path, None if args.max_samples <= 0 else args.max_samples)
    questions = [ex["question"] for ex in examples]
    gold_answers = [ex["gold_answer"] for ex in examples]
    question_batches = build_question_batches(
        questions=questions,
        tokenizer=tokenizer,
        model=model,
        batch_size=args.batch_size,
        remove_eos=training_args.remove_eos,
        device=device,
    )

    per_pass: List[Dict[str, object]] = []
    for pass_index in range(args.inf_num_iterations):
        seed = args.seed + pass_index
        set_all_seeds(seed)
        result = run_single_pass(
            model=model,
            tokenizer=tokenizer,
            training_args=training_args,
            question_batches=question_batches,
            gold_answers=gold_answers,
            questions=questions,
            args=args,
            pass_index=pass_index,
        )
        result["seed"] = seed
        per_pass.append(result)

    avg_accuracy = sum(item["accuracy"] for item in per_pass) / max(1, len(per_pass))
    avg_mean_generated_tokens = sum(item["mean_generated_tokens"] for item in per_pass) / max(1, len(per_pass))
    summary = {
        "eval_mode": "official_teststyle_local",
        "ckpt_dir": str(args.ckpt_dir),
        "local_test_path": str(args.local_test_path),
        "num_eval_examples": len(examples),
        "inf_num_iterations": args.inf_num_iterations,
        "average_accuracy": avg_accuracy,
        "average_mean_generated_tokens": avg_mean_generated_tokens,
        "per_pass_accuracy": [item["accuracy"] for item in per_pass],
        "per_pass_mean_generated_tokens": [item["mean_generated_tokens"] for item in per_pass],
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "greedy": args.greedy,
        "remove_eos": args.remove_eos,
        "num_latent": args.num_latent,
        "inf_latent_iterations": args.inf_latent_iterations,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (args.output_dir / "per_pass_results.json").write_text(
        json.dumps(
            [
                {
                    "pass_index": item["pass_index"],
                    "seed": item["seed"],
                    "accuracy": item["accuracy"],
                    "mean_generated_tokens": item["mean_generated_tokens"],
                }
                for item in per_pass
            ],
            indent=2,
            ensure_ascii=False,
        )
    )
    (args.output_dir / "predictions_pass0.json").write_text(
        json.dumps(per_pass[0]["predictions"], indent=2, ensure_ascii=False)
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Local official test.py-style evaluator for CODI checkpoints")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--local-test-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=0, help="<=0 means use full local test set")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--model-max-length", type=int, default=512)
    parser.add_argument("--num-latent", type=int, default=6)
    parser.add_argument("--inf-latent-iterations", type=int, default=6)
    parser.add_argument("--inf-num-iterations", type=int, default=5)
    parser.add_argument("--lora-r", type=int, default=128)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--prj-dim", type=int, default=2048)
    parser.add_argument("--use-prj", action="store_true")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.set_defaults(remove_eos=False)
    parser.add_argument("--remove-eos", dest="remove_eos", action="store_true")
    parser.add_argument("--keep-eos", dest="remove_eos", action="store_false")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
