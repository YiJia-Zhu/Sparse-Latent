#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_answer_number(text: str) -> float:
    text = text.replace(",", "")
    matches = re.findall(r"-?\d+\.?\d*", text)
    if not matches:
        return float("inf")
    return float(matches[-1])


def build_prompt(question: str, tokenizer) -> str:
    user_text = (
        "Solve the following GSM8K math word problem. "
        "You may reason briefly, but end with a final line in the format "
        "'The answer is: <number>'.\n\n"
        f"Question: {question}"
    )
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return user_text


def load_examples(local_test_path: Path, max_samples: int) -> List[Dict[str, object]]:
    df = pd.read_parquet(local_test_path)
    if max_samples > 0:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate plain Llama model on local GSM8K parquet")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--local-test-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=0, help="<=0 means full local test set")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--model-max-length", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    examples = load_examples(args.local_test_path, args.max_samples)
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path),
        model_max_length=args.model_max_length,
        padding_side="left",
        use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        torch_dtype=torch.bfloat16 if device == "cuda" else None,
    )
    model = model.to(device)
    model.eval()

    prompts = [build_prompt(ex["question"], tokenizer) for ex in examples]
    predictions: List[Dict[str, object]] = []
    token_lengths: List[int] = []

    for start in range(0, len(prompts), args.batch_size):
        batch_examples = examples[start : start + args.batch_size]
        batch_prompts = prompts[start : start + args.batch_size]
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.model_max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        prompt_lengths = enc["attention_mask"].sum(dim=1).tolist()

        with torch.no_grad():
            outputs = model.generate(
                **enc,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for idx, (example, prompt_len) in enumerate(zip(batch_examples, prompt_lengths)):
            full_ids = outputs[idx]
            new_ids = full_ids[int(prompt_len) :]
            decoded = tokenizer.decode(new_ids, skip_special_tokens=True)
            pred = extract_answer_number(decoded)
            gen_len = int(new_ids.shape[0])
            token_lengths.append(gen_len)
            predictions.append(
                {
                    "question": example["question"],
                    "gold_answer": example["gold_answer"],
                    "prediction": pred,
                    "correct": pred == example["gold_answer"],
                    "generated_text": decoded,
                    "generated_tokens": gen_len,
                }
            )

    accuracy = sum(1 for row in predictions if row["correct"]) / max(1, len(predictions))
    mean_generated_tokens = sum(token_lengths) / max(1, len(token_lengths))
    summary = {
        "eval_mode": "plain_llama_local_gsm8k",
        "model_path": str(args.model_path),
        "local_test_path": str(args.local_test_path),
        "num_eval_examples": len(predictions),
        "accuracy": accuracy,
        "mean_generated_tokens": mean_generated_tokens,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "model_max_length": args.model_max_length,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (args.output_dir / "predictions.json").write_text(json.dumps(predictions, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
