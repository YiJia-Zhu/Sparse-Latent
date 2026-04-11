#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from peft import LoraConfig, TaskType
import os

CODI_MODEL_IMPL = os.environ.get("CODI_MODEL_IMPL", "adaptive").strip().lower()
if CODI_MODEL_IMPL == "official":
    from src.model import CODI, DataArguments, ModelArguments, TrainingArguments
elif CODI_MODEL_IMPL == "adaptive":
    from src.model_adaptive import CODI, DataArguments, ModelArguments, TrainingArguments
else:
    raise ValueError(f"Unsupported CODI_MODEL_IMPL={CODI_MODEL_IMPL!r}. Expected 'official' or 'adaptive'.")


def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(",", "")
    pred = [s for s in re.findall(r"-?\d+\.?\d*", sentence)]
    if not pred:
        return float("inf")
    return float(pred[-1])


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


def load_examples(local_data_path: Path, max_samples: int) -> List[Dict[str, object]]:
    df = pd.read_parquet(local_data_path)
    rows: List[Dict[str, object]] = []
    for record in df.to_dict(orient="records")[:max_samples]:
        answer_text = str(record["answer"])
        final_answer = answer_text.split("####")[-1].strip()
        rows.append(
            {
                "question": str(record["question"]).strip(),
                "gold_answer": float(final_answer.replace(",", "")),
            }
        )
    return rows


def run_eval(args: argparse.Namespace) -> Dict[str, object]:
    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    print(f"[evaluate_local_codi.py] CODI_MODEL_IMPL={CODI_MODEL_IMPL}")

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
        remove_eos=True,
        greedy=args.greedy,
        inf_latent_iterations=args.inf_latent_iterations,
        inf_num_iterations=1,
        report_to=[],
        logging_steps=1,
    )

    lora_config = build_lora_config(str(args.model_path), args.lora_r, args.lora_alpha)
    model = CODI(model_args, training_args, lora_config)

    ckpt_path = args.ckpt_dir / "pytorch_model.bin"
    state_dict = torch.load(ckpt_path, map_location="cpu")
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

    examples = load_examples(args.local_test_path, args.max_samples)
    questions = [ex["question"] for ex in examples]
    gold_answers = [ex["gold_answer"] for ex in examples]

    predictions: List[Dict[str, object]] = []
    gen_lengths: List[int] = []

    for start in range(0, len(questions), args.batch_size):
        batch_questions = questions[start : start + args.batch_size]
        batch_gold = gold_answers[start : start + args.batch_size]
        enc = tokenizer(batch_questions, return_tensors="pt", padding="longest")
        if training_args.remove_eos:
            bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).expand(enc["input_ids"].size(0), 1)
        else:
            bot_tensor = torch.tensor([tokenizer.eos_token_id, model.bot_id], dtype=torch.long).expand(
                enc["input_ids"].size(0), 2
            )
        enc["input_ids"] = torch.cat((enc["input_ids"], bot_tensor), dim=1)
        enc["attention_mask"] = torch.cat((enc["attention_mask"], torch.ones_like(bot_tensor)), dim=1)
        batch = {k: v.to(device) for k, v in enc.items()}

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

            eot_emb = model.get_embd(model.codi, model.model_name)(
                torch.tensor([model.eot_id], dtype=torch.long, device=device)
            ).unsqueeze(0)
            eot_emb = eot_emb.expand(batch["input_ids"].size(0), -1, -1)

            output = eot_emb
            finished = torch.zeros(len(batch_questions), dtype=torch.bool, device=device)
            pred_tokens = [[] for _ in range(len(batch_questions))]
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
                    probs = F.softmax(logits / 0.1, dim=-1)
                    next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)

                for b in range(len(batch_questions)):
                    if not finished[b]:
                        pred_tokens[b].append(next_token_ids[b].item())
                        if next_token_ids[b] == tokenizer.eos_token_id:
                            finished[b] = True
                if finished.all():
                    break
                output = model.get_embd(model.codi, model.model_name)(next_token_ids).unsqueeze(1).to(device)

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
                }
            )

    accuracy = sum(1 for row in predictions if row["correct"]) / max(1, len(predictions))
    mean_generated_tokens = sum(gen_lengths) / max(1, len(gen_lengths))

    summary = {
        "model_impl": CODI_MODEL_IMPL,
        "ckpt_dir": str(args.ckpt_dir),
        "local_test_path": str(args.local_test_path),
        "num_eval_examples": len(predictions),
        "accuracy": accuracy,
        "mean_generated_tokens": mean_generated_tokens,
        "num_latent": args.num_latent,
        "inf_latent_iterations": args.inf_latent_iterations,
        "greedy": args.greedy,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (args.output_dir / "predictions.json").write_text(json.dumps(predictions, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate local CODI checkpoint on local GSM8K parquet")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--local-test-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--model-max-length", type=int, default=512)
    parser.add_argument("--num-latent", type=int, default=6)
    parser.add_argument("--inf-latent-iterations", type=int, default=6)
    parser.add_argument("--lora-r", type=int, default=128)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--prj-dim", type=int, default=2048)
    parser.add_argument("--use-prj", action="store_true")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
