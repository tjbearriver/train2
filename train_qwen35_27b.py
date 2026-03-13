#!/usr/bin/env python3
"""Fine-tune Qwen3.5-27B with bf16 LoRA on 1000 articles (for RunPod A100).

Usage:
  python train_qwen35_27b.py
  python train_qwen35_27b.py --size 1000 --num-eval 50
"""

import argparse
import gc
import json
import random
import re
import time
from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent
FULL_TRAIN_PATH = BASE_DIR / "data" / "train.json"
EVAL_PATH = BASE_DIR / "data" / "eval.json"
BASE_MODEL = "unsloth/Qwen3.5-27B"
MAX_SEQ_LENGTH = 8192

# LoRA config (same rank/targets as other models for fair comparison)
LORA_R = 64
LORA_ALPHA = 64
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Training config
EPOCHS = 3
BATCH_SIZE = 1
GRAD_ACCUM = 16
LEARNING_RATE = 2e-4
WARMUP_STEPS = 20
LR_SCHEDULER = "cosine"
OPTIMIZER = "adamw_8bit"
LOGGING_STEPS = 5

REL_PATTERN = re.compile(r"^([A-Z\-]+)\(([^/]+)/([^)]+)\)$")


def parse_csv_output(text: str) -> list[dict]:
    """Parse model CSV output into list of {name, relationship} dicts."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    results = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        parts = line.split(",", 2)
        if len(parts) < 2:
            continue
        name = parts[0].strip()
        rel = parts[1].strip()
        if name and REL_PATTERN.match(rel):
            results.append({"name": name.lower(), "relationship": rel})
    return results


def subsample_data(full_data: list[dict], num_articles: int, seed: int = 42) -> list[dict]:
    """Subsample training data to a fixed number of articles."""
    rng = random.Random(seed)
    if num_articles >= len(full_data):
        return full_data
    return rng.sample(full_data, num_articles)


def train_model(train_data: list[dict], output_dir: Path, checkpoint_dir: Path):
    """Train a bf16 LoRA adapter on Qwen3.5-27B."""
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    print(f"Training on {len(train_data)} examples, saving to {output_dir}")

    # bf16 LoRA (NOT QLoRA 4-bit) per Unsloth docs for Qwen3.5
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        max_seq_length=MAX_SEQ_LENGTH,
    )
    model.print_trainable_parameters()

    def formatting_func(examples):
        texts = []
        for convos in examples["conversations"]:
            text = tokenizer.apply_chat_template(convos, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        return {"text": texts}

    dataset = Dataset.from_list(train_data)
    dataset = dataset.map(formatting_func, batched=True)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_steps = (len(train_data) * EPOCHS) // (BATCH_SIZE * GRAD_ACCUM)
    save_steps = max(total_steps // 3, 10)

    training_args = SFTConfig(
        output_dir=str(checkpoint_dir),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_steps=min(WARMUP_STEPS, total_steps // 5),
        lr_scheduler_type=LR_SCHEDULER,
        optim=OPTIMIZER,
        bf16=True,
        logging_steps=LOGGING_STEPS,
        save_steps=save_steps,
        save_total_limit=2,
        seed=42,
        report_to="none",
        max_grad_norm=1.0,
        dataloader_num_workers=4,
        max_length=MAX_SEQ_LENGTH,
        eos_token=None,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Train only on assistant responses
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    print(f"Total optimizer steps: ~{total_steps}")
    t0 = time.time()
    train_result = trainer.train()
    elapsed = time.time() - t0

    print(f"Training complete in {elapsed/3600:.2f} hours, loss={train_result.training_loss:.4f}")

    # Save adapter
    adapter_path = str(output_dir / "lora_adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    stats = {
        "train_loss": train_result.training_loss,
        "train_runtime_seconds": elapsed,
        "train_samples": len(train_data),
        "epochs": EPOCHS,
        "total_steps": total_steps,
        "model": BASE_MODEL,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "max_seq_length": MAX_SEQ_LENGTH,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "quantization": "bf16 LoRA (not QLoRA)",
    }
    with open(output_dir / "training_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Free GPU memory before eval
    del trainer, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return stats


def evaluate_model(adapter_path: str, eval_data: list[dict], num_samples: int = 50) -> dict:
    """Evaluate the fine-tuned model."""
    from unsloth import FastLanguageModel
    from peft import PeftModel

    # Load in bf16 for eval (matches training; A100 has enough VRAM)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        load_in_16bit=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    FastLanguageModel.for_inference(model)

    # For VLM models, tokenizer may be a processor
    text_tokenizer = getattr(tokenizer, 'tokenizer', tokenizer)

    samples = eval_data[:num_samples]
    metrics = {
        "total_samples": num_samples,
        "format_valid": 0,
        "total_golden_rels": 0,
        "total_predicted_rels": 0,
        "name_matches": 0,
        "exact_matches": 0,
        "per_sample": [],
    }

    for i, ex in enumerate(samples):
        convos = ex["conversations"]
        system_msg = convos[0]["content"]
        user_msg = convos[1]["content"]
        golden_output = convos[2]["content"]

        golden_rels = parse_csv_output(golden_output)
        golden_names = {r["name"] for r in golden_rels}
        golden_pairs = {(r["name"], r["relationship"]) for r in golden_rels}

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        # Disable thinking mode so model generates CSV directly
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=False,
        )
        input_ids = text_tokenizer.encode(text, return_tensors="pt").to(model.device)
        attention_mask = torch.ones_like(input_ids)
        input_len = input_ids.shape[1]

        if input_len > MAX_SEQ_LENGTH - 1024:
            print(f"  [{i+1}/{len(samples)}] {ex['title']}: input too long, skipping")
            continue

        t0 = time.time()
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=4096,
            temperature=0.1,
            top_p=0.95,
            do_sample=True,
        )
        elapsed = time.time() - t0

        generated = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

        # Debug: print raw output for first 3 samples
        if i < 3:
            print(f"  [DEBUG] Raw output (first 500 chars): {generated[:500]}")
        pred_rels = parse_csv_output(generated)
        pred_names = {r["name"] for r in pred_rels}
        pred_pairs = {(r["name"], r["relationship"]) for r in pred_rels}

        format_ok = len(pred_rels) > 0
        name_hits = len(golden_names & pred_names)
        exact_hits = len(golden_pairs & pred_pairs)

        metrics["format_valid"] += int(format_ok)
        metrics["total_golden_rels"] += len(golden_rels)
        metrics["total_predicted_rels"] += len(pred_rels)
        metrics["name_matches"] += name_hits
        metrics["exact_matches"] += exact_hits

        sample_info = {
            "qid": ex["qid"],
            "title": ex["title"],
            "golden_count": len(golden_rels),
            "predicted_count": len(pred_rels),
            "name_recall": name_hits / len(golden_names) if golden_names else 0,
            "exact_match_rate": exact_hits / len(golden_rels) if golden_rels else 0,
            "format_valid": format_ok,
            "time_seconds": round(elapsed, 1),
        }
        metrics["per_sample"].append(sample_info)

        print(
            f"  [{i+1}/{len(samples)}] {ex['title']}: "
            f"golden={len(golden_rels)}, pred={len(pred_rels)}, "
            f"name_recall={sample_info['name_recall']:.2f}, "
            f"exact_match={sample_info['exact_match_rate']:.2f}, "
            f"time={elapsed:.1f}s"
        )

    n = len(metrics["per_sample"])
    if n > 0:
        metrics["summary"] = {
            "samples_evaluated": n,
            "format_compliance": metrics["format_valid"] / n,
            "avg_name_recall": sum(s["name_recall"] for s in metrics["per_sample"]) / n,
            "avg_exact_match_rate": sum(s["exact_match_rate"] for s in metrics["per_sample"]) / n,
            "overall_name_recall": metrics["name_matches"] / metrics["total_golden_rels"] if metrics["total_golden_rels"] else 0,
            "overall_exact_match": metrics["exact_matches"] / metrics["total_golden_rels"] if metrics["total_golden_rels"] else 0,
            "avg_time_per_sample": sum(s["time_seconds"] for s in metrics["per_sample"]) / n,
        }

    # P/R/F1
    golden = metrics["total_golden_rels"]
    predicted = metrics["total_predicted_rels"]
    for label, key in [("name", "name_matches"), ("tuple", "exact_matches")]:
        tp = metrics[key]
        fp = predicted - tp
        fn = golden - tp
        p = tp / predicted if predicted else 0
        r = tp / golden if golden else 0
        f1 = 2 * p * r / (p + r) if (p + r) else 0
        print(f"  {label.title()}: P={p:.1%}  R={r:.1%}  F1={f1:.1%}  TP={tp}  FP={fp}  FN={fn}")

    # Strip per-sample details for file
    summary_metrics = {k: v for k, v in metrics.items() if k != "per_sample"}
    summary_metrics["per_sample_count"] = n

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return summary_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1000, help="Number of articles to train on")
    parser.add_argument("--num-eval", type=int, default=50, help="Number of eval samples")
    args = parser.parse_args()

    size = args.size
    tag = f"1000art_qwen35_27b"

    print(f"\n{'='*72}")
    print(f"  TRAINING: Qwen3.5-27B with {size} articles (bf16 LoRA)")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*72}\n")

    output_dir = BASE_DIR / "output" / tag
    checkpoint_dir = BASE_DIR / "checkpoints" / tag
    eval_output = BASE_DIR / "data" / f"eval_results_{tag}.json"

    # Load data
    full_train = json.loads(FULL_TRAIN_PATH.read_text())
    eval_data = json.loads(EVAL_PATH.read_text())

    # Subsample (same seed=42 as other 1000-art runs)
    train_subset = subsample_data(full_train, size)
    print(f"Subsampled {len(train_subset)} training examples from {len(full_train)}")

    # Save subset for reproducibility
    subset_path = BASE_DIR / "data" / f"train_{tag}.json"
    subset_path.write_text(json.dumps(train_subset))
    print(f"Saved training subset to {subset_path}")

    # Train
    print(f"\n--- Training [{time.strftime('%H:%M:%S')}] ---")
    train_stats = train_model(train_subset, output_dir, checkpoint_dir)
    print(f"Training done [{time.strftime('%H:%M:%S')}]: loss={train_stats['train_loss']:.4f}, "
          f"time={train_stats['train_runtime_seconds']/60:.1f}min")

    # Evaluate
    adapter_path = str(output_dir / "lora_adapter")
    print(f"\n--- Evaluation [{time.strftime('%H:%M:%S')}] ---")
    eval_results = evaluate_model(adapter_path, eval_data, num_samples=args.num_eval)

    Path(eval_output).write_text(json.dumps(eval_results, indent=2))
    print(f"\nEval results saved to {eval_output}")

    # Print summary
    print(f"\n{'='*72}")
    print(f"  SUMMARY: Qwen3.5-27B ({size} articles)")
    print(f"{'='*72}")
    print(f"  Train loss:        {train_stats['train_loss']:.4f}")
    print(f"  Train time:        {train_stats['train_runtime_seconds']/60:.1f} min")
    print(f"  Format compliance: {eval_results['summary']['format_compliance']:.1%}")
    print(f"  Avg time/sample:   {eval_results['summary']['avg_time_per_sample']:.1f}s")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
