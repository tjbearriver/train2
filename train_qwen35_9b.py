#!/usr/bin/env python3
"""Train and evaluate Qwen3.5-9B on 1000 articles using bf16 LoRA (NOT QLoRA).

Qwen3.5 models should NOT use QLoRA (4-bit) per Unsloth docs — bf16 LoRA instead.
Designed for RTX 5090 (32GB VRAM). Qwen3.5-9B bf16 LoRA needs ~22GB VRAM.

Usage:
  python train_qwen35_9b.py
"""

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
MAX_SEQ_LENGTH = 8192

MODEL_NAME = "unsloth/Qwen3.5-9B"
TAG = "1000art_qwen35_9b"

LORA_R = 64
LORA_ALPHA = 64
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

EPOCHS = 3
BATCH_SIZE = 1
GRAD_ACCUM = 16
LEARNING_RATE = 2e-4
WARMUP_STEPS = 20
LR_SCHEDULER = "cosine"
OPTIMIZER = "adamw_8bit"
LOGGING_STEPS = 5
NUM_ARTICLES = 1000

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
    """Train a LoRA adapter on the given data using bf16 LoRA (not QLoRA)."""
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    print(f"Training {MODEL_NAME} on {len(train_data)} examples (bf16 LoRA)")
    print(f"Output: {output_dir}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,      # NOT QLoRA — Qwen3.5 not recommended for 4-bit
        load_in_16bit=True,      # bf16 LoRA
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

    # Detect chat template format for train_on_responses_only
    sample_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
        tokenize=False, add_generation_prompt=False,
    )
    if "<|im_start|>" in sample_text:
        inst_part = "<|im_start|>user\n"
        resp_part = "<|im_start|>assistant\n"
    elif "<|start_header_id|>" in sample_text:
        inst_part = "<|start_header_id|>user<|end_header_id|>\n\n"
        resp_part = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    else:
        raise ValueError(f"Unknown chat template format. Sample: {sample_text[:200]}")
    print(f"Chat template: instruction_part={inst_part!r}, response_part={resp_part!r}")

    trainer = train_on_responses_only(
        trainer,
        instruction_part=inst_part,
        response_part=resp_part,
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
        "model_name": MODEL_NAME,
        "train_loss": train_result.training_loss,
        "train_runtime_seconds": elapsed,
        "train_samples": len(train_data),
        "epochs": EPOCHS,
        "total_steps": total_steps,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "quantization": "bf16 LoRA (not QLoRA)",
        "max_seq_length": MAX_SEQ_LENGTH,
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

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    FastLanguageModel.for_inference(model)

    # For VLM models, tokenizer might be a processor
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

        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
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
            max_new_tokens=2048,
            temperature=0.1,
            top_p=0.95,
            do_sample=True,
        )
        elapsed = time.time() - t0

        generated = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
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

    # Strip per-sample details for file size
    summary_metrics = {k: v for k, v in metrics.items() if k != "per_sample"}
    summary_metrics["per_sample_count"] = n
    return summary_metrics


def compute_comparison(eval_results: dict) -> dict:
    """Compute P/R/F1/FP from eval result counts."""
    golden = eval_results["total_golden_rels"]
    predicted = eval_results["total_predicted_rels"]
    out = {}
    for label, key in [("name", "name_matches"), ("tuple", "exact_matches")]:
        tp = eval_results[key]
        fp = predicted - tp
        fn = golden - tp
        precision = tp / predicted if predicted else 0
        recall = tp / golden if golden else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        out[label] = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
    out["total_golden"] = golden
    out["total_predicted"] = predicted
    out["format_compliance"] = eval_results["summary"]["format_compliance"]
    return out


def main():
    print(f"\n{'='*72}")
    print(f"  QWEN3.5-9B FINE-TUNING (bf16 LoRA, 1000 articles)")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*72}\n")

    # Paths
    output_dir = BASE_DIR / "output" / TAG
    checkpoint_dir = BASE_DIR / "checkpoints" / TAG
    eval_output = BASE_DIR / "data" / f"eval_results_{TAG}.json"
    train_data_path = BASE_DIR / "data" / f"train_{TAG}.json"

    # Load data
    full_train = json.loads(FULL_TRAIN_PATH.read_text())
    eval_data = json.loads(EVAL_PATH.read_text())

    # Subsample (same seed=42, same 1000 articles as other comparison runs)
    train_subset = subsample_data(full_train, NUM_ARTICLES)
    print(f"Subsampled {len(train_subset)} training examples from {len(full_train)}")

    # Save subsample for reproducibility
    train_data_path.write_text(json.dumps(train_subset))
    print(f"Saved training subset to {train_data_path}")

    # Train
    print(f"\n--- Training [{time.strftime('%H:%M:%S')}] ---")
    train_stats = train_model(train_subset, output_dir, checkpoint_dir)
    print(f"Training done [{time.strftime('%H:%M:%S')}]: loss={train_stats['train_loss']:.4f}, "
          f"time={train_stats['train_runtime_seconds']/60:.1f}min")

    # Evaluate
    adapter_path = str(output_dir / "lora_adapter")
    print(f"\n--- Evaluation [{time.strftime('%H:%M:%S')}] ---")
    eval_results = evaluate_model(adapter_path, eval_data, 50)
    eval_output.write_text(json.dumps(eval_results, indent=2))
    print(f"Eval done [{time.strftime('%H:%M:%S')}]")

    # Free GPU memory
    gc.collect()
    torch.cuda.empty_cache()

    # Compare
    comp = compute_comparison(eval_results)
    print(f"\n{'='*72}")
    print(f"  RESULTS: {MODEL_NAME} — {NUM_ARTICLES} articles")
    print(f"{'='*72}")
    print(f"  Format compliance: {comp['format_compliance']:.1%}")
    print(f"  Name:  P={comp['name']['precision']:.1%}  R={comp['name']['recall']:.1%}  F1={comp['name']['f1']:.1%}  TP={comp['name']['tp']}  FP={comp['name']['fp']}  FN={comp['name']['fn']}")
    print(f"  Tuple: P={comp['tuple']['precision']:.1%}  R={comp['tuple']['recall']:.1%}  F1={comp['tuple']['f1']:.1%}  TP={comp['tuple']['tp']}  FP={comp['tuple']['fp']}  FN={comp['tuple']['fn']}")
    print(f"  Train loss: {train_stats['train_loss']:.4f}")
    print(f"  Train time: {train_stats['train_runtime_seconds']/60:.1f} min")
    print(f"  Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
