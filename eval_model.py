#!/usr/bin/env python3
"""Evaluate a model on the held-out eval set. Works for both base and fine-tuned models.

Usage:
  # Base model (4-bit):
  python eval_model.py --base

  # Fine-tuned model (LoRA adapter):
  python eval_model.py --adapter ./checkpoints/final

  # Control sample size:
  python eval_model.py --base --num-samples 50
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

from unsloth import FastLanguageModel

BASE_DIR = Path(__file__).resolve().parent
EVAL_PATH = BASE_DIR / "data" / "eval.json"
BASE_MODEL = "unsloth/Qwen3-8B-bnb-4bit"
MAX_SEQ_LENGTH = 8192

REL_PATTERN = re.compile(r"^([A-Z\-]+)\(([^/]+)/([^)]+)\)$")


def parse_csv_output(text: str) -> list[dict]:
    """Parse model CSV output into list of {name, relationship} dicts."""
    # Strip Qwen3 think blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    results = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        # Split on first two commas: name, relationship, evidence...
        parts = line.split(",", 2)
        if len(parts) < 2:
            continue
        name = parts[0].strip()
        rel = parts[1].strip()
        if name and REL_PATTERN.match(rel):
            results.append({"name": name.lower(), "relationship": rel})
    return results


def evaluate(model, tokenizer, examples: list[dict], num_samples: int) -> dict:
    """Run evaluation and compute metrics."""
    samples = examples[:num_samples]
    metrics = {
        "total_samples": len(samples),
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

        # Parse golden
        golden_rels = parse_csv_output(golden_output)
        golden_names = {r["name"] for r in golden_rels}
        golden_pairs = {(r["name"], r["relationship"]) for r in golden_rels}

        # Build prompt using chat template
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        tokenized = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = tokenized["input_ids"].to(model.device)
        attention_mask = tokenized["attention_mask"].to(model.device)

        input_len = input_ids.shape[1]
        if input_len > MAX_SEQ_LENGTH - 1024:
            print(f"  [{i+1}/{len(samples)}] {ex['title']}: input too long ({input_len} tokens), skipping")
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

        # Parse predicted
        pred_rels = parse_csv_output(generated)
        pred_names = {r["name"] for r in pred_rels}
        pred_pairs = {(r["name"], r["relationship"]) for r in pred_rels}

        # Metrics for this sample
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

    # Aggregate
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

    return metrics


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--base", action="store_true", help="Evaluate the base Qwen3-8B model")
    group.add_argument("--adapter", type=str, help="Path to LoRA adapter directory")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of eval samples")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    print(f"Loading eval data from {EVAL_PATH}")
    with open(EVAL_PATH) as f:
        eval_data = json.load(f)
    print(f"Eval set: {len(eval_data)} examples, evaluating {args.num_samples}")

    print(f"\nLoading model: {BASE_MODEL}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    if args.adapter:
        print(f"Loading adapter: {args.adapter}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)

    FastLanguageModel.for_inference(model)

    print("\n=== Running evaluation ===")
    metrics = evaluate(model, tokenizer, eval_data, args.num_samples)

    if "summary" in metrics:
        print("\n=== Summary ===")
        for k, v in metrics["summary"].items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    # Save results
    out_path = args.output
    if out_path is None:
        tag = "base" if args.base else "finetuned"
        out_path = str(BASE_DIR / "data" / f"eval_results_{tag}.json")

    # Don't save per-sample details to keep file manageable
    save_metrics = {k: v for k, v in metrics.items() if k != "per_sample"}
    save_metrics["per_sample_count"] = len(metrics["per_sample"])

    with open(out_path, "w") as f:
        json.dump(save_metrics, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
