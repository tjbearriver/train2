#!/usr/bin/env python3
"""Eval-only script for a pre-trained adapter. Reuses logic from train_model_compare.py."""

import gc
import json
import re
import sys
import time
from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent
EVAL_PATH = BASE_DIR / "data" / "eval.json"
MAX_SEQ_LENGTH = 8192
REL_PATTERN = re.compile(r"^([A-Z\-]+)\(([^/]+)/([^)]+)\)$")


def parse_csv_output(text):
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


def main():
    model_name = sys.argv[1]  # e.g. unsloth/Qwen3.5-4B
    adapter_path = sys.argv[2]  # e.g. output/1000art_qwen35_4b/lora_adapter
    output_path = sys.argv[3]  # e.g. data/eval_results_1000art_qwen35_4b.json
    num_samples = int(sys.argv[4]) if len(sys.argv) > 4 else 50

    from unsloth import FastLanguageModel
    from peft import PeftModel

    print(f"Loading {model_name} + adapter from {adapter_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name, max_seq_length=MAX_SEQ_LENGTH, load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    FastLanguageModel.for_inference(model)

    # For VLM models, tokenizer is a processor — get the underlying text tokenizer
    text_tokenizer = getattr(tokenizer, 'tokenizer', tokenizer)

    eval_data = json.loads(EVAL_PATH.read_text())
    samples = eval_data[:num_samples]

    metrics = {
        "total_samples": num_samples, "format_valid": 0,
        "total_golden_rels": 0, "total_predicted_rels": 0,
        "name_matches": 0, "exact_matches": 0, "per_sample": [],
    }

    for i, ex in enumerate(samples):
        convos = ex["conversations"]
        system_msg, user_msg, golden_output = convos[0]["content"], convos[1]["content"], convos[2]["content"]
        golden_rels = parse_csv_output(golden_output)
        golden_names = {r["name"] for r in golden_rels}
        golden_pairs = {(r["name"], r["relationship"]) for r in golden_rels}

        messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        input_ids = text_tokenizer.encode(text, return_tensors="pt").to(model.device)
        attention_mask = torch.ones_like(input_ids)
        input_len = input_ids.shape[1]

        if input_len > MAX_SEQ_LENGTH - 1024:
            print(f"  [{i+1}/{len(samples)}] {ex['title']}: input too long, skipping")
            continue

        t0 = time.time()
        outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=2048, temperature=0.1, top_p=0.95, do_sample=True,
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
            "qid": ex["qid"], "title": ex["title"],
            "golden_count": len(golden_rels), "predicted_count": len(pred_rels),
            "name_recall": name_hits / len(golden_names) if golden_names else 0,
            "exact_match_rate": exact_hits / len(golden_rels) if golden_rels else 0,
            "format_valid": format_ok, "time_seconds": round(elapsed, 1),
        }
        metrics["per_sample"].append(sample_info)
        print(f"  [{i+1}/{len(samples)}] {ex['title']}: golden={len(golden_rels)}, pred={len(pred_rels)}, "
              f"name_recall={sample_info['name_recall']:.2f}, exact_match={sample_info['exact_match_rate']:.2f}, time={elapsed:.1f}s")

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

    summary = {k: v for k, v in metrics.items() if k != "per_sample"}
    summary["per_sample_count"] = n
    Path(output_path).write_text(json.dumps(summary, indent=2))

    # Compute P/R/F1
    golden = summary["total_golden_rels"]
    predicted = summary["total_predicted_rels"]
    for label, key in [("name", "name_matches"), ("tuple", "exact_matches")]:
        tp = summary[key]
        fp = predicted - tp
        fn = golden - tp
        p = tp / predicted if predicted else 0
        r = tp / golden if golden else 0
        f1 = 2 * p * r / (p + r) if (p + r) else 0
        print(f"  {label.title()}: P={p:.1%}  R={r:.1%}  F1={f1:.1%}  TP={tp}  FP={fp}  FN={fn}")

    print(f"\nResults saved to {output_path}")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
