#!/usr/bin/env python3
"""Standalone eval for the Qwen3.5-4B-Claude-Opus model using saved adapter.
Runs with unbuffered output so progress is visible in real time.
"""
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
LOG_FILE = BASE_DIR / "eval_opus_progress.log"
RESUME_FILE = BASE_DIR / "eval_opus_resume.json"

MODEL_NAME = "Jackrong/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled"
ADAPTER_PATH = str(BASE_DIR / "output" / "1000art_qwen35_4b_opus" / "lora_adapter")
TAG = "1000art_qwen35_4b_opus"


def log(msg: str):
    """Write to log file with immediate flush."""
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
        f.flush()

REL_PATTERN = re.compile(r"^([A-Z\-]+)\(([^/]+)/([^)]+)\)$")


def parse_csv_output(text: str) -> list[dict]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    results = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        m = REL_PATTERN.match(line)
        if m:
            results.append({"relationship": m.group(1), "name": f"{m.group(2)}/{m.group(3)}"})
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            results.append({"name": parts[0], "relationship": parts[1]})
    return results


def main():
    # Resume from previous run if available
    completed_indices = set()
    prev_metrics = None
    if RESUME_FILE.exists():
        prev_metrics = json.loads(RESUME_FILE.read_text())
        completed_indices = {s["index"] for s in prev_metrics["per_sample"]}
        log(f"Resuming from {len(completed_indices)} completed samples")
    else:
        LOG_FILE.write_text("")

    eval_data = json.loads(EVAL_PATH.read_text())
    num_samples = 50
    samples = eval_data[:num_samples]

    log(f"=== Eval: {MODEL_NAME} ===")
    log(f"Adapter: {ADAPTER_PATH}")
    log(f"Samples: {num_samples}")
    log(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    from unsloth import FastLanguageModel
    from peft import PeftModel

    log("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    FastLanguageModel.for_inference(model)
    log("Model loaded.")

    text_tokenizer = getattr(tokenizer, 'tokenizer', tokenizer)

    if prev_metrics:
        metrics = prev_metrics
    else:
        metrics = {
            "total_samples": num_samples,
            "format_valid": 0,
            "total_golden_rels": 0,
            "total_predicted_rels": 0,
            "name_matches": 0,
            "exact_matches": 0,
            "per_sample": [],
        }

    eval_start = time.time()

    for i, ex in enumerate(samples):
        if i in completed_indices:
            continue
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
        # Force-close the <think> block so the model generates output directly
        if text.endswith("<think>\n"):
            text = text + "</think>\n"
        input_ids = text_tokenizer.encode(text, return_tensors="pt").to(model.device)
        attention_mask = torch.ones_like(input_ids)
        input_len = input_ids.shape[1]

        if input_len > MAX_SEQ_LENGTH - 1024:
            log(f"  [{i+1}/{len(samples)}] {ex['title']}: input too long, skipping")
            completed_indices.add(i)
            continue

        try:
            t0 = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=2048,
                    temperature=0.1,
                    top_p=0.95,
                    do_sample=True,
                )
            elapsed = time.time() - t0
            gen_tokens = outputs.shape[1] - input_len

            generated = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        except Exception as e:
            log(f"  [{i+1}/{len(samples)}] {ex['title']}: ERROR: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            continue
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
            "index": i,
            "qid": ex["qid"],
            "title": ex["title"],
            "golden_count": len(golden_rels),
            "predicted_count": len(pred_rels),
            "name_recall": name_hits / len(golden_names) if golden_names else 0,
            "exact_match_rate": exact_hits / len(golden_rels) if golden_rels else 0,
            "format_valid": format_ok,
            "time_seconds": round(elapsed, 1),
            "gen_tokens": gen_tokens,
        }
        metrics["per_sample"].append(sample_info)
        completed_indices.add(i)

        # Save resume checkpoint after each sample
        RESUME_FILE.write_text(json.dumps(metrics, indent=2))

        # Clear CUDA cache to prevent OOM
        del outputs, input_ids, attention_mask
        torch.cuda.empty_cache()
        gc.collect()

        wall = time.time() - eval_start
        log(
            f"  [{i+1}/{len(samples)}] {ex['title']}: "
            f"golden={len(golden_rels)}, pred={len(pred_rels)}, "
            f"name_recall={sample_info['name_recall']:.2f}, "
            f"exact_match={sample_info['exact_match_rate']:.2f}, "
            f"time={elapsed:.1f}s, tokens={gen_tokens}, "
            f"wall={wall:.0f}s"
        )

    eval_elapsed = time.time() - eval_start
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
            "total_eval_time_seconds": round(eval_elapsed, 1),
        }

    # Save with per_sample for debugging, and summary version for comparison
    summary_metrics = {k: v for k, v in metrics.items() if k != "per_sample"}
    summary_metrics["per_sample_count"] = n

    eval_output = BASE_DIR / "data" / f"eval_results_{TAG}.json"
    eval_output.write_text(json.dumps(summary_metrics, indent=2))
    log(f"Saved eval results to {eval_output}")

    # Compute comparison
    golden = summary_metrics["total_golden_rels"]
    predicted = summary_metrics["total_predicted_rels"]
    for label, key in [("name", "name_matches"), ("tuple", "exact_matches")]:
        tp = summary_metrics[key]
        fp = predicted - tp
        fn = golden - tp
        precision = tp / predicted if predicted else 0
        recall = tp / golden if golden else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        log(f"  {label.title()}: P={precision:.1%}  R={recall:.1%}  F1={f1:.1%}  TP={tp}  FP={fp}  FN={fn}")

    log(f"  Format compliance: {summary_metrics['summary']['format_compliance']:.1%}")
    log(f"  Eval time: {eval_elapsed/60:.1f} min")
    log(f"  Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
