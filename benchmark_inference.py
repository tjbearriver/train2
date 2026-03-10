#!/usr/bin/env python3
"""Benchmark inference tok/s for each fine-tuned model variant.

Uses the same 5 eval articles across all models for apples-to-apples comparison.
Reports input tokens, output tokens, wall-clock time, and tok/s (output only).
"""

import gc
import json
import time
from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent
EVAL_PATH = BASE_DIR / "data" / "eval.json"
MAX_SEQ_LENGTH = 8192

# 5 eval articles at indices 0-4 (same across all models)
SAMPLE_INDICES = [0, 1, 2, 3, 4]

MODELS = [
    {
        "tag": "Qwen3-8B (full 4726-art)",
        "base": "unsloth/Qwen3-8B-bnb-4bit",
        "adapter": "output/lora_adapter",
    },
    {
        "tag": "Qwen3-8B (1000-art)",
        "base": "unsloth/Qwen3-8B-bnb-4bit",
        "adapter": "output/ablation_1000art/lora_adapter",
    },
    {
        "tag": "Gemma3-4B (1000-art)",
        "base": "unsloth/gemma-3-4b-it-bnb-4bit",
        "adapter": "output/1000art_gemma3_4b/lora_adapter",
    },
    {
        "tag": "Qwen3.5-4B (1000-art)",
        "base": "unsloth/Qwen3.5-4B",
        "adapter": "output/1000art_qwen35_4b/lora_adapter",
    },
]


def load_samples():
    with open(EVAL_PATH) as f:
        data = json.load(f)
    return [data[i] for i in SAMPLE_INDICES]


def benchmark_model(model_cfg, samples):
    from unsloth import FastLanguageModel
    from peft import PeftModel

    tag = model_cfg["tag"]
    print(f"\n{'='*70}")
    print(f"  Benchmarking: {tag}")
    print(f"  Base: {model_cfg['base']}")
    print(f"  Adapter: {model_cfg['adapter']}")
    print(f"{'='*70}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["base"],
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(model, str(BASE_DIR / model_cfg["adapter"]))
    FastLanguageModel.for_inference(model)

    # Get the underlying text tokenizer (needed for VLM models)
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

    results = []
    for i, sample in enumerate(samples):
        title = sample["title"]
        messages = [
            {"role": "system", "content": sample["conversations"][0]["content"]},
            {"role": "user", "content": sample["conversations"][1]["content"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        input_ids = text_tokenizer.encode(text, return_tensors="pt").to(model.device)
        input_len = input_ids.shape[1]

        if input_len > MAX_SEQ_LENGTH - 256:
            print(f"  [{i+1}/5] {title}: SKIPPED (input {input_len} tokens)")
            continue

        attention_mask = torch.ones_like(input_ids)

        # Warm-up on first sample (don't count it separately, but still record)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=2048,
                temperature=0.1,
                top_p=0.95,
                do_sample=True,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        output_len = outputs.shape[1] - input_len
        toks = output_len / elapsed if elapsed > 0 else 0

        results.append({
            "title": title,
            "input_tokens": input_len,
            "output_tokens": output_len,
            "time_s": elapsed,
            "tok_s": toks,
        })
        print(
            f"  [{i+1}/5] {title}: "
            f"in={input_len}, out={output_len}, "
            f"time={elapsed:.1f}s, "
            f"tok/s={toks:.1f}"
        )

    # Summary
    if results:
        avg_toks = sum(r["tok_s"] for r in results) / len(results)
        total_out = sum(r["output_tokens"] for r in results)
        total_time = sum(r["time_s"] for r in results)
        overall_toks = total_out / total_time if total_time > 0 else 0
        print(f"\n  >> {tag}: avg={avg_toks:.1f} tok/s, overall={overall_toks:.1f} tok/s ({len(results)} articles)")
    else:
        avg_toks = 0
        overall_toks = 0
        print(f"\n  >> {tag}: no articles benchmarked")

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "tag": tag,
        "base": model_cfg["base"],
        "per_article": results,
        "avg_tok_s": avg_toks,
        "overall_tok_s": overall_toks,
    }


def main():
    samples = load_samples()
    print(f"Loaded {len(samples)} eval articles for benchmarking:")
    for i, s in enumerate(samples):
        print(f"  {i+1}. {s['title']} (QID: {s['qid']})")

    all_results = []
    for model_cfg in MODELS:
        result = benchmark_model(model_cfg, samples)
        all_results.append(result)

    # Final comparison
    print(f"\n{'='*70}")
    print("  INFERENCE SPEED COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Model':<30} {'Avg tok/s':>10} {'Overall tok/s':>14}")
    print(f"  {'-'*30} {'-'*10} {'-'*14}")
    for r in sorted(all_results, key=lambda x: x["overall_tok_s"], reverse=True):
        print(f"  {r['tag']:<30} {r['avg_tok_s']:>10.1f} {r['overall_tok_s']:>14.1f}")

    # Save results
    out_path = BASE_DIR / "data" / "benchmark_inference.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
