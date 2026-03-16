#!/usr/bin/env python3
"""Evaluate all checkpoints from train_llama32_3b_checkpoints.py and plot the results.

For each checkpoint saved in checkpoints/llama32_3b_ckpt/, loads the adapter,
runs the 50-sample eval, then plots Name F1 and Tuple F1 vs articles trained on.

Usage:
  nohup /home/tj/workspace/tj/cranberry/train2/.venv/bin/python -u \
    eval_checkpoints_llama32_3b.py > output/llama32_3b_ckpt/eval_checkpoints.log 2>&1 &
"""

import gc
import json
import re
import time
from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent
EVAL_PATH = BASE_DIR / "data" / "eval.json"
CHECKPOINT_DIR = BASE_DIR / "checkpoints" / "llama32_3b_ckpt"
OUTPUT_DIR = BASE_DIR / "output" / "llama32_3b_ckpt"
RESULTS_PATH = BASE_DIR / "data" / "eval_results_llama32_3b_ckpt_all.json"
PLOT_PATH = OUTPUT_DIR / "checkpoint_eval_curve.png"

MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"
MAX_SEQ_LENGTH = 8192
GRAD_ACCUM = 16
EPOCHS = 3
NUM_EVAL_SAMPLES = 50

REL_PATTERN = re.compile(r"^([A-Z\-]+)\(([^/]+)/([^)]+)\)$")


def parse_csv_output(text: str) -> list[dict]:
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


def find_checkpoints() -> list[tuple[int, Path]]:
    """Return sorted list of (step, path) for all checkpoints."""
    checkpoints = []
    for d in CHECKPOINT_DIR.iterdir():
        if d.is_dir() and d.name.startswith("checkpoint-"):
            try:
                step = int(d.name.split("-")[1])
                checkpoints.append((step, d))
            except (IndexError, ValueError):
                pass
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def step_to_articles(step: int) -> int:
    """Convert training step to approximate unique articles trained on."""
    # Each step = GRAD_ACCUM total examples across EPOCHS epochs
    # step × GRAD_ACCUM / EPOCHS = cumulative unique-article-equivalent
    return round(step * GRAD_ACCUM / EPOCHS)


def evaluate_checkpoint(checkpoint_path: Path, eval_data: list[dict], step: int) -> dict:
    """Load a checkpoint adapter, run eval, return metrics dict."""
    from unsloth import FastLanguageModel
    from peft import PeftModel

    print(f"\n[Step {step} | ~{step_to_articles(step)} articles] Loading {checkpoint_path.name}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(model, str(checkpoint_path))
    FastLanguageModel.for_inference(model)

    text_tokenizer = getattr(tokenizer, 'tokenizer', tokenizer)

    samples = eval_data[:NUM_EVAL_SAMPLES]
    total_golden = 0
    total_predicted = 0
    name_tp = 0
    tuple_tp = 0
    format_valid = 0
    per_sample = []

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
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        input_ids = text_tokenizer.encode(text, return_tensors="pt").to(model.device)
        attention_mask = torch.ones_like(input_ids)
        input_len = input_ids.shape[1]

        if input_len > MAX_SEQ_LENGTH - 1024:
            print(f"  [{i+1}/{len(samples)}] {ex['title']}: input too long, skipping")
            continue

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

        generated = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        pred_rels = parse_csv_output(generated)
        pred_names = {r["name"] for r in pred_rels}
        pred_pairs = {(r["name"], r["relationship"]) for r in pred_rels}

        n_hits = len(golden_names & pred_names)
        t_hits = len(golden_pairs & pred_pairs)

        format_valid += int(len(pred_rels) > 0)
        total_golden += len(golden_rels)
        total_predicted += len(pred_rels)
        name_tp += n_hits
        tuple_tp += t_hits

        per_sample.append({
            "title": ex["title"],
            "golden": len(golden_rels),
            "predicted": len(pred_rels),
            "name_recall": n_hits / len(golden_names) if golden_names else 0,
            "exact_match": t_hits / len(golden_rels) if golden_rels else 0,
        })
        print(
            f"  [{i+1}/{len(samples)}] {ex['title']}: "
            f"golden={len(golden_rels)}, pred={len(pred_rels)}, "
            f"name_recall={per_sample[-1]['name_recall']:.2f}, "
            f"exact_match={per_sample[-1]['exact_match']:.2f}, "
            f"time={elapsed:.1f}s"
        )

    n = len(per_sample)
    name_p = name_tp / total_predicted if total_predicted else 0
    name_r = name_tp / total_golden if total_golden else 0
    name_f1 = 2 * name_p * name_r / (name_p + name_r) if (name_p + name_r) else 0
    tuple_p = tuple_tp / total_predicted if total_predicted else 0
    tuple_r = tuple_tp / total_golden if total_golden else 0
    tuple_f1 = 2 * tuple_p * tuple_r / (tuple_p + tuple_r) if (tuple_p + tuple_r) else 0

    result = {
        "step": step,
        "articles_equivalent": step_to_articles(step),
        "format_compliance": format_valid / n if n else 0,
        "name_precision": name_p,
        "name_recall": name_r,
        "name_f1": name_f1,
        "tuple_precision": tuple_p,
        "tuple_recall": tuple_r,
        "tuple_f1": tuple_f1,
        "total_golden": total_golden,
        "total_predicted": total_predicted,
        "name_tp": name_tp,
        "tuple_tp": tuple_tp,
        "samples_evaluated": n,
    }
    print(
        f"  => Name F1={name_f1:.1%} (P={name_p:.1%} R={name_r:.1%})  "
        f"Tuple F1={tuple_f1:.1%} (P={tuple_p:.1%} R={tuple_r:.1%})"
    )

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return result


def evaluate_base_model(eval_data: list[dict]) -> dict:
    """Load the raw base model (no adapter), run eval, return metrics dict."""
    from unsloth import FastLanguageModel

    print(f"\n[Base model | 0 articles] Loading {MODEL_NAME} without adapter...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    text_tokenizer = getattr(tokenizer, 'tokenizer', tokenizer)

    samples = eval_data[:NUM_EVAL_SAMPLES]
    total_golden = 0
    total_predicted = 0
    name_tp = 0
    tuple_tp = 0
    format_valid = 0
    per_sample = []

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
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        input_ids = text_tokenizer.encode(text, return_tensors="pt").to(model.device)
        attention_mask = torch.ones_like(input_ids)
        input_len = input_ids.shape[1]

        if input_len > MAX_SEQ_LENGTH - 1024:
            print(f"  [{i+1}/{len(samples)}] {ex['title']}: input too long, skipping")
            continue

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

        generated = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        pred_rels = parse_csv_output(generated)
        pred_names = {r["name"] for r in pred_rels}
        pred_pairs = {(r["name"], r["relationship"]) for r in pred_rels}

        n_hits = len(golden_names & pred_names)
        t_hits = len(golden_pairs & pred_pairs)

        format_valid += int(len(pred_rels) > 0)
        total_golden += len(golden_rels)
        total_predicted += len(pred_rels)
        name_tp += n_hits
        tuple_tp += t_hits

        per_sample.append({
            "title": ex["title"],
            "golden": len(golden_rels),
            "predicted": len(pred_rels),
            "name_recall": n_hits / len(golden_names) if golden_names else 0,
            "exact_match": t_hits / len(golden_rels) if golden_rels else 0,
        })
        print(
            f"  [{i+1}/{len(samples)}] {ex['title']}: "
            f"golden={len(golden_rels)}, pred={len(pred_rels)}, "
            f"name_recall={per_sample[-1]['name_recall']:.2f}, "
            f"exact_match={per_sample[-1]['exact_match']:.2f}, "
            f"time={elapsed:.1f}s"
        )

    n = len(per_sample)
    name_p = name_tp / total_predicted if total_predicted else 0
    name_r = name_tp / total_golden if total_golden else 0
    name_f1 = 2 * name_p * name_r / (name_p + name_r) if (name_p + name_r) else 0
    tuple_p = tuple_tp / total_predicted if total_predicted else 0
    tuple_r = tuple_tp / total_golden if total_golden else 0
    tuple_f1 = 2 * tuple_p * tuple_r / (tuple_p + tuple_r) if (tuple_p + tuple_r) else 0

    result = {
        "step": 0,
        "articles_equivalent": 0,
        "label": "base",
        "format_compliance": format_valid / n if n else 0,
        "name_precision": name_p,
        "name_recall": name_r,
        "name_f1": name_f1,
        "tuple_precision": tuple_p,
        "tuple_recall": tuple_r,
        "tuple_f1": tuple_f1,
        "total_golden": total_golden,
        "total_predicted": total_predicted,
        "name_tp": name_tp,
        "tuple_tp": tuple_tp,
        "samples_evaluated": n,
    }
    print(
        f"  => Name F1={name_f1:.1%} (P={name_p:.1%} R={name_r:.1%})  "
        f"Tuple F1={tuple_f1:.1%} (P={tuple_p:.1%} R={tuple_r:.1%})"
    )

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return result


def plot_results(all_results: list[dict]):
    """Generate Name F1 and Tuple F1 vs articles curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    articles = [r["articles_equivalent"] for r in all_results]
    name_f1 = [r["name_f1"] * 100 for r in all_results]
    tuple_f1 = [r["tuple_f1"] * 100 for r in all_results]
    name_p = [r["name_precision"] * 100 for r in all_results]
    name_r = [r["name_recall"] * 100 for r in all_results]
    tuple_p = [r["tuple_precision"] * 100 for r in all_results]
    tuple_r = [r["tuple_recall"] * 100 for r in all_results]

    # Separate base (step=0) from fine-tuned points for distinct styling
    base_idx = next((i for i, r in enumerate(all_results) if r["step"] == 0), None)
    ft_articles = [a for i, a in enumerate(articles) if i != base_idx]
    ft_name_f1  = [v for i, v in enumerate(name_f1)  if i != base_idx]
    ft_tuple_f1 = [v for i, v in enumerate(tuple_f1) if i != base_idx]
    ft_name_p   = [v for i, v in enumerate(name_p)   if i != base_idx]
    ft_name_r   = [v for i, v in enumerate(name_r)   if i != base_idx]
    ft_tuple_p  = [v for i, v in enumerate(tuple_p)  if i != base_idx]
    ft_tuple_r  = [v for i, v in enumerate(tuple_r)  if i != base_idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle("Llama-3.2-3B (QLoRA 4-bit) — Learning Curve vs Articles Trained", fontsize=14, fontweight="bold")

    def _add_base_marker(ax, y_vals, colors):
        """Draw a vertical dashed line and open markers for the base model."""
        if base_idx is None:
            return
        ax.axvline(x=0, color="gray", linestyle=":", linewidth=1.2, alpha=0.7)
        for color, y in zip(colors, y_vals):
            ax.plot(0, y, marker="D", markersize=8, color=color,
                    markerfacecolor="white", markeredgewidth=2, zorder=5)
            ax.annotate(f"{y:.0f}", (0, y), textcoords="offset points",
                        xytext=(8, 0), fontsize=8, color=color, va="center")

    # Left: F1 curves
    ax1 = axes[0]
    ax1.plot(ft_articles, ft_name_f1, "b-o", linewidth=2, markersize=6, label="Name F1")
    ax1.plot(ft_articles, ft_tuple_f1, "r-o", linewidth=2, markersize=6, label="Tuple F1")
    _add_base_marker(ax1, [name_f1[base_idx], tuple_f1[base_idx]], ["blue", "red"])
    ax1.set_xlabel("Articles trained on (cumulative, incl. epochs)", fontsize=12)
    ax1.set_ylabel("F1 Score (%)", fontsize=12)
    ax1.set_title("F1 vs Articles", fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    # Annotate final values
    if ft_articles:
        ax1.annotate(f"{ft_name_f1[-1]:.1f}%", (ft_articles[-1], ft_name_f1[-1]),
                     textcoords="offset points", xytext=(5, 4), fontsize=9, color="blue")
        ax1.annotate(f"{ft_tuple_f1[-1]:.1f}%", (ft_articles[-1], ft_tuple_f1[-1]),
                     textcoords="offset points", xytext=(5, -12), fontsize=9, color="red")
    # Annotate each fine-tuned data point
    for i, (a, nf, tf) in enumerate(zip(ft_articles, ft_name_f1, ft_tuple_f1)):
        if i % 2 == 0:
            ax1.annotate(f"{nf:.0f}", (a, nf), textcoords="offset points", xytext=(0, 8),
                         fontsize=7, color="blue", ha="center")
            ax1.annotate(f"{tf:.0f}", (a, tf), textcoords="offset points", xytext=(0, -14),
                         fontsize=7, color="red", ha="center")

    # Right: Precision and Recall breakdown
    ax2 = axes[1]
    ax2.plot(ft_articles, ft_name_p, "b--s", linewidth=1.5, markersize=5, alpha=0.8, label="Name P")
    ax2.plot(ft_articles, ft_name_r, "b-s",  linewidth=1.5, markersize=5, alpha=0.8, label="Name R")
    ax2.plot(ft_articles, ft_tuple_p, "r--s", linewidth=1.5, markersize=5, alpha=0.8, label="Tuple P")
    ax2.plot(ft_articles, ft_tuple_r, "r-s",  linewidth=1.5, markersize=5, alpha=0.8, label="Tuple R")
    _add_base_marker(ax2,
                     [name_p[base_idx], name_r[base_idx], tuple_p[base_idx], tuple_r[base_idx]],
                     ["blue", "blue", "red", "red"])
    ax2.set_xlabel("Articles trained on (cumulative, incl. epochs)", fontsize=12)
    ax2.set_ylabel("Score (%)", fontsize=12)
    ax2.set_title("Precision & Recall vs Articles", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(PLOT_PATH), dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {PLOT_PATH}")
    plt.close()


def main():
    print(f"\n{'='*72}")
    print(f"  CHECKPOINT EVAL: Llama-3.2-3B (llama32_3b_ckpt)")
    print(f"  Checkpoint dir: {CHECKPOINT_DIR}")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*72}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoints = find_checkpoints()
    if not checkpoints:
        print(f"ERROR: No checkpoints found in {CHECKPOINT_DIR}")
        print("Run train_llama32_3b_checkpoints.py first.")
        return

    print(f"Found {len(checkpoints)} checkpoints:")
    for step, path in checkpoints:
        print(f"  Step {step:4d} (~{step_to_articles(step):5d} articles): {path.name}")

    # Load existing results so we can resume if interrupted
    all_results = []
    if RESULTS_PATH.exists():
        existing = json.loads(RESULTS_PATH.read_text())
        all_results = existing
        done_steps = {r["step"] for r in all_results}
        print(f"\nResuming: {len(done_steps)} checkpoints already evaluated.")
    else:
        done_steps = set()

    eval_data = json.loads(EVAL_PATH.read_text())
    print(f"Eval set: {len(eval_data)} examples, evaluating first {NUM_EVAL_SAMPLES}")

    # Evaluate the base model (step=0) first if not already done
    if 0 not in done_steps:
        base_result = evaluate_base_model(eval_data)
        all_results.append(base_result)
        all_results_sorted = sorted(all_results, key=lambda x: x["step"])
        RESULTS_PATH.write_text(json.dumps(all_results_sorted, indent=2))
        done_steps.add(0)
        print(f"  Base model result saved to {RESULTS_PATH}")
    else:
        print("  [Base model] already evaluated, skipping.")

    for step, ckpt_path in checkpoints:
        if step in done_steps:
            print(f"  [Step {step}] already evaluated, skipping.")
            continue

        result = evaluate_checkpoint(ckpt_path, eval_data, step)
        all_results.append(result)
        # Save after each checkpoint so we can resume
        all_results_sorted = sorted(all_results, key=lambda x: x["step"])
        RESULTS_PATH.write_text(json.dumps(all_results_sorted, indent=2))
        print(f"  Results saved to {RESULTS_PATH}")

    all_results_sorted = sorted(all_results, key=lambda x: x["step"])

    # Print summary table
    print(f"\n{'='*72}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*72}")
    print(f"  {'Step':>6}  {'Articles':>8}  {'Name F1':>8}  {'Name P':>7}  {'Name R':>7}  {'Tuple F1':>9}  {'Tuple P':>8}  {'Tuple R':>8}  {'Format':>8}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*9}  {'-'*8}  {'-'*8}  {'-'*8}")
    for r in all_results_sorted:
        step_label = "base" if r["step"] == 0 else str(r["step"])
        print(f"  {step_label:>6}  {r['articles_equivalent']:>8}  "
              f"{r['name_f1']:>7.1%}  {r['name_precision']:>6.1%}  {r['name_recall']:>6.1%}  "
              f"{r['tuple_f1']:>8.1%}  {r['tuple_precision']:>7.1%}  {r['tuple_recall']:>7.1%}  "
              f"{r['format_compliance']:>7.1%}")
    print(f"{'='*72}\n")

    # Generate plot
    print("Generating plot...")
    plot_results(all_results_sorted)

    print(f"\nAll done [{time.strftime('%H:%M:%S')}]")
    print(f"Results: {RESULTS_PATH}")
    print(f"Plot:    {PLOT_PATH}")


if __name__ == "__main__":
    main()
