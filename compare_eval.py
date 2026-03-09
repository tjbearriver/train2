#!/usr/bin/env python3
"""Compare base vs fine-tuned eval results with precision/recall/F1 and FP analysis."""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def compute_metrics(d: dict, label: str) -> dict:
    """Compute P/R/F1 from stored eval result counts."""
    golden = d["total_golden_rels"]
    predicted = d["total_predicted_rels"]

    out = {"label": label}
    for match_label, match_key in [("name", "name_matches"), ("tuple", "exact_matches")]:
        tp = d[match_key]
        fp = predicted - tp
        fn = golden - tp

        precision = tp / predicted if predicted else 0
        recall = tp / golden if golden else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

        out[match_label] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
        }

    out["total_golden"] = golden
    out["total_predicted"] = predicted
    out["format_valid"] = d["format_valid"]
    out["samples"] = d["summary"]["samples_evaluated"]
    out["format_compliance"] = d["summary"]["format_compliance"]
    out["avg_time"] = d["summary"]["avg_time_per_sample"]
    return out


def fmt_pct(v: float) -> str:
    return f"{v:>7.1%}"


def print_section(title: str, key: str, base: dict, ft: dict):
    b, f = base[key], ft[key]
    print(f"\n  {title}")
    print(f"  {'─' * 66}")
    print(f"  {'':32} {'Base Qwen3-8B':>15} {'Fine-tuned':>15}")
    print(f"  {'Precision':32} {fmt_pct(b['precision']):>15} {fmt_pct(f['precision']):>15}")
    print(f"  {'Recall':32} {fmt_pct(b['recall']):>15} {fmt_pct(f['recall']):>15}")
    print(f"  {'F1':32} {fmt_pct(b['f1']):>15} {fmt_pct(f['f1']):>15}")
    print()
    print(f"  {'True Positives (TP)':32} {b['tp']:>15} {f['tp']:>15}")
    print(f"  {'False Positives (FP)':32} {b['fp']:>15} {f['fp']:>15}")
    print(f"  {'False Negatives (FN)':32} {b['fn']:>15} {f['fn']:>15}")


def main():
    base_path = BASE_DIR / "data" / "eval_results_base.json"
    ft_path = BASE_DIR / "data" / "eval_results_finetuned.json"

    base = compute_metrics(json.loads(base_path.read_text()), "Base Qwen3-8B")
    ft = compute_metrics(json.loads(ft_path.read_text()), "Fine-tuned")

    report = []
    def out(s=""):
        report.append(s)
        print(s)

    out("=" * 72)
    out("  Evaluation Comparison: Base Qwen3-8B vs Fine-tuned (LoRA)")
    out("=" * 72)
    out(f"\n  Eval samples:      {base['samples']}")
    out(f"  Golden relations:  {base['total_golden']}")
    out(f"  Base predicted:    {base['total_predicted']}")
    out(f"  FT predicted:      {ft['total_predicted']}")

    # General
    out(f"\n  {'':32} {'Base Qwen3-8B':>15} {'Fine-tuned':>15}")
    out(f"  {'─' * 66}")
    out(f"  {'Format compliance':32} {fmt_pct(base['format_compliance']):>15} {fmt_pct(ft['format_compliance']):>15}")
    out(f"  {'Avg time/sample (s)':32} {base['avg_time']:>14.1f}s {ft['avg_time']:>14.1f}s")

    # Name-level
    print_section("Name-level matching", "name", base, ft)
    for line in report[-7:]:
        pass  # already printed

    # Tuple-level
    print_section("Name+Relationship tuple matching", "tuple", base, ft)

    out(f"\n{'=' * 72}")

    # Save report
    log_path = BASE_DIR / "data" / "eval_comparison.log"
    # Re-capture everything since print_section also printed
    # Just write the full report
    return base, ft


if __name__ == "__main__":
    base, ft = main()

    # Save structured log
    log_path = BASE_DIR / "data" / "eval_comparison.log"
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main()
    log_path.write_text(buf.getvalue())
    print(f"\nReport saved to {log_path}")
