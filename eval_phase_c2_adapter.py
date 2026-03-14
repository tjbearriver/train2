#!/usr/bin/env python3
"""Eval-only runner for Phase C2 adapters using bf16 base weights.

Usage:
  python -u eval_phase_c2_adapter.py --tag phase_c2_r8 --rank 8
"""

import argparse
import gc
import json
import time
from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_NAME = "unsloth/Qwen3.5-9B"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved Phase C2 adapter")
    parser.add_argument("--tag", required=True, help="Run tag, e.g. phase_c2_r8")
    parser.add_argument("--rank", type=int, required=True, help="LoRA rank for reporting only")
    parser.add_argument("--samples", type=int, default=50, help="Number of eval samples")
    return parser.parse_args()


def main():
    from train_phase_c2 import compute_comparison, evaluate_model

    args = parse_args()
    output_dir = BASE_DIR / "output" / args.tag
    adapter_path = str(output_dir / "lora_adapter")
    eval_output = DATA_DIR / f"eval_results_{args.tag}.json"
    eval_data = json.loads((DATA_DIR / "eval.json").read_text())

    print(f"\n{'='*72}")
    print(f"  PHASE C2 EVAL RECOVERY — {args.tag}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Adapter: {adapter_path}")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*72}\n")

    eval_results = evaluate_model(adapter_path, eval_data, args.samples)
    eval_output.write_text(json.dumps(eval_results, indent=2))
    print(f"Eval done [{time.strftime('%H:%M:%S')}]")

    gc.collect()
    torch.cuda.empty_cache()

    comp = compute_comparison(eval_results)
    print(f"\n{'='*72}")
    print(f"  PHASE C2 RESULTS: r={args.rank} — recovery eval")
    print(f"{'='*72}")
    print(f"  Format compliance: {comp['format_compliance']:.1%}")
    print(f"  Name:  P={comp['name']['precision']:.1%}  R={comp['name']['recall']:.1%}  F1={comp['name']['f1']:.1%}  TP={comp['name']['tp']}  FP={comp['name']['fp']}  FN={comp['name']['fn']}")
    print(f"  Tuple: P={comp['tuple']['precision']:.1%}  R={comp['tuple']['recall']:.1%}  F1={comp['tuple']['f1']:.1%}  TP={comp['tuple']['tp']}  FP={comp['tuple']['fp']}  FN={comp['tuple']['fn']}")
    print(f"  Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n  >>> Tuple F1 = {comp['tuple']['f1']:.1%} (r={args.rank}, recovery eval) <<<")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()