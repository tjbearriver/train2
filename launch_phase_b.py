#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["runpod"]
# ///
"""Launch Phase B LR sweep on two RunPod 5090 pods (parallel).

Provisions two pods — one for lr=1e-4, one for lr=5e-5 — each with separate
output directories so they can run fully in parallel.

Usage:
  export RUNPOD_API_KEY=rpa_...
  uv run launch_phase_b.py
"""

from runpod_launch import launch

PHASE_B_RUNS = [
    {
        "phase": "B_lr1e4",
        "pod_name": "phase-b-lr1e4-qwen35-9b",
        "output_subdir": "output/phase_b_lr1e4",
        "lr": 1e-4,
        "tag": "phase_b_lr1e4",
    },
    {
        "phase": "B_lr5e5",
        "pod_name": "phase-b-lr5e5-qwen35-9b",
        "output_subdir": "output/phase_b_lr5e5",
        "lr": 5e-5,
        "tag": "phase_b_lr5e5",
    },
]

FILES_TO_UPLOAD = [
    "train_phase_b.py",
    "data/train.json",
    "data/eval.json",
]


if __name__ == "__main__":
    for run in PHASE_B_RUNS:
        train_command = (
            f"cd /workspace/train7 && "
            f"mkdir -p {run['output_subdir']} && "
            f"nohup bash -c 'python -u train_phase_b.py --lr {run['lr']} --tag {run['tag']} 2>&1 | tee {run['output_subdir']}/train.log' "
            f"> /dev/null 2>&1 &"
        )

        print(f"\n{'='*60}")
        print(f"  Launching {run['phase']}: lr={run['lr']}")
        print(f"{'='*60}")

        launch(
            phase=run["phase"],
            pod_name=run["pod_name"],
            train_script="train_phase_b.py",
            files_to_upload=FILES_TO_UPLOAD,
            output_subdir=run["output_subdir"],
            train_command=train_command,
        )
