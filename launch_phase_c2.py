#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["runpod"]
# ///
"""Launch Phase C2 LoRA rank sweep (full data) on two RunPod 5090 pods (parallel).

Provisions two pods — one for r=8, one for r=16 — each with
separate output directories so they can run fully in parallel.

Usage:
  export RUNPOD_API_KEY=rpa_...
  uv run launch_phase_c2.py
"""

from runpod_launch import launch

PHASE_C2_RUNS = [
    {
        "phase": "C2_r8",
        "pod_name": "phase-c2-r8-qwen35-9b",
        "output_subdir": "output/phase_c2_r8",
        "rank": 8,
        "tag": "phase_c2_r8",
    },
    {
        "phase": "C2_r16",
        "pod_name": "phase-c2-r16-qwen35-9b",
        "output_subdir": "output/phase_c2_r16",
        "rank": 16,
        "tag": "phase_c2_r16",
    },
]

FILES_TO_UPLOAD = [
    "train_phase_c2.py",
    "data/train.json",
    "data/eval.json",
]


if __name__ == "__main__":
    for run in PHASE_C2_RUNS:
        train_command = (
            f"cd /workspace/train7 && "
            f"mkdir -p {run['output_subdir']} && "
            f"nohup bash -c 'python -u train_phase_c2.py --rank {run['rank']} --tag {run['tag']} 2>&1 | tee {run['output_subdir']}/train.log' "
            f"> /dev/null 2>&1 &"
        )

        print(f"\n{'='*60}")
        print(f"  Launching {run['phase']}: rank={run['rank']} (full 4726 articles)")
        print(f"{'='*60}")

        launch(
            phase=run["phase"],
            pod_name=run["pod_name"],
            train_script="train_phase_c2.py",
            files_to_upload=FILES_TO_UPLOAD,
            output_subdir=run["output_subdir"],
            train_command=train_command,
        )
