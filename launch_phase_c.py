#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["runpod"]
# ///
"""Launch Phase C LoRA rank sweep on three RunPod 5090 pods (parallel).

Provisions three pods — one for r=16, one for r=32, one for r=128 — each with
separate output directories so they can run fully in parallel.

Usage:
  export RUNPOD_API_KEY=rpa_...
  uv run launch_phase_c.py
"""

from runpod_launch import launch

PHASE_C_RUNS = [
    {
        "phase": "C_r16",
        "pod_name": "phase-c-r16-qwen35-9b",
        "output_subdir": "output/phase_c_r16",
        "rank": 16,
        "tag": "phase_c_r16",
    },
    {
        "phase": "C_r32",
        "pod_name": "phase-c-r32-qwen35-9b",
        "output_subdir": "output/phase_c_r32",
        "rank": 32,
        "tag": "phase_c_r32",
    },
    {
        "phase": "C_r128",
        "pod_name": "phase-c-r128-qwen35-9b",
        "output_subdir": "output/phase_c_r128",
        "rank": 128,
        "tag": "phase_c_r128",
    },
]

FILES_TO_UPLOAD = [
    "train_phase_c.py",
    "data/train.json",
    "data/eval.json",
]


if __name__ == "__main__":
    for run in PHASE_C_RUNS:
        train_command = (
            f"cd /workspace/train7 && "
            f"mkdir -p {run['output_subdir']} && "
            f"nohup bash -c 'python -u train_phase_c.py --rank {run['rank']} --tag {run['tag']} 2>&1 | tee {run['output_subdir']}/train.log' "
            f"> /dev/null 2>&1 &"
        )

        print(f"\n{'='*60}")
        print(f"  Launching {run['phase']}: rank={run['rank']}")
        print(f"{'='*60}")

        launch(
            phase=run["phase"],
            pod_name=run["pod_name"],
            train_script="train_phase_c.py",
            files_to_upload=FILES_TO_UPLOAD,
            output_subdir=run["output_subdir"],
            train_command=train_command,
        )
