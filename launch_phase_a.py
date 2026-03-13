#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["runpod"]
# ///
"""Launch Phase A training on a RunPod 5090 pod.

Provisions the pod, uploads code+data, installs deps, starts training.

Usage:
  export RUNPOD_API_KEY=rpa_...
  uv run launch_phase_a.py
"""

from runpod_launch import launch

if __name__ == "__main__":
    launch(
        phase="A",
        pod_name="phase-a-qwen35-9b",
        train_script="train_phase_a.py",
        files_to_upload=[
            "train_phase_a.py",
            "data/train.json",
            "data/eval.json",
        ],
        output_subdir="output/phase_a",
    )
