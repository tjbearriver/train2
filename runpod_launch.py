"""Shared RunPod pod provisioning and training launch utilities.

Used by launch_phase_a.py, launch_phase_b.py, etc.
"""

import json
import os
import subprocess
import sys
import time

import runpod

# Defaults shared across all phases
DEFAULT_GPU_TYPE = "NVIDIA GeForce RTX 5090"
DEFAULT_CLOUD_TYPE = "SECURE"
DEFAULT_CONTAINER_DISK_GB = 250
DEFAULT_IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
DEFAULT_REMOTE_WORK_DIR = "/workspace/train7"
DEFAULT_SETUP_COMMANDS = """
set -e
cd /workspace/train7
pip install --break-system-packages --upgrade pip
pip install --break-system-packages unsloth
"""

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_api_key():
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        print("ERROR: Set RUNPOD_API_KEY environment variable")
        print("  export RUNPOD_API_KEY=your_key_here")
        sys.exit(1)
    runpod.api_key = key
    return key


def find_gpu_type(name_fragment="5090"):
    """Find a GPU type ID by name fragment."""
    gpus = runpod.get_gpus()
    for gpu in gpus:
        if name_fragment in gpu.get("displayName", "") or name_fragment in gpu.get("id", ""):
            print(f"Found GPU: {gpu['id']} — {gpu.get('displayName', 'N/A')}")
            return gpu["id"]
    print(f"GPU matching '{name_fragment}' not found. Available GPUs:")
    for gpu in gpus:
        print(f"  {gpu['id']} — {gpu.get('displayName', 'N/A')}")
    sys.exit(1)


def create_pod(
    pod_name,
    gpu_type_id,
    cloud_type=DEFAULT_CLOUD_TYPE,
    container_disk_gb=DEFAULT_CONTAINER_DISK_GB,
    image=DEFAULT_IMAGE,
):
    """Create a RunPod pod and return its ID."""
    print(f"\nCreating pod: {pod_name}")
    print(f"  GPU: {gpu_type_id}")
    print(f"  Cloud: {cloud_type}")
    print(f"  Disk: {container_disk_gb}GB")

    pod = runpod.create_pod(
        name=pod_name,
        gpu_type_id=gpu_type_id,
        cloud_type=cloud_type,
        container_disk_in_gb=container_disk_gb,
        image_name=image,
        gpu_count=1,
        start_ssh=True,
        support_public_ip=True,
        ports="22/tcp",
    )
    pod_id = pod["id"]
    print(f"  Pod ID: {pod_id}")
    return pod_id


def wait_for_pod(pod_id, timeout=900):
    """Wait for pod to be RUNNING with SSH ready. Returns (host, port)."""
    print(f"\nWaiting for pod {pod_id} to be ready...")
    start = time.time()
    while time.time() - start < timeout:
        pod = runpod.get_pod(pod_id)
        status = pod.get("desiredStatus", "UNKNOWN")
        runtime = pod.get("runtime")

        if status == "RUNNING" and runtime:
            for p in runtime.get("ports", []):
                if p.get("privatePort") == 22:
                    ssh_port = p.get("publicPort")
                    ssh_host = p.get("ip")
                    if ssh_port and ssh_host:
                        print(f"  Pod RUNNING — SSH: ssh root@{ssh_host} -p {ssh_port}")
                        return ssh_host, ssh_port

        elapsed = int(time.time() - start)
        print(f"  Status: {status}, runtime={'yes' if runtime else 'no'} (elapsed: {elapsed}s)")
        time.sleep(10)

    print("ERROR: Pod did not become ready within timeout")
    sys.exit(1)


def run_ssh(host, port, cmd, check=True):
    """Run a command on the pod via SSH."""
    ssh_cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
        "-p", str(port), f"root@{host}", cmd,
    ]
    result = subprocess.run(ssh_cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"SSH command failed: {cmd}")
        print(f"  stderr: {result.stderr}")
        return None
    return result.stdout.strip()


def upload_files(host, port, files, remote_dirs=None, remote_work_dir=DEFAULT_REMOTE_WORK_DIR):
    """Upload files to the pod via scp.

    Args:
        files: List of paths relative to PROJECT_DIR.
        remote_dirs: Extra remote directories to create (relative to remote_work_dir).
    """
    print(f"\nUploading files to pod...")

    # Build mkdir command for remote dirs
    dirs_to_create = {f"{remote_work_dir}/data"}
    if remote_dirs:
        for d in remote_dirs:
            dirs_to_create.add(f"{remote_work_dir}/{d}")
    run_ssh(host, port, f"mkdir -p {' '.join(sorted(dirs_to_create))}")

    for f in files:
        local = os.path.join(PROJECT_DIR, f)
        remote = f"{remote_work_dir}/{f}"
        if not os.path.exists(local):
            print(f"  SKIP (not found): {f}")
            continue
        size_mb = os.path.getsize(local) / (1024 * 1024)
        print(f"  Uploading {f} ({size_mb:.1f} MB)...")

        scp_cmd = [
            "scp", "-o", "StrictHostKeyChecking=no",
            "-P", str(port), local, f"root@{host}:{remote}",
        ]
        result = subprocess.run(scp_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"    FAILED: {result.stderr}")
            sys.exit(1)
        print(f"    Done")


def setup_environment(host, port, commands=DEFAULT_SETUP_COMMANDS):
    """Install dependencies on the pod."""
    print(f"\nInstalling dependencies on pod...")
    output = run_ssh(host, port, commands, check=False)
    if output:
        for line in output.split("\n")[-10:]:
            print(f"  {line}")
    print("  Dependencies installed")


def start_training(host, port, train_command, phase, process_name, log_path,
                   remote_work_dir=DEFAULT_REMOTE_WORK_DIR):
    """Start a training script in the background and print monitor commands."""
    print(f"\nStarting Phase {phase} training...")
    run_ssh(host, port, train_command, check=False)
    time.sleep(2)

    # Verify it's running
    output = run_ssh(host, port, f"pgrep -f {process_name}", check=False)
    if output:
        print(f"  Training started (PID: {output})")
    else:
        print("  WARNING: Training process may not have started. Check logs.")

    print(f"\n{'='*60}")
    print(f"  Phase {phase} training launched!")
    print(f"  Monitor logs:")
    print(f"    ssh root@{host} -p {port} 'tail -f {log_path}'")
    print(f"  Check GPU:")
    print(f"    ssh root@{host} -p {port} 'nvidia-smi'")
    print(f"{'='*60}\n")


def save_pod_info(pod_id, host, port, gpu_type_id, phase, output_dir):
    """Save pod connection info to a JSON file."""
    pod_info = {
        "pod_id": pod_id,
        "host": host,
        "port": port,
        "gpu_type": gpu_type_id,
        "phase": phase,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    info_path = os.path.join(PROJECT_DIR, output_dir, "pod_info.json")
    os.makedirs(os.path.dirname(info_path), exist_ok=True)
    with open(info_path, "w") as f:
        json.dump(pod_info, f, indent=2)
    print(f"Pod info saved to {info_path}")
    return pod_info


def launch(
    phase,
    pod_name,
    train_script,
    files_to_upload,
    output_subdir,
    train_command=None,
    setup_commands=DEFAULT_SETUP_COMMANDS,
    remote_work_dir=DEFAULT_REMOTE_WORK_DIR,
):
    """Full launch flow: provision pod, upload, install, train.

    Args:
        phase: Phase label (e.g. "A", "B_lr1").
        pod_name: RunPod pod name.
        train_script: Name of the training script (for pgrep).
        files_to_upload: List of local files to upload (relative to project dir).
        output_subdir: Output subdirectory (e.g. "output/phase_a").
        train_command: Shell command to run training. Defaults to a standard nohup pattern.
        setup_commands: Shell commands for dependency installation.
        remote_work_dir: Remote working directory.
    """
    log_path = f"{remote_work_dir}/{output_subdir}/train.log"

    if train_command is None:
        train_command = (
            f"cd {remote_work_dir} && "
            f"mkdir -p {output_subdir} && "
            f"nohup python -u {train_script} "
            f"> >(tee {output_subdir}/train.log) 2>&1 &"
        )

    get_api_key()
    gpu_type_id = find_gpu_type()
    pod_id = create_pod(pod_name, gpu_type_id)
    host, port = wait_for_pod(pod_id)
    upload_files(host, port, files_to_upload, remote_dirs=[output_subdir])
    setup_environment(host, port, setup_commands)
    start_training(host, port, train_command, phase, train_script, log_path, remote_work_dir)
    save_pod_info(pod_id, host, port, gpu_type_id, phase, output_subdir)

    print(f"  Download results when done:")
    print(f"    scp -P {port} -r root@{host}:{remote_work_dir}/{output_subdir}/ {output_subdir}/")
