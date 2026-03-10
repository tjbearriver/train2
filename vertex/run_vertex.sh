#!/bin/bash
set -e

# Support passing the bucket as the first script argument
if [ -n "$1" ]; then
    OUTPUT_BUCKET=$1
fi

echo "Starting training job..."
python3 train_model_compare.py --model unsloth/Qwen3.5-27B --tag qwen35_27b_vertex

if [ -n "$OUTPUT_BUCKET" ]; then
    # In Vertex AI, Cloud Storage buckets are automatically mounted at /gcs/
    # If OUTPUT_BUCKET is set to "my-bucket/cranberry-train", we copy to /gcs/my-bucket/cranberry-train/
    echo "Saving artifacts to GCS auto-mount at /gcs/$OUTPUT_BUCKET/ ..."
    
    # Small pause to guarantee file flushes are complete
    sleep 3
    
    mkdir -p /gcs/$OUTPUT_BUCKET/output
    mkdir -p /gcs/$OUTPUT_BUCKET/data
    mkdir -p /gcs/$OUTPUT_BUCKET/logs
    
    # Copy LoRA Adapter and stats
    if [ -d "/workspace/output/1000art_qwen35_27b_vertex" ]; then
        cp -r /workspace/output/1000art_qwen35_27b_vertex /gcs/$OUTPUT_BUCKET/output/
    fi
    
    # Copy Evaluation JSONs
    cp /workspace/data/*1000art_qwen35_27b_vertex* /gcs/$OUTPUT_BUCKET/data/ || true
    
    # Copy any training logs
    cp /workspace/*.log /gcs/$OUTPUT_BUCKET/logs/ || true

    echo "Artifacts successfully saved to Cloud Storage!"
else
    echo "OUTPUT_BUCKET environment variable not set. Skipping artifact upload."
fi
