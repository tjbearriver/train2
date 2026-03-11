#!/bin/bash
# submit_vertex.sh
# Before running this script, ensure you have:
# 1. gcloud auth login
# 2. gcloud auth configure-docker <region>-docker.pkg.dev
# 3. Created an Artifact Registry repo named "cranberry-train"

set -e

PROJECT_ID="project-158a3861-f146-4316-a63"
REGION="us-central1"
REPO_NAME="cranberry-train"
IMAGE_NAME="qwen-vertex"
IMAGE_TAG="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:latest"

# Ensure we are running from the workspace root
cd "$(dirname "$0")/.."

echo "Building Docker Image..."
docker build -t ${IMAGE_NAME} -f vertex/Dockerfile.vertex .

echo "Tagging and Pushing Image to Artifact Registry..."
docker tag ${IMAGE_NAME} ${IMAGE_TAG}
docker push ${IMAGE_TAG}

echo "Submitting Vertex AI Custom Job..."
# Replace with your actual GCS bucket path (DO NOT INCLUDE gs:// prefix)
# E.g., GCS_OUTPUT_BUCKET="my-bucket-name/cranberry-train-output"
GCS_OUTPUT_BUCKET="my-cranberry-bucket/cranberry-train-output"

gcloud ai custom-jobs create \
    --region=${REGION} \
    --display-name="train-qwen35-27b-1000art" \
    --worker-pool-spec=machine-type=g2-standard-12,accelerator-type=NVIDIA_L4,accelerator-count=1,replica-count=1,container-image-uri=${IMAGE_TAG} \
    --args="${GCS_OUTPUT_BUCKET}" \
    --project=${PROJECT_ID}

echo "Job submitted! Check Vertex AI console for logs."