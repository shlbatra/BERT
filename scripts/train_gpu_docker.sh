#!/bin/bash

# GPU Training Deployment Script
# Usage: ./scripts/train_gpu_lambda.sh [SSH_HOST] [NUM_GPUS]
# Example: ./scripts/train_gpu_lambda.sh paperspace@184.105.3.177 1

set -e

# Check if SSH host is provided
if [ -z "$1" ]; then
    echo "Usage: $0 [SSH_HOST] [NUM_GPUS]"
    echo "Example: $0 paperspace@184.105.3.177 1"
    exit 1
fi

SSH_HOST="$1"
NUM_GPUS="${2:-1}"  # Default to 1 GPU if not specified
echo "Deploying to: $SSH_HOST with $NUM_GPUS GPUs"

# Function to run commands on remote host
run_remote() {
    ssh -o StrictHostKeyChecking=no "$SSH_HOST" "$@"
}

# Function to copy files to remote host
copy_to_remote() {
    scp -o StrictHostKeyChecking=no "$1" "$SSH_HOST:$2"
}

# Function to wait for host to come back online after reboot
wait_for_reboot() {
    echo "Waiting for host to reboot and come back online..."
    sleep 30  # Initial wait for reboot to start
    
    while ! ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "$SSH_HOST" "echo 'Host is up'" >/dev/null 2>&1; do
        echo "Still waiting for host..."
        sleep 10
    done
    
    echo "Host is back online!"
    sleep 5  # Extra wait for services to stabilize
}

echo "Step 1: Setting up remote environment..."
run_remote '
sudo apt update
'

echo "Step 2: Copying GCP credentials..."
if [ -f "gcp-key.json" ]; then
    run_remote "mkdir -p ~/my-gpu-project"
    copy_to_remote "gcp-key.json" "~/my-gpu-project/"
else
    echo "Error: gcp-key.json not found in current directory"
    echo "Please ensure gcp-key.json exists in the same directory as this script"
    exit 1
fi

echo "Step 3: Testing GPU and pulling Docker image..."
run_remote '
nvidia-smi
cd my-gpu-project/
sudo docker system prune -af
sudo docker pull shlbatra123/gpu_docker_bert:latest
'

echo "Step 4: Setting up directories and running training..."
run_remote '
cd my-gpu-project/
mkdir -p checkpoints logs
sudo chown -R 1001:1001 checkpoints logs
sudo docker run --runtime=nvidia \
  -v $(pwd)/gcp-key.json:/app/gcp-key.json \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/logs:/app/logs \
  --rm shlbatra123/gpu_docker_bert:latest \
  bash -c "torchrun --nproc_per_node='"$NUM_GPUS"' train_bert.py"
'

echo "Training completed successfully!"
echo "To download results: scp -r $SSH_HOST:~/my-gpu-project/checkpoints ./"