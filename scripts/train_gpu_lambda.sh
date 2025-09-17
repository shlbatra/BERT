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
mkdir -p my-gpu-project && cd my-gpu-project
sudo snap install docker 
sudo apt install nvidia-utils-570-server -y
sudo apt install nvidia-driver-570-server -y

# Download and setup NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/amd64 /" | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install nvidia-container-toolkit -y
sudo nvidia-ctk runtime configure --runtime=docker
'

echo "Step 2: Restarting machine..."
run_remote 'sudo reboot' || true  # Allow this to "fail" since connection drops

# Wait for machine to come back online
wait_for_reboot

echo "Step 3: Copying GCP credentials..."
if [ -f "gcp-key.json" ]; then
    copy_to_remote "gcp-key.json" "~/my-gpu-project/"
else
    echo "Warning: gcp-key.json not found in current directory"
fi



echo "Step 4: Testing GPU and pulling Docker image..."
run_remote '
nvidia-smi
cd my-gpu-project/
sudo docker system prune -af
sudo docker pull shlbatra123/gpu_docker_image:latest
'

echo "Step 5: Setting up directories and running training..."
run_remote '
cd my-gpu-project/
mkdir -p checkpoints logs
sudo chown -R 1001:1001 checkpoints logs
sudo docker run --runtime=nvidia \
  -v $(pwd)/gcp-key.json:/app/gcp-key.json \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/logs:/app/logs \
  --rm shlbatra123/gpu_docker_image:latest \
  bash -c "torchrun --nproc_per_node='"$NUM_GPUS"' train_bert.py"
'

echo "Training completed successfully!"
echo "To download results: scp -r $SSH_HOST:~/my-gpu-project/checkpoints ./"