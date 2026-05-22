#!/bin/bash
set -e

# Verify Docker socket is accessible (DooD approach)
echo "Verifying Docker access..."
if ! docker info >/dev/null 2>&1; then
    echo "ERROR: Cannot connect to Docker daemon. Is the socket mounted correctly?"
    exit 1
fi

echo "Docker is accessible!"
docker --version

# Verify GPU access if available
if docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
    echo "GPU access verified!"
else
    echo "WARNING: GPU access check failed. Ensure nvidia-container-toolkit is configured."
fi

# Configure the GitHub Actions runner
echo "Configuring GitHub Actions runner..."
./config.sh \
    --unattended \
    --url "${GITHUB_RUNNER_URL}" \
    --token "${GITHUB_RUNNER_TOKEN}" \
    --name "${GITHUB_RUNNER_NAME:-romi-github-runner}" \
    --labels "${GITHUB_RUNNER_LABELS:-self-hosted,linux,docker,x64,gpu}" \
    --replace

# Start the runner
echo "Starting GitHub Actions runner..."
./run.sh