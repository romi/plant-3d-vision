#!/bin/bash
set -e

# Verify Docker socket is accessible (DooD approach)
echo "Verifying Docker access..."

# Try to access docker - if it fails, provide helpful error message
if ! docker info >/dev/null 2>&1; then
    echo "ERROR: Cannot connect to Docker daemon."
    echo "Checking socket permissions..."
    ls -la /var/run/docker.sock 2>/dev/null || echo "Socket not found at /var/run/docker.sock"

    echo ""
    echo "For rootless Docker, ensure:"
    echo "1. Socket is mounted: -v /run/user/1000/docker.sock:/var/run/docker.sock"
    echo "2. Socket is accessible on host: chmod 666 /run/user/1000/docker.sock"
    echo "3. Or run container without USER directive (as root inside container)"
    exit 1
fi

echo "Docker is accessible!"
docker --version
docker buildx --version

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