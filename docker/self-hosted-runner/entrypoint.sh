#!/bin/bash
set -e

# Before starting dockerd, ensure nvidia runtime is configured
sudo nvidia-ctk runtime configure --runtime=docker --config=/etc/docker/daemon.json
# Start Docker daemon in the background (requires privileged mode)
echo "Starting Docker daemon..."
sudo dockerd --host=unix:///var/run/docker.sock &

# Wait for Docker daemon to be ready
echo "Waiting for Docker daemon to be ready..."
timeout=30
elapsed=0
until sudo docker info >/dev/null 2>&1; do
    if [ $elapsed -ge $timeout ]; then
        echo "ERROR: Docker daemon failed to start within ${timeout}s"
        exit 1
    fi
    sleep 1
    elapsed=$((elapsed + 1))
done
echo "Docker daemon is ready!"

# Verify Docker is working
sudo docker --version

# Configure the GitHub Actions runner
echo "Configuring GitHub Actions runner..."
./config.sh \
    --unattended \
    --url "${GITHUB_RUNNER_URL}" \
    --token "${GITHUB_RUNNER_TOKEN}" \
    --name "${GITHUB_RUNNER_NAME:-romi-github-runner}" \
    --labels "${GITHUB_RUNNER_LABELS:-self-hosted,linux,docker,x64}" \
    --replace

# Start the runner
echo "Starting GitHub Actions runner..."
./run.sh