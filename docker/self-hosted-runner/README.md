# Self-Hosted GitHub Runner in Docker Container

Running the runner in a Rootless Docker container provides several security benefits:

- **No privileged containers** - container escape doesn't give root
- **Rootless Docker on host** - even host Docker runs as regular user
- **Non-root runner user** - runner process has minimal privileges
- **Read-only socket mount** - runner can't manipulate Docker daemon
- **GPU passthroug**h - direct device access without security compromise
- **Ephemeral runners** - auto-cleanup after each job
- **Minimal attack surface** - only Docker CLI installed, no daemon

This setup gives you secure GPU access for your CUDA builds while maintaining strong isolation!

## Host Setup Guide

### 1. Set up Rootless Docker on Host

First, set up rootless Docker on your host machine:
```bash
# Install rootless Docker (if not already done)
curl -fsSL https://get.docker.com/rootless | sh

# Enable systemd service
systemctl --user enable docker
systemctl --user start docker

# Set environment variables (add to ~/.bashrc)
export DOCKER_HOST=unix:///run/user/$(id -u)/docker.sock
export PATH=/home/$USER/bin:$PATH

# Verify rootless Docker is working
docker ps
```

For more details, see the [official Docker documentation](https://docs.docker.com/engine/security/rootless/).

### 2. Configure NVIDIA Runtime for Rootless Docker

```bash
# Install nvidia-container-toolkit on host if not already installed
sudo apt-get install -y nvidia-container-toolkit

# Configure for rootless Docker
nvidia-ctk runtime configure --runtime=docker --config=$HOME/.config/docker/daemon.json --rootless

# Restart rootless Docker
systemctl --user restart docker

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```


## Container Setup Guide

### Base Runner Image

The `Dockerfile` will be used to create a Docker image that contains the GitHub Actions runner and any necessary dependencies.

### Entrypoint

The `entrypoint.sh` file will:
1. verify docker is available
2. verify the GPU is available
3. configure the github runner using the `.env` file, if needed
4. start the github runner

### Docker Compose Configuration

The `docker-compose.yml` file will run the GitHub Actions runner.

### Runtime configuration

Use an `.env` file to pass the configuration at runtime only:

```dotenv
# GitHub Runner Configuration
GITHUB_RUNNER_URL=https://github.com/YOUR_USERNAME/YOUR_REPOSITORY
GITHUB_RUNNER_TOKEN=your-registration-token-here

# Runner Configuration
GITHUB_RUNNER_LABELS=self-hosted,linux,docker,x64,gpu
GITHUB_RUNNER_NAME=romi-github-runner

# Docker socket path (rootless Docker)
DOCKER_SOCK=/run/user/1000/docker.sock
```

## Usage

### Build and start the runner
``` bash
docker compose up -d --build
```

### View logs
``` bash
docker compose logs -f runner
```

You should see: `√ Connected to GitHub`

### Stop the runner
``` bash
docker compose down
```

### Rebuild after changes
``` bash
docker compose down && docker compose build --no-cache && docker compose up -d
```


## Verify the Runner on GitHub

1. Navigate to your GitHub repository settings
2. Go to **Settings** → **Actions** → **Runners**
3. Your runner should appear as **Idle** in the list
