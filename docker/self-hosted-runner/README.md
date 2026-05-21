# Self-Hosted GitHub Runner in Docker Container

Running the runner in a Docker container provides several security benefits:

- **Natural isolation**: Each build task runs in its own container, which is destroyed after use
- **Ephemeral cleanup**: The runner container is destroyed after each job completes, preventing state persistence between workflows
- **Reduced attack surface**: If a malicious script compromises the runner container, you simply delete and recreate it
- **Least privilege by default**: Containers don't have direct access to the host system by default
- **Easier to rollback**: If the runner container is corrupted, you just redeploy it from your image registry

## Architecture Overview

```
┌────────────────────────────────────────────────────────────┐
│                   GitHub Repository                        │
│                   (GitHub Actions Workflows)               │
└──────────────────────────┬─────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────┐
│                  Self-Hosted Runner Container              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  GitHub Actions Runner Process                      │   │
│  │  - Connects to GitHub API                           │   │
│  │  - Accepts jobs                                     │   │
│  │  - Runs workflow steps                              │   │
│  └─────────────────────────────────────────────────────┘   │
│                     │                                      │
│                     │ (mounts Docker socket)               │
│                     ▼                                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Docker Daemon (if using Docker)                    │   │
│  │  - Builds container images                          │   │
│  │  - Runs containers                                  │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────┐
│                    Host System                             │
│  - N/A (runner is isolated)                                │
└────────────────────────────────────────────────────────────┘
```

## Setup Guide

### Base Runner Image

The `Dockerfile` will be used to create a Docker image that contains the GitHub Actions runner and any necessary dependencies.

### Docker Compose Configuration

The `docker-compose.yml` file will run the GitHub Actions runner.

## Build the Runner

Then simply run:
``` bash
docker compose up -d --build
```

## Run the Runner

Use an `.env` file to pass the configuration at runtime only:

```dotenv
# GitHub Runner Configuration
GITHUB_RUNNER_URL=https://github.com/YOUR_USERNAME/YOUR_REPOSITORY
GITHUB_RUNNER_TOKEN=your-registration-token-here
# Runner Configuration
GITHUB_RUNNER_LABELS=self-hosted,linux,docker,x64
GITHUB_RUNNER_NAME=romi-github-runner
```

From the directory containing your `Dockerfile` and `docker-compose.yml`:

```bash
# Run the runner
docker-compose up -d
```

## Verify the Runner

1. Navigate to your GitHub repository settings
2. Go to **Settings** → **Actions** → **Runners**
3. Your runner should appear as **Idle** in the list
4. Check the Docker container logs:

```bash
docker compose logs runner
```

You should see: `√ Connected to GitHub`