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
┌─────────────────────────────────────────────────────────────┐
│                    Host System                              │
│  - N/A (runner is isolated)                                 │
└─────────────────────────────────────────────────────────────┘
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
GITHUB_RUNNER_NAME=runner-$(hostname)
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
docker-compose logs runner
```

You should see: `√ Connected to GitHub: Listening for Jobs`

## Advanced Configuration Options

### Enable Ephemeral Runners

To ensure the runner container is destroyed after each job:

```yaml
services:
  runner:
    image: your-registry.com/self-hosted-runner:latest
    # ... other configuration
    environment:
      - GITHUB_RUNNER_EPHEMERAL=true
```

### Use Rootless Docker (More Secure)

For even better security, run the Docker daemon as a non-root user:

```yaml
services:
  runner:
    image: your-registry.com/self-hosted-runner:latest
    privileged: false
    user: "1000:1000"  # Run as a specific user
    security_opt:
      - no-new-privileges:true
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro  # Read-only mount
      - runner-cache:/actions-runner/_work
    environment:
      - DOCKER_DAEMON_CONFIG=/etc/docker/daemon.json
```

Create a `daemon.json` file:

```json
{
  "userns-remap": "default",
  "log-level": "warn",
  "live-restore": true
}
```

### Configure Resource Limits

```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G
    reservations:
      cpus: '0.5'
      memory: 1G
```

### Set Up Health Checks

```yaml
services:
  runner:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:32767/_diag"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

### Network Isolation

If you're using Kubernetes, create a NetworkPolicy to restrict outbound traffic:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: runner-egress
  namespace: github-actions
spec:
  podSelector:
    matchLabels:
      app: github-runner
  egress:
    - to:
        - ipBlock:
            cidr: 140.82.112.0/20  # GitHub API IP range
      ports:
        - protocol: TCP
          port: 443
    - to:
        - ipBlock:
            cidr: 192.168.0.0/16  # Your internal registry
      ports:
        - protocol: TCP
          port: 5000
```

## Security Best Practices

### 1. Use Read-Only Docker Socket

```yaml
volumes:
  - /var/run/docker.sock:/var/run/docker.sock:ro
```

This prevents the runner from modifying the Docker daemon.

### 2. Restrict Capabilities

```yaml
cap