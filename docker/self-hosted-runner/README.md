# Self‑Hosted GitHub Runner in a Rootless Docker Container

A **secure, GPU‑enabled** self‑hosted GitHub Actions runner that runs completely inside a rootless Docker container.

The container:

- **Never runs as root** – both the host Docker daemon and the runner process are unprivileged.
- Uses a **read‑only Docker socket mount** so the runner can start jobs but cannot tamper with the host Docker daemon.
- Provides **direct GPU access** via the NVIDIA runtime without granting extra privileges.
- Is **ephemeral** – containers are removed after each job, leaving no leftover state.

> **Why rootless?**  
> Rootless Docker isolates the Docker daemon from the host’s root user, dramatically reducing the impact of a container escape.
> Combined with a non‑root runner user, this gives you a very small attack surface while still allowing GPU‑accelerated builds.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Host Setup](#host-setup)
    - [Install rootless Docker](#1-install-rootless-docker)
    - [Configure NVIDIA runtime](#2-configure-the-nvidia-runtime-for-rootless-docker)
    - [Make the Docker socket accessible](#3-make-the-docker-socket-accessible)
3. [Container Overview](#container-overview)
    - [Base image](#base-runner-image)
    - [Entrypoint script](#entrypoint-script)
    - [Docker Compose file](#docker-compose-file)
4. [Running the Runner](#running-the-runner)
    - [Create a dotenv secret file](#1-create-a-dotenv-secret-file)
    - [Build & start the runner](#2-build--start-the-runner)
    - [View logs](#3-view-logs)
    - [Stop the runner](#4-stop-the-runner)
    - [Rebuild the runner after making changes](#5-rebuild-after-making-changes)
5. [Verify the Runner on GitHub](#verify-on-github)
6. [Troubleshooting & FAQs](#troubleshooting--faqs)

---

## Prerequisites

| Item                                                                                                         | Minimum version / notes                                   |
|--------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| Ubuntu 22.04 (or any recent Debian‑based distro)                                                             | Tested on 22.04 LTS                                       |
| Docker Engine **v20.10+** (rootless mode)                                                                    | `docker version` should show `Rootless: true` after setup |
| NVIDIA driver **470** or newer + **nvidia‑container‑toolkit**                                                | Required for GPU passthrough                              |
| `uidmap` package                                                                                             | Needed for rootless Docker                                |
| `curl`, `git`                                                                                                | Standard utilities                                        |
| A GitHub personal access token with **repo** and **admin:repo_hook** scopes (or a runner registration token) | See GitHub docs for generating a runner token             |

> **Tip:** If you already have a regular (rootful) Docker installation, you can keep it alongside rootless Docker.
> The rootless daemon runs under your user account and uses a separate socket (`/run/user/$UID/docker.sock`).

---

## Host Setup

### 1. Install Rootless Docker

```bash
# Install uidmap (required for rootless mode)
sudo apt-get -y install uidmap

# Download and run the installer
curl -fsSL https://get.docker.com/rootless | bash
```

If the installer fails, check the terminal output or consult Docker’s [troubleshooting guide](https://docs.docker.com/engine/security/rootless/troubleshoot/).

Enable and start the user‑level systemd service:

```bash
systemctl --user enable docker

# Enable service to run even if user is logged out
sudo loginctl enable-linger $(whoami)

systemctl --user start docker
```

Add the following lines to `~/.bashrc` (or your preferred shell rc file) so every new session knows where to find the rootless Docker socket:

```bash
echo -e 'export DOCKER_HOST=unix:///run/user/$(id -u)/docker.sock\nexport PATH=$HOME/bin:$PATH' >> ~/.bashrc
source ~/.bashrc   # reload the file in the current terminal
```

Verify that Docker is running in rootless mode:

```bash
docker info | grep -i rootless
# Expected output: Rootless: true
```

### 2. Configure the NVIDIA Runtime for Rootless Docker

```bash
# Install the NVIDIA container toolkit (if not already installed)
sudo apt-get install -y nvidia-container-toolkit

# Generate a user‑specific daemon.json for rootless Docker
nvidia-ctk runtime configure --runtime=docker --config=$HOME/.config/docker/daemon.json

# Restart the rootless daemon to pick up the new config
systemctl --user restart docker
```

Test GPU access inside a container:

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

You should see the familiar `nvidia-smi` table with your GPU details.

### 3. Make the Docker socket accessible

Rootless Docker uses a per‑user socket located at `/run/user/$UID/docker.sock`.  
The container needs read‑only access to this socket.
If the socket permissions are too restrictive, the runner cannot communicate with the daemon.

#### Quick fix (temporary)

```bash
chmod 660 /run/user/$(id -u)/docker.sock
```

> **Note:**
> Setting the socket to `600` grants read/write access to the owner only (`srw-------`).
> Setting the socket to `660` grants read/write access to the owner and group (`srw-rw----`).

### 4. Create the work directories

```shell
for i in 1 2 3 4; do
  sudo mkdir -p /var/lib/github-runners/runner${i}/_work
  sudo chown -R 1000:1000 /var/lib/github-runners/runner${i}
done
```

Ensure the host directories are created with the correct offset UID:
```shell
sudo install -d -o 100999 -g 100999 \
  /var/lib/github-runners/runner1/_work \
  /var/lib/github-runners/runner2/_work \
  /var/lib/github-runners/runner3/_work \
  /var/lib/github-runners/runner4/_work
```

---

## Container Overview

### Base Runner Image

The `Dockerfile` builds an image that contains:

- The official **GitHub Actions runner** binaries.
- Minimal runtime dependencies (Docker CLI, `bash`, `curl`, `jq`).
- The **NVIDIA runtime** libraries, needed for GPU jobs.

### Entrypoint Script

The `entrypoint.sh` script runs when the container starts and performs four checks:

1. **Docker availability**: ensures the host Docker socket is reachable.
2. **GPU availability**: runs `nvidia-smi` to confirm the device is visible.
3. **Runner registration**: reads the `.env` file and registers the runner if needed.
4. **Start the runner**: launches the GitHub runner in the foreground.

### Docker Compose File

The `docker-compose.yml` file defines a single service named `runner`:

- Mounts the **rootless Docker socket** (`/run/user/<uid>/docker.sock`) as read‑only.
- Passes the `.env` file for runtime configuration.
- Sets the `runtime: nvidia` key so the container gets GPU access.

---

## Running the Runners

### 1. Create a dotenv secret file

Create a file named `.env` in the project root (same directory as `docker‑compose.yml`).  
The file holds **all per‑runner tokens and names** required by the compose file.  
Only the values inside this file are read at container start – the file itself is **not** baked into the image.

```dotenv
# ---------- GitHub ----------
GITHUB_RUNNER_URL=https://github.com/<YOUR_USERNAME>/<YOUR_REPOSITORY>
# ---------- Runner Tokens ----------
GITHUB_RUNNER_TOKEN_1=TOKEN_FOR_RUNNER_1
GITHUB_RUNNER_TOKEN_2=TOKEN_FOR_RUNNER_2
GITHUB_RUNNER_TOKEN_3=TOKEN_FOR_RUNNER_3
GITHUB_RUNNER_TOKEN_4=TOKEN_FOR_RUNNER_4
# ---------- Common Runner Settings ----------
GITHUB_RUNNER_LABELS=self-hosted,linux,docker,x64,gpu
# ---------- Docker ----------
# Path to the rootless Docker socket
DOCKER_SOCK=/run/user/1000/docker.sock
# Set the GID of the rootless Docker socket
DOCKER_SOCKET_GID=100983
```

> **Security note:** 
> Treat this file like a secret.
> Do **not** commit it to version control.
> Add it to `.gitignore` if you keep the repo locally.

Make sure the user id on the host is `1000`, or change it to the user id you are using, et the value with `echo $(id -u)`.

Use `$(stat -c '%g' /run/user/$(id -u)/docker.sock)` to get the value for `DOCKER_SOCKET_GID`.

If you need more runners, simply add another block (`GITHUB_RUNNER_TOKEN_N` / `GITHUB_RUNNER_NAME`) and a matching service definition in `docker‑compose.yml`.


### 2. Build & start the runner

```bash
docker compose up -d --build
```

Docker builds the custom runner image (once) and starts **all** services defined in the compose file 
(`runner1`, `runner2`, ...) in detached mode.

### 3. View logs

To view the log of a specific runner, says `runner1`:
```bash
docker compose logs -f runner1
```

You should see a line similar to:

```
√ Connected to GitHub
```

### 4. Stop the runners

```bash
docker compose down
```

This stops **all** runner containers and removes the network.

### 5. Rebuild after making changes

If you modify the Dockerfile, entrypoint script, or any other source, rebuild the image and restart the services:

```bash
docker compose down && docker compose build --no-cache && docker compose up -d
```

---

## Verify the Runner on GitHub

1. Open your repository on GitHub.
2. Navigate to **Settings → Actions → Runners**.
3. The newly created runner appears in the list, usually with the status **Idle**.

You can now reference it in workflow files:

```
yaml
jobs:
  build:
    runs-on: self-hosted
    steps:
      - uses: actions/checkout@v3
      # your build steps here
```

---

## Troubleshooting & FAQs

| Symptom                                                                         | Likely cause                                                                                 | Fix                                                                                                                                                    |
|---------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| `docker: permission denied while trying to connect to the Docker daemon socket` | Docker socket not mounted read‑only or `$DOCKER_HOST` not set correctly inside the container. | Ensure `DOCKER_SOCK` in `.env` matches the host socket path (`/run/user/<uid>/docker.sock`). Verify `volumes:` in `docker-compose.yml` uses `ro` mode. |
| `nvidia-smi: command not found`                                                 | NVIDIA runtime not loaded in the container.                                                  | Make sure the host’s `nvidia-container-toolkit` is installed and the `runtime: nvidia` option is present in `docker-compose.yml`.                      |
| Runner repeatedly unregisters itself                                            | Missing or expired `GITHUB_RUNNER_TOKEN`.                                                    | Regenerate a fresh registration token from GitHub and update `.env`.                                                                                   |
| `Rootless: false` after `docker info`                                           | The rootless daemon is not running.                                                          | Run `systemctl --user start docker` and re‑source your shell to set `DOCKER_HOST`.                                                                     |
| GPU not visible inside the container                                            | Host driver version mismatch or missing `--gpus all` flag.                                   | Verify `docker run --gpus all nvidia/cuda:... nvidia-smi` works on the host first.                                                                     |
| “Failed to register runner” errors                                              | Incorrect URL or missing repository access rights.                                           | Double‑check `GITHUB_RUNNER_URL` (must include the full `https://github.com/...` path) and that the token has `admin:repo_hook` scope.                 |
| ERROR: Cannot connect to Docker daemon. Is the socket mounted correctly?        |                                                                                              | See [Make the Docker socket accessible](#3-make-the-docker-socket-accessible)                                                                          |

**Additional resources**

- Docker Rootless Docs: <https://docs.docker.com/engine/security/rootless/>
- NVIDIA Container Toolkit: <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>
- GitHub Self‑Hosted Runner docs: <https://docs.github.com/en/actions/hosting-your-own-runners/about-self-hosted-runners>
