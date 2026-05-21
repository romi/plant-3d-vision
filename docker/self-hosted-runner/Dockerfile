FROM ubuntu:24.04

# Disable TLS (simplifies DinD)
ENV DOCKER_TLS_CERTDIR=""

# Install system dependencies + nvidia-container-toolkit for inner Docker daemon
# but **do not** install docker.io via apt
RUN apt-get update && apt-get install -y \
            curl \
            git \
            jq \
            sudo \
            ca-certificates \
            gnupg \
        && install -m 0755 -d /etc/apt/keyrings \
        && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
            | gpg --dearmor -o /etc/apt/keyrings/nvidia-container-toolkit.gpg \
        && curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
            | sed 's#deb https://#deb [signed-by=/etc/apt/keyrings/nvidia-container-toolkit.gpg] https://#g' \
            > /etc/apt/sources.list.d/nvidia-container-toolkit.list \
        && apt-get update && apt-get install -y nvidia-container-toolkit \
        && nvidia-ctk runtime configure --runtime=docker \
        && rm -rf /var/lib/apt/lists/*

# Install Docker Engine using Docker’s official convenience script
RUN curl -fsSL https://get.docker.com -o get-docker.sh \
        && sh get-docker.sh \
        && rm get-docker.sh

# Create a dedicated user for the runner
RUN useradd -m -s /bin/bash github-runner \
    && echo "github-runner ALL=(ALL) NOPASSWD: /usr/bin/docker" >> /etc/sudoers.d/github-runner \
    && usermod -aG docker github-runner

# Set working directory and download runner as root, then fix ownership
WORKDIR /actions-runner

# Fetch the latest runner version dynamically to avoid going stale
RUN RUNNER_VERSION=$(curl -fsSL https://api.github.com/repos/actions/runner/releases/latest \
        | jq -r '.tag_name' | sed 's/^v//') \
    && curl -L -o actions-runner.tar.gz \
        "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz" \
    && tar xzf actions-runner.tar.gz \
    && rm actions-runner.tar.gz \
    && ./bin/installdependencies.sh \
    && mkdir -p /actions-runner/_work/_tool \
    && chown -R github-runner:github-runner /actions-runner

# Copy and set permissions BEFORE dropping to non-root user
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Switch to runner user
USER github-runner

WORKDIR /actions-runner
ENTRYPOINT ["/entrypoint.sh"]