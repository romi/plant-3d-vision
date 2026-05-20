FROM ubuntu:24.04

# --- Build arguments (with defaults for labels and name) -----------------
ARG GITHUB_RUNNER_URL
ARG GITHUB_RUNNER_TOKEN
ARG GITHUB_RUNNER_LABELS=self-hosted,linux,docker,x64
ARG GITHUB_RUNNER_NAME=romi-github-runner

# Export the arguments as environment variables for the container runtime
ENV GITHUB_RUNNER_URL=${GITHUB_RUNNER_URL}
ENV GITHUB_RUNNER_TOKEN=${GITHUB_RUNNER_TOKEN}
ENV GITHUB_RUNNER_LABELS=${GITHUB_RUNNER_LABELS}
ENV GITHUB_RUNNER_NAME=${GITHUB_RUNNER_NAME}
# -------------------------------------------------------------------------

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    docker.io \
    jq \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Create a dedicated user for the runner
RUN useradd -m -s /bin/bash github-runner \
    && echo "github-runner ALL=(ALL) NOPASSWD: /usr/bin/docker" >> /etc/sudoers.d/github-runner

# Set working directory
WORKDIR /actions-runner

# Download and extract GitHub Actions Runner
RUN curl -O -L https://github.com/actions/runner/releases/download/v2.334.0/actions-runner-linux-x64-2.334.0.tar.gz -o actions-runner.tar.gz \
    && tar xzf actions-runner.tar.gz \
    && rm actions-runner.tar.gz

# Set ownership
RUN chown -R github-runner:github-runner /actions-runner

# Switch to runner user
USER github-runner

# Copy docker-compose.yml or runner configuration
COPY docker-compose.yml /actions-runner/
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /actions-runner
ENTRYPOINT ["/entrypoint.sh"]