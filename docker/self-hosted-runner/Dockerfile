FROM ubuntu:24.04

# Install minimal dependencies - NO Docker daemon needed
RUN apt-get update && apt-get install -y \
        curl \
        git \
        jq \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Docker CLI only (not the daemon)
RUN curl -fsSL https://download.docker.com/linux/static/stable/x86_64/docker-29.5.2.tgz \
    | tar xz --strip-components=1 -C /usr/local/bin docker/docker \
    && chmod +x /usr/local/bin/docker

# Set working directory and download runner
WORKDIR /actions-runner

# Fetch the latest runner version dynamically
RUN RUNNER_VERSION=$(curl -fsSL https://api.github.com/repos/actions/runner/releases/latest \
        | jq -r '.tag_name' | sed 's/^v//') \
    && curl -L -o actions-runner.tar.gz \
        "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz" \
    && tar xzf actions-runner.tar.gz \
    && rm actions-runner.tar.gz \
    && ./bin/installdependencies.sh \
    && mkdir -p /actions-runner/_work/_tool \
    && chown -R ubuntu:ubuntu /actions-runner

# Copy entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Switch to non-root user
USER ubuntu

WORKDIR /actions-runner
ENTRYPOINT ["/entrypoint.sh"]