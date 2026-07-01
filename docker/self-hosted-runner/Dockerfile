FROM ubuntu:24.04

# Install minimal dependencies - NO Docker daemon needed
RUN apt-get update && apt-get install -y \
        curl \
        git \
        jq \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Docker CLI & buildx only (not the daemon)
RUN curl -fsSL https://download.docker.com/linux/static/stable/x86_64/docker-29.6.1.tgz \
    | tar xz --strip-components=1 -C /usr/local/bin docker/docker \
    && chmod +x /usr/local/bin/docker \
    && mkdir -p /usr/local/lib/docker/cli-plugins \
    && curl -fsSL https://github.com/docker/buildx/releases/download/v0.35.0/buildx-v0.35.0.linux-amd64 -o /usr/local/lib/docker/cli-plugins/docker-buildx \
    && chmod +x /usr/local/lib/docker/cli-plugins/docker-buildx

# Set working directory and download runner
WORKDIR /actions-runner

# Fetch the latest runner version dynamically
RUN curl -L -o actions-runner.tar.gz "https://github.com/actions/runner/releases/tag/v2.335.1" \
    && tar xzf actions-runner.tar.gz \
    && rm actions-runner.tar.gz \
    && ./bin/installdependencies.sh \
    && chown -R 1000:1000 /actions-runner

# Copy entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Switch to non-root user
USER ubuntu

WORKDIR /actions-runner
ENTRYPOINT ["/entrypoint.sh"]