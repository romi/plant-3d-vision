#!/usr/bin/env bash
set -euo pipefail

# --------- start Docker daemon for DinD ----------
# Run dockerd in the background (the container runs as the non‑root github‑runner user,
# which already has password‑less sudo rights for /usr/bin/docker as set in the Dockerfile)
sudo dockerd > /dev/null 2>&1 &

# Wait until the Docker socket is responsive
until sudo docker info > /dev/null 2>&1; do
  sleep 1
done
# ------------------------------------------------

# Working directory where the runner was unpacked
cd /actions-runner

# If the runner hasn't been configured yet, run the config script.
# The environment variables are supplied by docker‑compose.
if [ ! -f .runner ]; then
  echo "Configuring GitHub Actions runner..."
  ./config.sh \
    --unattended \
    --url "${GITHUB_RUNNER_URL}" \
    --token "${GITHUB_RUNNER_TOKEN}" \
    --labels "${GITHUB_RUNNER_LABELS}" \
    --name "${GITHUB_RUNNER_NAME}" \
    --replace
  touch .runner   # marker to avoid re‑configuring on every start
else
  echo "GitHub Actions runner already configured..."
fi

# Finally start the runner (foreground)
exec ./run.sh "$@"