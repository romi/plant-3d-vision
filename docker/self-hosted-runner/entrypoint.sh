#!/usr/bin/env bash
set -euo pipefail

# Working directory where the runner was unpacked
cd /actions-runner

# If the runner hasn't been configured yet, run the config script.
# The environment variables are supplied by docker‑compose.
if [ ! -f .runner ]; then
  echo "Configuring GitHub Actions runner..."
  ./config.sh \
    --url "${GITHUB_RUNNER_URL}" \
    --token "${GITHUB_RUNNER_TOKEN}" \
    --labels "${GITHUB_RUNNER_LABELS}" \
    --name "${GITHUB_RUNNER_NAME}" \
    --unattended \
    --replace \
    && touch .runner   # marker to avoid re‑configuring on every start
fi

# Finally start the runner (foreground)
exec ./run.sh "$@"