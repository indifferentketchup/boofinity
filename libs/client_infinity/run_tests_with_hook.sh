#!/bin/bash

set -euo pipefail

# Function to handle cleanup
cleanup() {
  echo "Cleaning up..."
  if [[ -n "${INFINITY_PID:-}" ]]; then
    kill "$INFINITY_PID"
  fi
}

# Set up the trap to run the cleanup function on EXIT or any error
trap cleanup EXIT

# Start boofinity in the background
DO_NOT_TRACK=1 boofinity v2 --log-level error --engine debugengine --no-model-warmup --port 7994 &
INFINITY_PID=$!
echo "boofinity started with PID $INFINITY_PID"

# Wait for boofinity to be ready
for i in {1..10}; do
  if wget -q --spider http://0.0.0.0:7994/openapi.json; then
    echo "boofinity is ready."
    break
  else
    echo "Waiting for boofinity to be ready..."
    sleep 1
  fi
done

# Run the tests
cd infinity_client && \
poetry install && \
poetry run pip install pytest requests && \
poetry run python -m pytest ../tests

# copy the readme to docs
cd ..
cp ./infinity_client/README.md ./../../docs/docs/client_infinity.md

# Cleanup will be called due to the trap