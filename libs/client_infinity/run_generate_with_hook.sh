#!/bin/bash

set -euo pipefail

# Function to handle cleanup
cleanup() {
  echo "Cleaning up..."
  if [[ -n "${BOOFINITY_PID:-}" ]]; then
    kill "$BOOFINITY_PID"
  fi
}

# Set up the trap to run the cleanup function on EXIT or any error
trap cleanup EXIT

# Start boofinity in the background
DO_NOT_TRACK=1 boofinity v2 --log-level error --engine debugengine --port 7993 &
BOOFINITY_PID=$!
echo "boofinity started with PID $BOOFINITY_PID"

# Wait for boofinity to be ready
for i in {1..10}; do
  if wget -q --spider http://0.0.0.0:7993/openapi.json; then
    echo "boofinity is ready."
    break
  else
    echo "Waiting for boofinity to be ready..."
    sleep 1
  fi
done

# Run the tests
python -m pip install openapi-python-client==0.21.1 && \
	 openapi-python-client generate  \
	  --url http://0.0.0.0:7993/openapi.json \
	  --config client_config.yaml \
	  --meta poetry \
	  --overwrite \
	  --custom-template-path=./template

# copy the readme to docs
cp ./template/vision_client.py ./infinity_client/infinity_client/vision_client.py
cp ./infinity_client/README.md ./../../docs/docs/client_infinity.md
# Cleanup will be called due to the trap
