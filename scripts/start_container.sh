#!/bin/bash

# Default image name if not provided
IMAGE_NAME=${1:-htr-ood-image}
DEVICE=${2:-0}
SHM_SIZE=${3:-24}

# Build the Docker image
echo "Building Docker image as '$IMAGE_NAME'..."
docker build -t "$IMAGE_NAME" .

# Run the container
echo "Running container from image '$IMAGE_NAME'..."
docker run -itd --rm \
  --name "${IMAGE_NAME}-container" \
  --gpus device="$DEVICE" \
  -v "$(pwd)":/workspace/ \
  --shm-size="${SHM_SIZE}gb" \
  "$IMAGE_NAME"