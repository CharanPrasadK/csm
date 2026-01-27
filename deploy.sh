#!/bin/bash

# Configuration
CONTAINER_NAME="csm-tts-container"
IMAGE_NAME="csm-tts-cuda"
PORT=5006

# 1. Determine Environment Variable Strategy
if [ -f ".env" ]; then
    ENV_ARG="--env-file .env"
else
    ENV_ARG="-e HF_TOKEN=$HF_TOKEN"
fi

# 2. Cleanup old container
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "Stopping and removing old container..."
    docker stop $CONTAINER_NAME >/dev/null 2>&1
    docker rm $CONTAINER_NAME >/dev/null 2>&1
fi

# 3. Rebuild
echo "Rebuilding image '$IMAGE_NAME'..."
docker build -t $IMAGE_NAME .
if [ $? -ne 0 ]; then
    echo "Error: Docker build failed."
    exit 1
fi

# 4. Run with Certificates Mounted
echo "Starting container with GPU and SSL support..."
# We map /etc/letsencrypt on host -> /etc/letsencrypt in container
docker run -d \
  --name $CONTAINER_NAME \
  --gpus all \
  -p $PORT:$PORT \
  $ENV_ARG \
  -v /etc/letsencrypt:/etc/letsencrypt:ro \
  $IMAGE_NAME

if [ $? -eq 0 ]; then
    echo "Deployment successful!"
    echo "Logs: docker logs -f $CONTAINER_NAME"
else
    echo "Error: Failed to start container."
    exit 1
fi