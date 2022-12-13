#!/usr/bin/env bash

DATA_PATH=${1:-'~/omnigibson-data'}

docker login nvcr.io
docker run \
    --gpus all \
    --privileged \
    -e DISPLAY \
    -e OMNIGIBSON_HEADLESS=1 \
    -v DATA_PATH:/data \
    --network=host --rm -it omnigibson bash -c "source ~/.bashrc && jupyter lab --allow-root"
