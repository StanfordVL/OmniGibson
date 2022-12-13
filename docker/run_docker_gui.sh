#!/usr/bin/env bash

DATA_PATH=${1:-'~/omnigibson-data'}

docker login nvcr.io
docker run \
    --gpus all \
    --privileged \
    -e DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v DATA_PATH:/data \
    --network=host --rm -it stanfordvl/omnigibson
