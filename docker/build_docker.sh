#!/usr/bin/env bash

docker build \
    -t StanfordVL/omnigibson:latest \
    -t StanfordVL/omnigibson:$(sed -ne "s/.*version= *['\"]\([^'\"]*\)['\"] *.*/\1/p" setup.py) \
    -f docker/Dockerfile \
    .