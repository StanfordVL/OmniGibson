#!/usr/bin/env bash

docker push nvcr.io/ stanfordvl/omnigibson:latest
docker push nvcr.io/ stanfordvl/omnigibson:$(sed -ne "s/.*version= *['\"]\([^'\"]*\)['\"] *.*/\1/p" setup.py)