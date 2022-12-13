#!/usr/bin/env bash

docker push nvcr.io/ StanfordVL/omnigibson:latest
docker push nvcr.io/ StanfordVL/omnigibson:$(sed -ne "s/.*version= *['\"]\([^'\"]*\)['\"] *.*/\1/p" setup.py)