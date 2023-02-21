#!/bin/bash
set -e

# First, unzip the dataset into the temp directory
# TODO
unzip /data/ig_pipeline/artifacts/og_dataset.zip -d /tmp/og_dataset_zip
mv /tmp/og_dataset_zip/aggregate /tmp/og_dataset

# Then, run the conversion script
cd /data/ig_pipeline
python -m b1k_pipeline.usd_conversion.usdify_dataset

# Then, re-zip the dataset
cd /tmp
tar -czvf /data/ig_pipeline/artifacts/og_dataset_encrypted.tar.gz og_dataset