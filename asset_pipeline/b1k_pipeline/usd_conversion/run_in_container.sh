#!/bin/bash
set -e

apt-get update
apt-get install -y zip unzip
pip install tqdm

OMNIGIBSON_NO_OMNIVERSE=1 python -c "from omnigibson.utils.asset_utils import download_key; download_key()"

# First, unzip the dataset into the temp directory
unzip /ig_pipeline/artifacts/og_dataset.zip -d /tmp/og_dataset_zip
mv /tmp/og_dataset_zip/aggregate /tmp/og_dataset

# Then, run the conversion script
cd /ig_pipeline
python -m b1k_pipeline.usd_conversion.usdify_dataset

# Then, re-zip the dataset
cd /tmp
tar -czvf /ig_pipeline/artifacts/og_dataset_encrypted.tar.gz og_dataset
