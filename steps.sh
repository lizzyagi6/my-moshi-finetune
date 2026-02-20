#!/bin/bash

#Define your local path
LOCAL_DIR="../moshi_ft_runs/data/elon_f"
RUN_DIR="../moshi_ft_runs/runs"

mkdir -p "$LOCAL_DIR"
mkdir -p "$RUN_DIR"

# 1. Load variables from .env file in the current directory
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "Error: .env file not found!"
    exit 1
fi

echo "Starting sync from R2 to $LOCAL_DIR..."

# 4. Use uv to run the aws cli (ensures it's installed/available)
# We add --size-only to speed up training starts if files haven't changed
uv run aws s3 sync \
    s3://duplexdata/elon_f \
    "$LOCAL_DIR" \
    --endpoint-url "$R2_END_POINT_URL" \
    --region auto \
    --size-only

echo "Data Sync complete. ..."

echo "Generate data manifest file"
uv run --with sphn python gen_manifest.py $LOCAL_DIR/data_stereo

echo "Training starts with 1 GPU"
export CUDA_VISIBLE_DEVICES=0
uv run --no-build-isolation torchrun --nproc-per-node 1 \
    -m train example/elon_f.yaml