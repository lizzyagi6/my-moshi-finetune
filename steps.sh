#!/bin/bash
set -e

# --- Configuration ---
LOCAL_DATA_DIR="../moshi_ft_runs/data/elon_f"
RUN_DIR="../moshi_ft_runs/runs"
MODE=${1:-"all"} # Default to 'all' if no argument is provided

# Load .env variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "Warning: .env file not found!"
fi

# --- Stages ---
setup() {
    echo "--- Stage: Setup ---"
    mkdir -p "$LOCAL_DATA_DIR" "$RUN_DIR"
    # Install uv if missing
    if ! command -v uv &> /dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        # Ensure it's available in the current shell session
        export PATH="$HOME/.cargo/bin:$PATH" 
    fi
    uv add ipdb
}

sync_down() {
    echo "--- Stage: Syncing Data FROM R2 ---"
    uv run aws s3 sync \
        s3://duplexdata/data/elon_f \
        "$LOCAL_DATA_DIR" \
        --endpoint-url "$R2_END_POINT_URL" \
        --region auto \
        --size-only

    echo "--- Stage: Syncing Experiment/Run dir FROM R2 ---"
    uv run aws s3 sync \
        "s3://duplexdata/experiments/user_$USER_NAME/" \
        "$RUN_DIR" \
        --endpoint-url "$R2_END_POINT_URL" \
        --region auto \
        --size-only
}

manifest() {
    echo "--- Stage: Generating Data Manifest ---"
    # Note: Using the directory you defined in your previous script
    uv run --with sphn python gen_manifest.py "$LOCAL_DATA_DIR/data_stereo"
}

train() {
    echo "--- Stage: Training (1 GPU) ---"
    export CUDA_VISIBLE_DEVICES=0
    # Use --no-build-isolation here as requested for your environment
    uv run --no-build-isolation torchrun --nproc-per-node 1 \
        -m train example/elon_f.yaml
}

sync_up() {
    echo "--- Stage: Syncing Results TO R2 ---"
    uv run aws s3 sync \
        "$RUN_DIR" \
        "s3://duplexdata/experiments/user_$USER_NAME/" \
        --endpoint-url "$R2_END_POINT_URL" \
        --region auto \
        --size-only 
}

# --- Execution Logic ---

case $MODE in
    "setup")
        setup
        ;;
    "sync-down")
        sync_down
        ;;
    "manifest")
        manifest
        ;;
    "train")
        train
        ;;
    "sync-up")
        sync_up
        ;;
    "all")
        setup
        sync_down
        manifest
        train
        ;;
    *)
        echo "Usage: $0 {setup|sync-down|manifest|train|sync-up|all}"
        exit 1
        ;;
esac

echo "Done: $MODE"
