#!/bin/bash
set -e

# --- Configuration ---
LOCAL_DATA_DIR="../moshi_ft_runs/training_data/elon_convo"
RUN_DIR="../moshi_ft_runs/runs"
MODE=${1:-"all"} # Default to 'all' if no argument is provided
SUB_DIR=$2  # Capture the second argument (the folder name)

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
        #export PATH="$HOME/.cargo/bin:$PATH" 
    fi
    #uv add ipdb sphn pydub mutagen librosa
    uv sync
}

sync_down() {
    echo "--- Stage: Syncing Data FROM R2 ---"
    uv run aws s3 sync \
        s3://duplexdata/training_data/elon_convo/ \
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


train() {
    echo "--- Stage: Training (1 GPU) ---"
    export CUDA_VISIBLE_DEVICES=0
    # Use --no-build-isolation here as requested for your environment
    uv run torchrun --nproc-per-node 1 -m train example/elon_pod.yaml 
}

sync_up() {
    local target_folder=$1

    # If a subfolder is provided, append it to the paths; otherwise use base dirs
    local src="$RUN_DIR/${target_folder}"
    local dest="s3://duplexdata/experiments/user_$USER_NAME/${target_folder}"

    echo "--- Stage: Syncing $src TO $dest ---"

    uv run aws s3 sync \
        "$src" \
        "$dest" \
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
        sync_up "$SUB_DIR"
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
