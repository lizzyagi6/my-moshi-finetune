#!/bin/bash
set -e

FORCE=false
POS_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE=true
            shift
            ;;
        *)
            POS_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ ${#POS_ARGS[@]} -lt 2 ]; then
    echo "Usage: $0 [--force] /path/to/source_folder /path/to/output_folder"
    exit 1
fi

TARGET_DIR="${POS_ARGS[0]}"
OUTPUT_DIR="${POS_ARGS[1]}"

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Source directory $TARGET_DIR not found."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Updated to loop through both .opus and .wav files
for f in "$TARGET_DIR"/*.{opus,wav}; do
    [ -e "$f" ] || continue
    
    # Dynamically remove the extension regardless of length
    base_name=$(basename "${f%.*}")
    output_file="$OUTPUT_DIR/${base_name}.flac"

    if [ -f "$output_file" ] && [ "$FORCE" = false ]; then
        echo "Skipping (already exists): $output_file"
        continue
    fi
    
    echo "Converting: $f -> $output_file"
    
    ffmpeg -y -i "$f" -c:a flac -sample_fmt s16 -ar 24000 -ac 2 "$output_file" -loglevel warning
done
echo "Done! All files processed in: $OUTPUT_DIR"

# usage: ./opus_to_flac.sh /mnt/dnn3/nfs/r2/training_data/elon_podcasts/data_stereo/ /mnt/dnn3/nfs/r2/training_data/elon_convo/pods