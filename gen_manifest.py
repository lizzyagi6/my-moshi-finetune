import os
import json
import argparse
import sphn
from mutagen import File
from pathlib import Path
import ipdb
import re


def get_durations(input_path, ext, align_dir):
    # Standard glob for all files, filtered by extension
    paths = [
        f for f in Path(input_path).iterdir() 
        if f.suffix.lower() == f".{ext}" and not f.name.startswith("._")
    ]

    valid_data = []
    
    # Ensure align_dir is a Path object relative to input_path
    align_path = Path(input_path) / align_dir
    
    if not align_path.is_dir():
        print(f"Error: Alignment directory '{align_path}' does not exist.")
        return [], []

    for p in sorted(paths):
        # Check for corresponding non-empty JSON file
        json_file = align_path / f"{p.stem}.json"
        
        if not json_file.exists() or json_file.stat().st_size == 0:
            print(f"Skipping {p.name}: Missing or empty alignment file {json_file.name}")
            continue

        audio = File(p)
        if audio is not None and audio.info is not None:
            valid_data.append((p, audio.info.length))
        else:
            print(f"Skipping {p.name}: Could not read audio info")
            
    if not valid_data:
        return [], []
        
    return zip(*valid_data)


def generate_jsonl(file_dir, ext: str, align_dir: str):
    input_path = Path(file_dir).resolve()
    paths, durations = get_durations(input_path, ext, align_dir)

    if not paths:
        print("No valid file pairs found. Exiting.")
        return

    out_file = input_path.parent / 'data_.jsonl'
    
    print(f"Writing to: {out_file}, for {len(paths)} audio files")
    with open(out_file, "w") as fobj:
        for p, d in zip(paths, durations):
            json.dump({"path": str(p), "duration": d}, fobj)
            fobj.write("\n")
    print(f"Done! Processed {len(paths)} files.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a JSONL manifest for audio and alignment files.")
    parser.add_argument("file_dir", type=str, help="Directory containing the audio files")
    parser.add_argument("--align_dir", type=str, required=True, help="Subdirectory containing .json alignments")
    parser.add_argument("--ext", type=str, default='wav', help="audio extension e.g. wav, flac, opus")
    args = parser.parse_args()
    
    generate_jsonl(args.file_dir, args.ext, args.align_dir)


#usage:
#uv run python ./gen_manifest.py  "/mnt/dnn3/nfs/r2/training_data/elon_convo/pods" --align_dir whisper --ext flac



