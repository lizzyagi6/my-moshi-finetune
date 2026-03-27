import os
import json
import argparse
import sphn
from mutagen import File
from pathlib import Path
import ipdb
import re

def de_root(paths, root_dir):
    """
    Converts absolute paths to relative paths based on the root_dir.
    """
    root = Path(root_dir).resolve()
    # Returns the path relative to root_dir
    return [p.resolve().relative_to(root) for p in paths]


def get_durations(input_path, ext, align_dir):
    # Standard glob for all files, filtered by extension
    paths = [
        f for f in Path(input_path).iterdir() 
        if f.suffix.lower() == f".{ext}" and not f.name.startswith("._")
    ]

    valid_data = []
    
    # Ensure align_dir is a Path object relative to input_path
    align_path = Path(input_path).parent / align_dir
    
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
        
    # Cast to list so it's indexable and reusable
    return list(zip(*valid_data))


def generate_jsonl(root_dir, folder: str, ext: str, align_dir: str):
    """
    root_dir: e.g. /mnt/.../elon_convo
    folder_dir: e.g. 'pod17'
    align_dir: e.g. 'whisper'
    """

    file_dir = os.path.join(root_dir, folder)
    input_path = Path(file_dir).resolve()
    
    # Unpack safely
    result = get_durations(input_path / 'data_stereo', ext, align_dir)
    if not result or not result[0]:
        print(f"No valid file pairs found for {folder}.")
        return

    paths, durations = result

    dataset_path = Path(root_dir).resolve() / folder

    # Convert to relative paths based on root_dir
    rel_paths = de_root(paths, dataset_path)

    # Use input_path.parent (the root_dir) or file_dir for the output manifest location
    out_file = dataset_path / '_all_.jsonl'

    print(f"Writing to: {out_file}, for {len(rel_paths)} audio files")
    with open(out_file, "w") as fobj:
        for p, d in zip(rel_paths, durations):
            # p is now a relative Path object
            json.dump({"path": str(p), "duration": d}, fobj)
            fobj.write("\n")
    print(f"Done! Processed {len(rel_paths)} files. -> {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a JSONL manifest for audio and alignment files.")
    parser.add_argument("root_dir", type=str, help="parent Directory containing of all the data")
    parser.add_argument("folder", type=str, help="Directory containing the audio files")
    parser.add_argument("--align_dir", type=str, required=True, help="Subdirectory containing .json alignments")
    parser.add_argument("--ext", type=str, default='wav', help="audio extension e.g. wav, flac, opus")
    args = parser.parse_args()
    
    generate_jsonl(args.root_dir, args.folder, args.ext, args.align_dir)


#usage:
#export DATA_DIR=/mnt/dnn3/nfs/r2/training_data/elon_convo
#uv run python gen_manifest.py $DATA_DIR pods17 --align_dir whisper --ext flac