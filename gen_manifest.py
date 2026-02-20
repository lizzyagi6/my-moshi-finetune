import os
import json
import argparse
import sphn
from pathlib import Path

def generate_jsonl(file_dir):
    # Convert to Path object for easier manipulation
    input_path = Path(file_dir).resolve()
    
    # Get all .wav files
    paths = [str(f) for f in input_path.glob("*.wav")]
    
    if not paths:
        print(f"No .wav files found in {input_path}")
        return

    print(f"Calculating durations for {len(paths)} files...")
    durations = sphn.durations(paths)

    # Place data.jsonl one directory up from the wav files
    out_file = input_path.parent / 'data.jsonl'
    
    print(f"Writing to: {out_file}")
    with open(out_file, "w") as fobj:
        for p, d in zip(paths, durations):
            if d is None:
                continue
            json.dump({"path": p, "duration": d}, fobj)
            fobj.write("\n")
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a JSONL manifest for wav files.")
    
    # Positional argument for the directory
    parser.add_argument("file_dir", type=str, help="Directory containing the .wav files")
    
    args = parser.parse_args()
    
    generate_jsonl(args.file_dir)
