import argparse
import gc
import gzip
import importlib
import json
import logging
import os
import sys
import time
import multiprocessing
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import sphn
import submitit
import torch
import torchaudio.functional as F
import whisper_timestamped as whisper
import torchaudio.transforms as T
import librosa
from tqdm import tqdm

transcribe = importlib.import_module("whisper_timestamped.transcribe")
old_get_vad_segments = transcribe.get_vad_segments
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000


# From https://github.com/facebookresearch/flashy/blob/3881ba496437cbbc34aea21c6ea42453d298e006/flashy/utils.py#L41
@contextmanager
def write_and_rename(path: Path, mode: str = "wb", suffix: str = ".tmp", pid=False):
    """
    Write to a temporary file with the given suffix, then rename it
    to the right filename. As renaming a file is usually much faster
    than writing it, this removes (or highly limits as far as I understand NFS)
    the likelihood of leaving a half-written checkpoint behind, if killed
    at the wrong time.
    """
    tmp_path = str(path) + suffix
    if pid:
        tmp_path += f".{os.getpid()}"
    with open(tmp_path, mode) as f:
        yield f
    os.rename(tmp_path, path)


def load_audio_paths(egs_path: Path) -> list[Path]:
    """Load audio paths from a JSONL egs file.

    Args:
        egs_path (Path): Path to JSONL file, might be gzipped.
    Returns:
        list of paths
    """
    open_fn = gzip.open if str(egs_path).lower().endswith(".gz") else open
    with open_fn(egs_path, "rb") as fp:  # type: ignore
        lines = fp.readlines()
    paths: list[Path] = []
    for line in lines:
        d = json.loads(line)
        paths.append(Path(d["path"]))
    return paths


def init_logging(verbose: bool = False):
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%m-%d %H:%M:%S",
        force=True,
    )


def process_one(
    in_file: Path,
    out_file: Path,
    language: str,
    w_model,
    params: "Params",
    channel: int = 0,
    seek_time: float | None = None,
    duration: float | None = None,
    use_librosa: bool = False, # NEW FLAG
):
    logger.debug("Loading audio %s", in_file)
    gc.collect()
    torch.cuda.empty_cache()

    if use_librosa or in_file.suffix.lower() == ".opus":
        if in_file.suffix.lower() == ".opus" and not use_librosa:
            logger.warning(".opus detected: forcing librosa as sphn does not support this format.")
        
        # LIBROSA PATH
        # offset=seek_time, duration=duration. sr=None keeps native rate.
        x, sr = librosa.load(
            str(in_file), 
            sr=None, 
            mono=False, 
            offset=seek_time if seek_time else 0.0, 
            duration=duration
        )
        if x.ndim == 1: x = x[None, :] # Ensure 2D for channel indexing
        x = torch.from_numpy(x).cuda()
    else:
        # OLD SPHN PATH
        x, sr = sphn.read(in_file, start_sec=seek_time, duration_sec=duration)
        x = torch.from_numpy(x).cuda()

    # Calculate duration & resample
    dur = x.shape[-1] / sr
    if dur > 3600 * 4:
        raise RuntimeError("File is too long for now.")

    vocals = x[channel][None]
    
    # Standardize resampling to 16kHz for Whisper
    if sr != SAMPLE_RATE:
        resampler = T.Resample(sr, SAMPLE_RATE).to(x.device)
        vocals = resampler(vocals)
        sr = SAMPLE_RATE

    def new_get_vad_segments(*args, **kwargs):
        segs = old_get_vad_segments(*args, **kwargs)
        outs = []
        last_end = 0
        d = int(SAMPLE_RATE * params.keep_silence_in_segments)
        logger.debug("Reintroducing %d samples at the boundaries of the segments.", d)
        for seg in segs:
            outs.append(
                {"start": max(last_end, seg["start"] - d), "end": seg["end"] + d}
            )
            last_end = outs[-1]["end"]
        return outs

    if params.keep_silence_in_segments:
        transcribe.get_vad_segments = new_get_vad_segments  # type: ignore

    vocals = vocals.cpu()
    vocals = vocals.numpy()[0]
    chunks = []
    this_duration = vocals.shape[-1] / sr
    logger.debug("Transcribing block in %s, of duration %.1f", language, this_duration)
    pipe_output = whisper.transcribe(
        w_model,
        vocals,
        language=language,
        vad="auditok" if this_duration > 10 else None,
        best_of=5,
        beam_size=5,
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        verbose=None,
    )

    for segment in pipe_output["segments"]:
        if "words" not in segment:
            logger.error("No words in %s: %r", in_file, segment)
            continue
        for word in segment["words"]:
            try:
                chunks.append(
                    {"text": word["text"], "timestamp": (word["start"], word["end"])}
                )
            except KeyError:
                logger.error("Missing key in %s: %r", in_file, word)
                raise
    outputs = {
        "alignments": [
            [chunk["text"], chunk["timestamp"], "SPEAKER_MAIN"] for chunk in chunks
        ]
    }
    logger.debug("Whisper applied.")
    with write_and_rename(out_file, "w", pid=True) as f:
        json.dump(outputs, f, ensure_ascii=False)
    logger.debug("Wrote file %s", out_file)


def run(params: "Params", shard: int = 0):
    init_logging(params.verbose)
    # local_rank = dora.distrib.get_distrib_spec().local_rank
    # shard += local_rank
    local_rank = 0
    logger.info("Hello, world, this is shard %d / %d.", shard, params.shards)
    params.shard = shard
    torch.cuda.set_device(local_rank)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    os.environ["OMP_NUM_THREADS"] = "2"

    logger.info("Loading all models.")
    device = torch.device(f"cuda:{local_rank}")
    w_model = whisper.load_model(params.whisper_model, device=device)

    logger.info("Loading egs %s.", params.egs)
    paths = load_audio_paths(params.egs)
    kept_paths = paths[shard :: params.shards]
    logger.info("Processing % 8d files out of % 8d.", len(kept_paths), len(paths))
    del paths

    for idx, path in tqdm(enumerate(kept_paths)):
        if (idx + 1) % 100 == 0:
            logger.info("Processing % 8d / % 8d files.", idx + 1, len(kept_paths))
        out_file = path.with_suffix(".json")
        err_file = path.with_suffix(".json.err")
        if out_file.exists():
            continue
        if err_file.exists() and not params.rerun_errors:
            continue
        try:
            if path.stat().st_size < 1000:
                logger.warning("Small file detected: %s", path)
                continue
            logger.debug("Processing file %s, out file is %s", path, out_file)
            process_one(
                path,
                out_file,
                channel=0,
                language=params.lang,
                w_model=w_model,
                params=params,
            )
        except Exception as err:
            if "cuda" in repr(err).lower():
                raise
            logger.exception("Error processing %s", path)
            err_file.touch()
            continue


def run_local(params: "Params", folder: str, shard: int = 0):
    # 1. Initialization
    # Assuming init_logging is defined elsewhere
    # init_logging(params.verbose) 
    
    local_rank = shard 
    logger.info("Hello, world, this is shard %d / %d.", shard, params.shards)
    params.shard = shard
    
    # Set GPU device
    torch.cuda.set_device(local_rank)
    os.environ["OMP_NUM_THREADS"] = "2"

    logger.info("Loading all models.")
    device = torch.device(f"cuda:{local_rank}")
    w_model = whisper.load_model(params.whisper_model, device=device)

    # 2. Corrected File Gathering
    # We use pathlib to find both .wav and .flac files
    folder_path = Path(folder)
    # This finds all files with either extension
    paths = [p for p in folder_path.glob("**/*") if p.suffix.lower() in {".wav", ".flac"}]
    
    # Sharding logic: slice the list based on current shard index
    kept_paths = paths[shard :: params.shards]
    logger.info(f"shard: {shard}. Processing {len(kept_paths):8d} files out of {len(paths):8d}.")
    del paths

    # 3. Processing Loop
    for idx, path in tqdm(enumerate(kept_paths), total=len(kept_paths)):
        if (idx + 1) % 100 == 0:
            logger.info("Processing % 8d / % 8d files.", idx + 1, len(kept_paths))
            
        # Define the output directory (./whisper/)
        output_dir = path.parent / "whisper"
        output_dir.mkdir(exist_ok=True)

        # Define files inside that folder
        out_file = output_dir / path.with_suffix(".json").name
        err_file = output_dir / path.with_suffix(".json.err").name
        
        # Skip logic
        if out_file.exists():
            continue

        if err_file.exists() and not getattr(params, 'rerun_errors', False):
            continue
            
        try:
            # Check file size (skipping empty/corrupt files)
            if path.stat().st_size < 1000:
                logger.warning("Small file detected: %s", path)
                continue
                
            logger.debug("Processing file %s, out file is %s", path, out_file)
            
            # Assuming process_one is defined elsewhere
            process_one(
                path,
                out_file,
                channel=0,
                language=params.lang,
                w_model=w_model,
                params=params,
            )
            
        except Exception as err:
            # If it's a CUDA error, re-raise to stop the process
            if "cuda" in repr(err).lower():
                raise
            logger.exception("Error processing %s", path)
            err_file.touch() # Mark as error to avoid infinite retry
            continue


@dataclass
class Params:
    egs: Path
    verbose: bool
    lang: str
    whisper_model: str
    keep_silence_in_segments: float
    rerun_errors: bool
    shards: int
    shard: int = 0
    folder: str = ""


def main():
    parser = argparse.ArgumentParser(
        description="Annotate with transcripts and diarization."
    )
    parser.add_argument("--egs", type=Path, help="Path to egs jsonl.gz file")
    parser.add_argument("--folder", type=Path, help="folder with .wav or .flac files")
    parser.add_argument(
        "--log_folder", type=Path, default=Path.home() / "tmp" / "mass_annotate"
    )
    parser.add_argument(
        "-S", "--shards", type=int, default=1, help="Number of shards to schedule."
    )
    parser.add_argument("--lang", default="en", help="Force the language.")
    parser.add_argument("--partition", default="", help="Which partition to use.")
    parser.add_argument(
        "--whisper_model",
        default="medium",
        help="Which whisper to use, use medium for stereo!",
    )
    parser.add_argument(
        "--rerun_errors",
        action="store_true",
        help="Ignore previous errors and rerun failed files.",
    )
    parser.add_argument(
        "--keep_silence_in_segments",
        type=bool,
        default=True,
        help="Keep some of the silence at the beginnning / end of segments"
        " in whisper timestamped. This can mitigate words being misplaced.",
    )
    parser.add_argument(
        "-l", "--local", action="store_true", help="Run locally to debug."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")

    args = parser.parse_args()

    if args.whisper_model == "large-v3":
        logger.warning(
            "You probably want to use medium whisper for stereo with VAD detection."
        )
    init_logging(args.verbose)
    kwargs = dict(args.__dict__)
    kwargs.pop("local")
    kwargs.pop("partition")
    kwargs.pop("log_folder")
    params = Params(**kwargs)

    if args.local:
        #params.shards = 1
        #run(params)

        params.shards = 2
        processes = []

        for i in range(params.shards):
            # We create a separate process for each shard
            p = multiprocessing.Process(
                target=run_local, 
                args=(params, str(args.folder), i)
            )
            p.start()
            processes.append(p)

        # Wait for both shards to finish
        for p in processes:
            p.join()


    else:
        executor = submitit.SlurmExecutor(folder=args.log_folder)
        executor.update_parameters(
            cpus_per_task=6,
            ntasks_per_node=1,
            gpus_per_node=1,
            time=60 * 24 * 10,
            signal_delay_s=30,
            partition=args.partition,
            stderr_to_stdout=True,
            array_parallelism=1000,
            exclude="",
            job_name="annotate",
        )
        jobs = []
        with executor.batch():
            for shard in range(args.shards):
                jobs.append(executor.submit(run, params, shard))
        print("Job id:", jobs[0].job_id)
        while True:
            done = 0
            for job in jobs:
                if job.done():
                    done += 1
            print(f"{done:04d} / {len(jobs):04d} jobs done.", end="\r")
            if done == len(jobs):
                break
            time.sleep(10.0)


if __name__ == "__main__":
    main()

#usage: uv run python annotate.py -l --rerun_errors --folder "/mnt/dnn3/nfs/r2/training_data/elon_convo/pods/"
