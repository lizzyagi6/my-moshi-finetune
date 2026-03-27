import os
import sys
if os.getenv('LOCAL_MOSHI_DIR'):
    sys.path.insert(0, os.getenv('LOCAL_MOSHI_DIR'))

import dataclasses
import logging
import pprint
import shutil
from contextlib import ExitStack
from pathlib import Path
import ipdb

import fire
import torch.cuda
import torch.distributed as dist
from torch.optim import AdamW, lr_scheduler, Muon
# from torch.profiler import ProfilerActivity, profile

from aim import Run

from finetune.args import TrainArgs
from finetune.checkpointing import Checkpointer
from finetune.data.data_loader import build_data_loader
from finetune.data import dataset
from finetune.data.interleaver import InterleavedTokenizer, Interleaver
from finetune.distributed import (
    BACKEND,
    avg_aggregate,
    get_rank,
    get_world_size,
    is_torchrun,
    set_device,
)
from finetune.eval import evaluate
from finetune.loss import compute_loss_with_mask
from finetune.mixed_precision import (
    downcast_mixed_precision,
    prepare_mixed_precision,
    upcast_mixed_precision,
)
from finetune.monitoring.metrics_logger import (
    MetricsLogger,
    eval_log_msg,
    get_eval_logs,
    get_train_logs,
    train_log_msg,
)
from finetune.monitoring.utils import set_logger
from finetune.utils import TrainState, logged_closing, set_random_seed
from finetune.wrapped_model import get_fsdp_model
from moshi.models import loaders

logger = logging.getLogger("train")

def main_logger_info(message: str) -> None:
    if get_rank() == 0:
        logger.info(message)

# --- Helper Utility for 2026 SOTA Optimizer Splitting ---
def get_optimizer_groups(model):
    muon_params = []
    adamw_params = []
    for name, p in model.named_parameters():
        print(name, p.ndim, p.requires_grad)

    #if int(os.environ.get("RANK", 0)) == 0: ipdb.set_trace()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Muon only handles >= 2D parameters (Weight Matrices)
        # We exclude Embeddings because they are usually updated sparsely
        if p.ndim >= 2 and "emb" not in name.lower():
            muon_params.append(p)
        else:
            adamw_params.append(p)
    return muon_params, adamw_params


def train(config: str):
    args: TrainArgs = TrainArgs.load(config, drop_extra_fields=False)
    set_logger(logging.INFO)

    with ExitStack() as exit_stack:
        _train(args, exit_stack)
    logger.info("Closed everything!")


def _train(args: TrainArgs, exit_stack: ExitStack):
    # 1. Initial setup and checks
    set_random_seed(args.seed)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Init NCCL
    if "LOCAL_RANK" in os.environ:
        set_device()
        logger.info("Going to init comms...")

        dist.init_process_group(backend=BACKEND)
    else:
        logger.error(
            "PyTorch environment is not correctly initialized. This message should only be displayed when testing."
        )

    # 2. Init run dir
    main_logger_info(f"Run dir: {args.run_dir}")
    run_dir = Path(args.run_dir)

    if is_torchrun():
        if run_dir.exists() and not args.overwrite_run_dir:
            raise RuntimeError(
                f"Run dir {run_dir} already exists. Make sure to either rename `run_dir` or remove {run_dir}."
            )
        elif run_dir.exists():
            main_logger_info(f"Removing run dir {run_dir}...")
            shutil.rmtree(run_dir)

    if args.full_finetuning:
        assert not args.lora.enable, "LoRA should not be enabled for full finetuning."
    else:
        assert args.lora.enable, "LoRA should be enabled for partial finetuning"

    dist.barrier()
    run_dir.mkdir(exist_ok=True, parents=True)

    args_path = run_dir / "args.yaml"
    if not args_path.exists():
        args.save(args_path)

    main_logger_info(f"TrainArgs: {pprint.pformat(dataclasses.asdict(args))}")

    # 3. Get loggers
    metrics_logger: MetricsLogger = MetricsLogger(
        run_dir,
        tag="train",
        is_master=get_rank() == 0,
        wandb_args=args.wandb,
        config=dataclasses.asdict(args),
    )
    exit_stack.enter_context(logged_closing(metrics_logger, "metrics_logger"))

    eval_logger: MetricsLogger = MetricsLogger(
        run_dir,
        tag="eval",
        is_master=get_rank() == 0,
        wandb_args=args.wandb,
        config=dataclasses.asdict(args),
    )
    exit_stack.enter_context(logged_closing(eval_logger, "eval_logger"))

    # 4.1 Load function calling audio encoder and tokenizer
    main_logger_info("Loading Mimi and Moshi...")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        hf_repo=args.moshi_paths.hf_repo_id,
        moshi_weights=args.moshi_paths.moshi_path,
        mimi_weights=args.moshi_paths.mimi_path,
        tokenizer=args.moshi_paths.tokenizer_path,
        config_path=args.moshi_paths.config_path,
    )

    lm_config = (
        loaders._lm_kwargs
        if checkpoint_info.raw_config is None
        else checkpoint_info.raw_config
    )
    lm_config["lora"] = args.lora.enable
    lm_config["lora_rank"] = args.lora.rank
    lm_config["lora_scaling"] = args.lora.scaling

    mimi = checkpoint_info.get_mimi(device="cuda")
    mimi.eval()
    for p in mimi.parameters():
        p.requires_grad = False

    # 4.2 Load and shard model, prepare interleaver for audio/text tokens.
    model = get_fsdp_model(args, checkpoint_info)

    spm = checkpoint_info.get_text_tokenizer()

    #if int(os.environ.get("RANK", 0)) == 0:    ipdb.set_trace()

    interleaver = Interleaver(
        spm,
        mimi.frame_rate,
        model.text_padding_token_id,
        model.end_of_text_padding_id,
        model.zero_token_id,
        keep_main_only=True,
    )
    interleaved_tokenizer = InterleavedTokenizer(
        mimi, interleaver, duration_sec=args.duration_sec, 
        transcription_folder_name='whisper'
    )

    #if int(os.environ.get("RANK", 0)) == 0: ipdb.set_trace()

    # 5. Load data loaders
    data_loader = build_data_loader(
        instruct_tokenizer=interleaved_tokenizer,
        args=args.data,
        batch_size=args.batch_size,
        seed=args.seed,
        rank=get_rank(),  # DDP rank
        world_size=get_world_size(),  # DDP world_size
        is_eval=False,
    )
    
    dataset.UniqueWavPaths = set()

    if 0:
        import sphn
        import numpy as np

        #path = "/mnt/dnn3/nfs/r2/training_data/elon_podcasts/data_stereo/aMcmuKTfr54_part3.flac"
        path = "/mnt/dnn3/nfs/r2/training_data/elon_podcasts/data.jsonl"
        reader = sphn.dataset_jsonl(path, duration_sec=30.0, 
                                    num_threads=4,
                                    sample_rate=24000,
                                    pad_last_segment=True,
                                    )

        # Try to read the first item
        try:
            for item in reader:
                # Access the 'data' key to get the numpy array
                audio_data = item['data'] 
                print(f"Successfully read FLAC!")
                print(f"Shape: {audio_data.shape}") # Now .shape will work
                print(f"Sample Rate: {item['sample_rate']}")
                break

        except Exception as e:
            print(f"Failed to read FLAC: {e}")

    if 0:
        for i in range(3):
            aa = next(data_loader)
            ipdb.set_trace()

    #if int(os.environ.get("RANK", 0)) == 0:    ipdb.set_trace()

    # 6. Load model
    # Define mixed precision
    param_dtype = getattr(torch, args.param_dtype)
    optim_dtype = torch.float32

    assert args.lora is not None, "`args.lora` should be set to a valid value."

    opt_muon = None
    scheduler_muon = None
    if args.use_muon:
        muon_lr = 5e-4
        muon_params, adamw_params = get_optimizer_groups(model)
        # Muon generally requires a MUCH higher learning rate (often 0.02)
        # than AdamW (3e-4) to take advantage of orthogonal updates.
        opt_muon = Muon(
            muon_params, 
            lr=muon_lr,
            momentum=0.95,                   # SOTA standard for Muon
            adjust_lr_fn='match_rms_adamw',
            weight_decay=0.0,
        ) if muon_params else None
        # Since OneCycleLR only takes one optimizer, we wrap them or use two schedulers
        # Here is the standard way to keep them in sync:
        scheduler_muon = lr_scheduler.OneCycleLR(
            opt_muon, 
            max_lr=muon_lr, 
            total_steps=args.max_steps, 
            pct_start=args.optim.pct_start
        ) if opt_muon else None

    #if int(os.environ.get("RANK", 0)) == 0: ipdb.set_trace(dataset_path)

    # 7. Load optimizer
    if not args.use_muon:
        adamw_params = model.parameters()

    optimizer = AdamW(
        adamw_params,
        lr=args.optim.lr,
        betas=(0.9, 0.95),
        eps=1e-08,
        weight_decay=args.optim.weight_decay,
    ) if adamw_params else None

    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.optim.lr,
        total_steps=args.max_steps,
        pct_start=args.optim.pct_start,
    ) if optimizer else None

    state = TrainState(args.max_steps)

    #if int(os.environ.get("RANK", 0)) == 0: ipdb.set_trace()

    # 8. Initialize checkpointer
    if args.do_ckpt:
        if opt_muon:
            assert False, "need to see how we can save the weights in this case"

        checkpointer = Checkpointer(
            model=model,
            state=state,
            config=lm_config,
            run_dir=run_dir,
            optimizer=optimizer,
            num_ckpt_keep=args.num_ckpt_keep,
            full_finetuning=args.full_finetuning,
        )
    # 9. Prepare mixed precision
    prepare_mixed_precision(
        model.parameters(), param_dtype=param_dtype, optim_dtype=optim_dtype
    )

    # 11. train!
    model.train()
    torch.cuda.empty_cache()

    #if int(os.environ.get("RANK", 0)) == 0: ipdb.set_trace()

    run = None
    if 1 and int(os.environ.get("RANK", 0)) == 0:
        run = Run(experiment='elon_pods100_local')
        run.description = run['description'] = args.description 
        #run.add_tag('gpu=1')

        run['hparams'] = {
            'user': 'ct',
            'learning_rate': args.optim.lr,
            'batch_sz': args.batch_size,
            'duration_sec': args.duration_sec,
            'max_steps': args.max_steps,
            'wt_decay': args.optim.weight_decay,
            'lora_rank': args.lora.rank,
        }

    while state.step < args.max_steps:
        state.start_step()
        is_last_step = state.step == args.max_steps

        if opt_muon:
            opt_muon.zero_grad(set_to_none=True)
        if optimizer:
            optimizer.zero_grad(set_to_none=True)

        loss = torch.tensor([0.0], device="cuda")
        tot_audio_loss = torch.tensor([0.0], device="cuda")
        tot_text_loss = torch.tensor([0.0], device="cuda")
        n_batch_tokens: int = 0
        n_real_tokens: int = 0

        for i in range(args.num_microbatches):
            batch = next(data_loader)
            codes = batch.codes 
            #[BatchSz by 17 by 750], B x n_q x T, these are just LongInt codebook ids

            condition_tensors = None
            if batch.condition_attributes is not None:
                condition_tensors = model.condition_provider.prepare(
                    batch.condition_attributes
                )

            # forward / backward
            output = model(codes=codes, condition_tensors=condition_tensors)
            #class LMOutput:
                # the logits are already re-aligned with the input codes
                # hence no extra shift is required, e.g. when computing ce
                #logits: torch.tensor  # [b, k, t, card]
                #mask: torch.tensor  # [b, k, t]
                #text_logits: torch.tensor  # [b, 1, t, text_card]
                #text_mask: torch.Tensor  # [B, 1, T]


            #if int(os.environ.get("RANK", 0)) == 0: ipdb.set_trace()

            text_loss = compute_loss_with_mask(
                output.text_logits,
                codes[:, : model.audio_offset],
                output.text_mask,
                mode="text",
                text_padding_weight=args.text_padding_weight,
                text_padding_ids={
                    model.text_padding_token_id,
                    model.end_of_text_padding_id,
                },
            )
            audio_loss = compute_loss_with_mask(
                output.logits,
                codes[:, model.audio_offset : model.audio_offset + model.dep_q],
                output.mask,
                mode="audio",
                first_codebook_weight_multiplier=args.first_codebook_weight_multiplier,
            )

            mb_loss = text_loss + audio_loss
            mb_loss.backward()

            loss += mb_loss.detach()
            tot_audio_loss += audio_loss.detach()
            tot_text_loss += text_loss.detach()

            n_batch_tokens += output.text_mask.numel() + output.mask.numel()
            n_real_tokens += (
                torch.sum(output.text_mask).item() + torch.sum(output.mask).item()
            )

            if i < args.num_microbatches - 1:
                # synchronize CUDA to re-run backward
                assert args.num_microbatches > 1  # should not happen
                torch.cuda.synchronize()

        if args.num_microbatches > 1:
            loss /= args.num_microbatches
            tot_audio_loss /= args.num_microbatches
            tot_text_loss /= args.num_microbatches

            for p in model.parameters():
                if p.requires_grad:
                    assert p.grad is not None
                    p.grad.div_(args.num_microbatches)

        # upcast params for optimizer update
        upcast_mixed_precision(model.parameters(), optim_dtype=optim_dtype)

        # clip grad norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)

        # optimizer step
        if opt_muon:
            opt_muon.step()
        if optimizer:
            optimizer.step()

        # downcast params for forward & backward
        downcast_mixed_precision(model.parameters(), param_dtype=param_dtype)

        if scheduler_muon:
            last_lr = scheduler_muon.get_last_lr()[0]
            scheduler_muon.step()

        if scheduler:
            last_lr = scheduler.get_last_lr()[0]
            scheduler.step()

        #if int(os.environ.get("RANK", 0)) == 0:
        #    ipdb.set_trace()

        # Host sync
        loss_item = loss.item()
        avg_loss = avg_aggregate(loss_item)

        if args.do_eval and (
            (args.eval_freq > 0 and state.step % args.eval_freq == 0) or is_last_step
        ):

            #if int(os.environ.get("RANK", 0)) == 0: ipdb.set_trace()

            # write perplexity to state
            #evaluate(model, eval_data_loader, state, args)
            evaluate(model, interleaved_tokenizer, state, args, total_eval_samples=100 )

            if run:
                run.track(state.this_eval_loss, name='eval_loss', step=state.step, context={ "subset":"eval" })
                run.track(state.this_audio_loss, name='eval_audio_loss', step=state.step, context={ "subset":"eval" })
                run.track(state.this_text_loss, name='eval_text_loss', step=state.step, context={ "subset":"eval" })

            eval_logs = get_eval_logs(
                state.step,
                avg_loss,
                state.this_eval_perplexity,
                state.this_eval_loss,
            )

            main_logger_info(eval_log_msg(eval_logs))
            eval_logger.log(eval_logs, step=state.step)

            #if int(os.environ.get("RANK", 0)) == 0: ipdb.set_trace()

        # Timing
        state.end_step(n_batch_tokens)

        if state.step % args.log_freq == 0:

            if run:
                run.track(avg_loss, name='train_loss', step=state.step, context={ "subset":"train" })
                run.track(tot_audio_loss.item(), name='train_audio_loss', step=state.step, context={ "subset":"train" })
                run.track(tot_text_loss.item(), name='train_text_loss', step=state.step, context={ "subset":"train" })

            train_logs = get_train_logs(
                state,
                avg_loss,
                n_real_tokens,
                last_lr,
                torch.cuda.max_memory_allocated(),
                torch.cuda.memory_allocated(),
                args,
            )
            logging.info(f'Training has seen {len(dataset.UniqueWavPaths)} audio files!')

            main_logger_info(train_log_msg(state, logs=train_logs, loss=avg_loss))
            metrics_logger.log(train_logs, step=state.step)

        if args.do_ckpt and (
            (args.ckpt_freq > 0 and state.step % args.ckpt_freq == 0) or is_last_step
        ):
            checkpointer.save_checkpoint(
                save_only_lora=not args.full_finetuning and args.save_adapters,
                dtype=param_dtype,
            )

    main_logger_info("done!")


if __name__ == "__main__":
    """See README.md for usage."""
    fire.Fire(train)
