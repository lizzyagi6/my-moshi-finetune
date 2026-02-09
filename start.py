from pathlib import Path

from huggingface_hub import snapshot_download

# these info is needed for training
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def download():
  Path("/content/data/daily-talk-contiguous").mkdir(parents=True, exist_ok=True)

  # Replace with your actual token from hf.co/settings/tokens
  # os.environ["HF_TOKEN"] = "hf_your_actual_token_here"
  # Download the dataset
  local_dir = snapshot_download(
      "kyutai/DailyTalkContiguous",
      repo_type="dataset",
      local_dir="/content/data/daily-talk-contiguous",
  )

def setup_yaml():
  # define training configuration
  # for your own use cases, you might want to change the data paths, model path, run_dir, and other hyperparameters
  import yaml
  
  config = """
  # data
  data:
    train_data: '/content/data/daily-talk-contiguous/dailytalk.jsonl' # Fill
    eval_data: '' # Optionally Fill
    shuffle: true
  
  # model
  moshi_paths:
    hf_repo_id: "kyutai/moshiko-pytorch-bf16"
  
  
  full_finetuning: false # Activate lora.enable if partial finetuning
  lora:
    enable: true
    rank: 128
    scaling: 2.
    ft_embed: false
  
  # training hyperparameters
  first_codebook_weight_multiplier: 100.
  text_padding_weight: .5
  
  
  # tokens per training steps = batch_size x num_GPUs x duration_sec
  # we recommend a sequence duration of 300 seconds
  # If you run into memory error, you can try reduce the sequence length
  duration_sec: 100
  batch_size: 1
  max_steps: 300
  
  gradient_checkpointing: true # Activate checkpointing of layers
  
  # optim
  optim:
    lr: 2.e-6
    weight_decay: 0.1
    pct_start: 0.05
  
  # other
  seed: 0
  log_freq: 10
  eval_freq: 1
  do_eval: False
  ckpt_freq: 10
  
  save_adapters: True
  
  run_dir: "/content/test"  # Fill
  """
  
  # save the same file locally into the example.yaml file
  with open("/content/example.yaml", "w") as file:
      yaml.dump(yaml.safe_load(config), file)


if __name__ == '__main__':

  download()
  setup()
  # make sure the run_dir has not been created before
  # only run this when you ran torchrun previously and created the /content/test_ultra file
  # ! rm -r /content/test







