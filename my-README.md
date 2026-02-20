# Moshi Fine-tuning Guide

This repository contains scripts and instructions for setting up and running the Moshi fine-tuning environment on a Linux server.

## 🚀 Training Steps

Follow these steps to initialize your environment and launch the server:

```bash
# 1. Clone the repository
git clone https://github.com
cd my-moshi-finetune

# 2. Add a .env file (Fill in your actual credentials)
cat <<EOF > .env
USER_NAME=john
R2_END_POINT_URL=your_r2_endpoint_url
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
EOF

# 3. Setup the environment
sh steps.sh setup

# 4. Sync data
sh steps.sh sync-down

# 5. Create and enter the model directory
mkdir -p ~/model
cd ~/model

# 6. Download experiment weights using uv and AWS CLI
uv run aws s3 cp s3://duplexdata/experiments/elonf ./ \
    --endpoint-url $R2_END_POINT_URL \
    --region auto \
    --recursive

# 7. Run the Moshi server
uv run python -m moshi.server \
    --lora-weight=consolidated/lora.safetensors \
    --config-path=consolidated/config.json
