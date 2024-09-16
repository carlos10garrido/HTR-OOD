#!/bin/bash

prefix_run=$1
dataset=$2


# Set parameters
img_heights=(32 40 64 128)
img_widths=(128 256 512)
batch_size=(32)
n_heads=(2 4 8)
n_layers=(2 4 8)
hidden_dims=(128 256 512 1024)
mlp_dims=(128 256 512 1024)
dropout=(0.2 0.3 0.4 0.5)

# Set random seed
RANDOM=$$$(date +%s)

# Function to select a random element from an array
function select_random {
  local arr=("$@")
  local random_index=$((RANDOM % ${#arr[@]}))
  echo "${arr[random_index]}"
}

# Iterate over the parameters

while true; do
  # Select random parameters
  img_height=$(select_random "${img_heights[@]}")
  img_width=$(select_random "${img_widths[@]}")
  batch_size=$(select_random "${batch_size[@]}")
  n_heads=$(select_random "${n_heads[@]}")
  n_layers=$(select_random "${n_layers[@]}")
  hidden_dims=$(select_random "${hidden_dims[@]}")
  mlp_dims=$(select_random "${mlp_dims[@]}")
  dropout=$(select_random "${dropout[@]}")

  run_name="${prefix_run}-${dataset}-img_height_${img_height}_img_width_${img_width}_n_heads_${n_heads}_n_layers_${n_layers}_hidden_dims_${hidden_dims}_mlp_dims_${mlp_dims}_dropout_${dropout}"

  if [ -d "runs_dummy/$run_name" ]; then
    echo "Skipping $run_name"
    continue
  fi

  mkdir -p "runs_dummy/$run_name"


  # Run experiment
  python src/train_transformer.py \
  paths.log_dir=data/logs \
  trainer.check_val_every_n_epoch=1 \
  trainer.num_sanity_val_steps=1 \
  data/train/train_config/datasets=[${dataset}] \
  data/val/val_config/datasets=[iam,washington,saint_gall,esposalles] \
  data/test/test_config/datasets=[iam,washington,saint_gall,esposalles] \
  data.train.train_config.img_size=[$img_height,$img_width] \
  data.train.train_config.batch_size=$batch_size \
  logger.wandb.name=${run_name} \
  logger.wandb.tags=[random_search_vit_col] \
  logger.wandb.offline=False \
  trainer.max_epochs=200 \
  model=transformer_big \
  model.net.use_backbone=True \
  model.net.patch_per_column=True \
  model.net.encoder_layers=$n_layers \
  model.net.decoder_layers=$n_layers \
  model.net.encoder_attention_heads=$n_heads \
  model.net.decoder_attention_heads=$n_heads \
  model.net.d_model=$hidden_dims \
  model.net.encoder_ffn_dim=$mlp_dims \
  model.net.decoder_ffn_dim=$mlp_dims \
  model.net.dropout=$dropout \
  callbacks.early_stopping.patience=50
done

# python src/train_transformer.py \
# data/train/train_config/datasets=[washington] \
# data/val/val_config/datasets=[iam,washington,saint_gall,esposalles] \
# data/test/test_config/datasets=[iam,washington,saint_gall,esposalles] \
# data.train.train_config.img_size=[64,256] \
# data.train.train_config.batch_size=64 \
# logger.wandb.name=ViT-Column-Random-Search-Washington \
# logger.wandb.tags=[random_search_vit_col] \
# logger.wandb.offline=False \
# trainer.max_epochs=200 \
# model=transformer_big \
# model.net.use_backbone=True \
# model.net.patch_per_column=True \
# callbacks.early_stopping.patience=50\