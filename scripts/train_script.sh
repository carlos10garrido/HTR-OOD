# #!/bin/bash

# Schedule execution of many runs
model_name=$1
img_size=$2
script_name=$3 # e.g. src/train_seq2seq.py

# Get all the checkpoints containing that model inside checkpoints folder
echo "Inside folder $(pwd)"
echo "Searching for checkpoints containing $model_name"
echo "Model name: $model_name"
echo "image size: $img_size"

checkpoints=$(ls checkpoints/ | grep $model_name)

echo "Number of checkpoints found: $(echo $checkpoints | wc -w)"

for checkpoint in $checkpoints
do
  # Remove .ckpt extension
  model_name_without_ckpt=$(echo $checkpoint | sed 's/.ckpt//')
  echo "Checkpoint: $checkpoint"
  # Test for each src ID
  eval="python src/$script_name.py \
      data/train/train_config/datasets=[iam] \
      data.train.train_config.img_size=$img_size \
      data.train.train_config.batch_size=16 \
      model=$model_name \
      callbacks.model_checkpoint_base.filename=$model_name_without_ckpt \
      paths.checkpoints_dir=checkpoints/ \
      callbacks.model_checkpoint_id.filename="\\\${callbacks.model_checkpoint_base.filename}" \
      tokenizer=tokenizers/char_tokenizer \
      callbacks/heldout_targets=[] \
      logger.wandb.offline=False \
      logger.wandb.name=${model_name_without_ckpt}_test \
      data.train.train_config.binarize=True \
      train=True"
  echo $eval
  # eval $eval
done