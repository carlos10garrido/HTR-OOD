# #!/bin/bash

# Schedule execution of many runs
model_name=$1
dataset_src=$2
img_size=$3
batch_size=$4


# Get all the checkpoints containing that model inside checkpoints folder
echo "Inside folder $(pwd)"
echo "Searching for checkpoints containing $model_name"
echo "Model name: $model_name"
echo "image size: $img_size"

checkpoints=$(ls checkpoints/ | grep $model_name | grep src_$dataset_src)

echo "Number of checkpoints found: $(echo $checkpoints | wc -w)"

# echo "Checkpoints found: $checkpoints"


for checkpoint in $checkpoints
do
  # Remove .ckpt extension
  model_name_without_ckpt=$(echo $checkpoint | sed 's/.ckpt//')
  echo "Checkpoint: $checkpoint"
  # Test for each src ID
  eval="python src/train_crnn_ctc.py 
      data/train/train_config/datasets=[$dataset_src] \
      data.train.train_config.img_size=$img_size \
      data.train.train_config.batch_size=$batch_size \
      model=$model_name \
      callbacks.model_checkpoint_base.filename=$model_name_without_ckpt \
      paths.checkpoints_dir=checkpoints/ \
      callbacks.model_checkpoint_id.filename="\\\${callbacks.model_checkpoint_base.filename}" \
      +pretrained_checkpoint=$model_name_without_ckpt \
      tokenizer=tokenizers/char_tokenizer \
      callbacks/heldout_targets=[] \
      logger.wandb.offline=False \
      logger.wandb.name=${model_name_without_ckpt}_test \
      data.train.train_config.binarize=True \
      train=False"
  echo $eval
  eval $eval
done