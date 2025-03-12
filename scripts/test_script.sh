# #!/bin/bash

# Schedule execution of many runs
python_module=$1 # e.g. src/train_seq2seq.py
model_name=$2
dataset_src=$3
img_size=$4
batch_size=$5

# Get all the checkpoints containing that model inside checkpoints folder
echo "Inside folder $(pwd)"
echo "Searching for checkpoints containing $model_name"
echo "Model name: $model_name"
echo "image size: $img_size"

checkpoints=$(ls checkpoints/ | grep "\b$model_name" | grep src_$dataset_src)

echo "Number of checkpoints found: $(echo $checkpoints | wc -w)"
echo "Checkpoints found: $checkpoints"

for checkpoint in $checkpoints
do
  # Remove .ckpt extension
  model_name_without_ckpt=$(echo $checkpoint | sed 's/.ckpt//')
  echo "Checkpoint: $checkpoint"
  # Test for each src ID
  eval="python $python_module \
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
      data.train.train_config.num_workers=16 \
      train=False"
  echo $eval
  eval $eval
done