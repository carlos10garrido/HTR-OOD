#!/bin/bash
# Schedule execution of many runs

dataset=$1
# Loop to iterate over masking noises values (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
# Iterate over values from 0.1 to 1.0 with step 0.2
for i in $(seq 0.0 0.1 1.0)
# Reverse loop
# for i in $(seq 1.0 -0.1 0.0)
do # Run from root folder with: bash scripts/schedule.sh
# Check if the folder exists
  if [ ! -d "results/Transformer-$dataset-CONV-PATCH-COLUMN-$i%masking-NOISE" ]; then
      # Create folder
      mkdir results/Transformer-$dataset-CONV-PATCH-COLUMN-$i%masking-NOISE
      
      # Run the experiment
      python src/train_transformer.py \
      data/train/train_config/datasets=[$dataset] \
      data/val/val_config/datasets=[$dataset] \
      data.train.train_config.img_size=[64,512] \
      data.train.train_config.batch_size=16 \
      logger.wandb.name=Transformer-$dataset-CONV-PATCH-COLUMN-$i%masking-NOISE \
      logger.wandb.tags=[masking_exp] \
      logger.wandb.offline=False \
      trainer.max_epochs=200 \
      model=transformer_big \
      model.net.use_backbone=True \
      model.net.patch_per_column=True \
      model.net.masking_noise=$i \
      callbacks.early_stopping.patience=50 
  fi
done

# python src/train_transformer.py \
#   data/train/train_config/datasets=[washington] \
#   data/val/val_config/datasets=[washington] \
#   logger.wandb.name=Transformer-washington-NO-CONV-PATCH-COLUMN-60%masking-NOISE \
#   trainer.max_epochs=200 \
#   model=transformer_big \
#   logger.wandb.tags=[masking_exp] \
#   data.train.train_config.img_size=[64,512] \
#   data.train.train_config.batch_size=16 \
#   model.net.use_backbone=True \
#   model.net.patch_per_column=True \
#   callbacks.early_stopping.patience=500 \
#   logger.wandb.offline=False model.net.masking_noise=0.6