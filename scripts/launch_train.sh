# #!/bin/bash

# Schedule execution of many runs

dataset_list="iam rimes washington saint_gall bentham rodrigo icfhr_2016"
model_pretrained=$1
# Train CRNN with real data

optim_targets=$(echo $dataset_list | sed "s/ /,/g")

for src_dataset in $dataset_list
do
  # Calculate heldout targets adding commas to the list of datasets
  heldout_targets=$(echo $dataset_list | sed "s/$src_dataset//")
  heldout_targets=$(echo $heldout_targets | sed "s/ /,/g")

  # Change epochs to 300 and offline to True
  eval="python src/train_crnn_ctc.py 
    data/train/train_config/datasets=[$src_dataset] \
    data.train.train_config.img_size=[128,1024] \
    data.train.train_config.batch_size=16 \
    trainer.max_epochs=500 \
    model=crnn_puig \
    callbacks.early_stopping.patience=100 \
    callbacks.model_checkpoint_base.filename=${model_pretrained}_src_${src_dataset} \
    +pretrained_checkpoint=${model_pretrained} \
    trainer.deterministic=False \
    callbacks.model_checkpoint_id.filename="\\\${callbacks.model_checkpoint_base.filename}_ID" \
    tokenizer=tokenizers/char_tokenizer \
    callbacks/heldout_targets=[$heldout_targets] \
    callbacks/optim_targets=[$optim_targets] \
    logger.wandb.offline=False \
    logger.wandb.name=${model_pretrained}_src_${src_dataset} \
    data.train.train_config.binarize=True \
    train=True"
  echo $eval
  echo ""
  # echo ""
done