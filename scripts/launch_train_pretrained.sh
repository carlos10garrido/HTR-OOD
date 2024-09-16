# #!/bin/bash

# Schedule execution of many runs

dataset_list="iam rimes washington saint_gall bentham rodrigo icfhr_2016"
model_pretrained=$1
# Train CRNN with real data

for src_dataset in $dataset_list
do
  # Calculate heldout targets adding commas to the list of datasets
  heldout_targets=$(echo $dataset_list | sed "s/$src_dataset//")
  heldout_targets=$(echo $heldout_targets | sed "s/ /,/g")

  # Change epochs to 300 and offline to True
  eval="python src/train_crnn_ctc.py 
    data/train/train_config/datasets=[$src_dataset] \
    data/val/val_config/datasets=[iam,rimes,washington,saint_gall,bentham,rodrigo,icfhr_2016] \
    data/test/test_config/datasets=[iam,rimes,washington,saint_gall,bentham,rodrigo,icfhr_2016] \
    data.train.train_config.img_size=[128,1024] \
    data.train.train_config.batch_size=16 \
    trainer.max_epochs=1 \
    model=crnn_puig \
    callbacks.early_stopping.patience=300 \
    model.optimizer.lr=0.0003 \
    callbacks.model_checkpoint_base.filename=crnn_puig_src_${src_dataset} \
    trainer.deterministic=False \
    callbacks.model_checkpoint_id.filename="\\\${callbacks.model_checkpoint_base.filename}_ID" \
    tokenizer=tokenizers/char_tokenizer \
    callbacks/heldout_targets=[$heldout_targets] \
    callbacks/optim_targets=$dataset_list \
    logger.wandb.offline=False \
    logger.wandb.name=CRNN_Puig_src_${src_dataset} \
    data.train.train_config.binarize=True \
    train=True"
  echo $eval
  echo ""
  # echo ""
done