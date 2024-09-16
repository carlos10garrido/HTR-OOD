# #!/bin/bash

# Schedule execution of many runs

# dataset_list="iam rimes washington saint_gall bentham rodrigo icfhr_2016"
dataset_source="saint_gall"
dataset_list="iam rimes washington saint_gall bentham rodrigo icfhr_2016"
# Train CRNN with real data

for src_dataset in $dataset_source
do
  # Calculate heldout targets adding commas to the list of datasets
  heldout_targets=$(echo $dataset_list | sed "s/$src_dataset//")
  heldout_targets=$(echo $heldout_targets | sed "s/ /,/g")

  # Test for each src ID
  eval="python src/train_crnn_ctc.py 
      data/train/train_config/datasets=[$src_dataset] \
      data/test/test_config/datasets=[iam,rimes,washington,saint_gall,bentham,rodrigo,icfhr_2016] \
      data.train.train_config.img_size=[128,1024] \
      data.train.train_config.batch_size=16 \
      model=crnn_puig \
      callbacks.model_checkpoint_base.filename=crnn_puig_no_bin_src_${src_dataset}_ID \
      paths.checkpoints_dir=checkpoints_to_test/ \
      trainer.deterministic=False \
      callbacks.model_checkpoint_id.filename="\\\${callbacks.model_checkpoint_base.filename}_test_delete" \
      tokenizer=tokenizers/char_tokenizer \
      callbacks/heldout_targets=[] \
      logger.wandb.offline=False \
      logger.wandb.name=CRNN_Puig_no_bin_src_${src_dataset}_ID \
      data.train.train_config.binarize=True \
      train=False"
    echo $eval
    eval $eval

  # Test for each heldout as target
  for tgt_dataset in $dataset_list
  do
    if [ $src_dataset != $tgt_dataset ]
    then
      eval="python src/train_crnn_ctc.py 
        data/train/train_config/datasets=[$src_dataset] \
        data/test/test_config/datasets=[iam,rimes,washington,saint_gall,bentham,rodrigo,icfhr_2016] \
        data.train.train_config.img_size=[128,1024] \
        data.train.train_config.batch_size=16 \
        trainer.max_epochs=1 \
        model=crnn_puig \
        callbacks.early_stopping.patience=300 \
        model.optimizer.lr=0.0003 \
        callbacks.model_checkpoint_base.filename=crnn_puig_no_bin_src_${src_dataset}_tgt_${tgt_dataset} \
        paths.checkpoints_dir=checkpoints_to_test/ \
        trainer.deterministic=False \
        callbacks.model_checkpoint_id.filename="\\\${callbacks.model_checkpoint_base.filename}_test_delete" \
        tokenizer=tokenizers/char_tokenizer \
        callbacks/heldout_targets=[$tgt_dataset] \
        logger.wandb.offline=False \
        logger.wandb.name=CRNN_Puig_no_bin_src_${src_dataset}_tgt_${tgt_dataset} \
        data.train.train_config.binarize=True \
        train=False"
      echo $eval
      eval $eval
    fi
  done
done