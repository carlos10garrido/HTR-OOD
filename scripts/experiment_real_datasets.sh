# #!/bin/bash

# Schedule execution of many runs

dataset_list="iam rimes washington saint_gall bentham rodrigo icfhr_2016"
# Train CRNN with real data
list_dataset=$(echo $dataset_list | sed "s/ /,/g")

for src_dataset in $dataset_list
do
  # Calculate heldout targets adding commas to the list of datasets
  heldout_targets=$(echo $dataset_list | sed "s/$src_dataset//")
  heldout_targets=$(echo $heldout_targets | sed "s/ /,/g")

  # Change epochs to 300 and offline to True
  eval="python src/train_crnn_ctc.py 
    data/train/train_config/datasets=[$src_dataset] \
    data/val/val_config/datasets=[$list_dataset] \
    data/test/test_config/datasets=[$list_dataset] \
    data.train.train_config.img_size=[128,1024] \
    data.train.train_config.batch_size=16 \
    trainer.max_epochs=500 \
    model=crnn_puig \
    callbacks.early_stopping.patience=100 \
    model.optimizer.lr=0.0003 \
    callbacks.model_checkpoint_base.filename=crnn_puig_src_${src_dataset} \
    trainer.deterministic=False \
    callbacks.model_checkpoint_id.filename="\\\${callbacks.model_checkpoint_base.filename}_ID" \
    tokenizer=tokenizers/char_tokenizer \
    callbacks/heldout_targets=[$heldout_targets] \
    callbacks/optim_targets=[$list_dataset] \
    logger.wandb.offline=False \
    logger.wandb.name=crnn_puig_src_${src_dataset} \
    data.train.train_config.binarize=True \
    model.log_val_metrics=False \
    train=True"
  echo $eval
  echo ""
  # eval $eval

  # Test for each heldout as target
  # for tgt_dataset in $dataset_list
  # do
  #   if [ $src_dataset != $tgt_dataset ]
  #   then
  #     eval="python src/train_crnn_ctc.py 
  #       data/train/train_config/datasets=[$src_dataset] \
  #       data/val/val_config/datasets=[$tgt_dataset] \
  #       data/test/test_config/datasets=[iam,rimes,washington,saint_gall,bentham,rodrigo,icfhr_2016] \
  #       data.train.train_config.img_size=[128,1024] \
  #       data.train.train_config.batch_size=16 \
  #       trainer.max_epochs=1 \
  #       model=crnn_puig \
  #       callbacks.early_stopping.patience=300 \
  #       model.optimizer.lr=0.0003 \
  #       callbacks.model_checkpoint_base.filename=crnn_puig_src_${src_dataset}_tgt_${tgt_dataset} \
  #       paths.checkpoints_dir=checkpoints_to_test/ \
  #       trainer.deterministic=False \
  #       callbacks.model_checkpoint_id.filename="\\\${callbacks.model_checkpoint_base.filename}_test_delete" \
  #       tokenizer=tokenizers/char_tokenizer \
  #       callbacks/heldout_targets=[$tgt_dataset] \
  #       logger.wandb.offline=False \
  #       logger.wandb.name=CRNN_Puig_src_${src_dataset}_tgt_${tgt_dataset} \
  #       data.train.train_config.binarize=True \
  #       train=False"
  #     echo $eval
  #     eval $eval
  #   fi
  # done

  # # Test for each src ID
  # # for tgt_dataset in $dataset_list
  # # do
  # #   if [ $src_dataset != $tgt_dataset ]
  # #   then
  #     eval="python src/train_crnn_ctc.py 
  #       data/train/train_config/datasets=[$src_dataset] \
  #       data/val/val_config/datasets=[$tgt_dataset] \
  #       data/test/test_config/datasets=[iam,rimes,washington,saint_gall,bentham,rodrigo,icfhr_2016] \
  #       data.train.train_config.img_size=[128,1024] \
  #       data.train.train_config.batch_size=16 \
  #       model=crnn_puig \
  #       callbacks.model_checkpoint_base.filename=crnn_puig_src_${src_dataset}_ID \
  #       paths.checkpoints_dir=checkpoints_to_test/ \
  #       trainer.deterministic=False \
  #       callbacks.model_checkpoint_id.filename="\\\${callbacks.model_checkpoint_base.filename}_test_delete" \
  #       tokenizer=tokenizers/char_tokenizer \
  #       callbacks/heldout_targets=[$tgt_dataset] \
  #       logger.wandb.offline=False \
  #       logger.wandb.name=CRNN_Puig_src_${src_dataset}_ID \
  #       data.train.train_config.binarize=True \
  #       train=False"
  #     echo $eval
  #     eval $eval
  # #   fi
  # # done


done