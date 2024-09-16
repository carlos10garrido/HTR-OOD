# #!/bin/bash

#!/bin/bash
# Schedule execution of many runs

dataset_list="iam rimes washington saint_gall bentham rodrigo icfhr_2016"
lr_list="0.01 0.001 0.0001 0.00001"
img_size_list="64,1024 64,512 128,1024 32,1024 32,512"
batch_size_list="16 32 64 128"

# Train CRNN with synth data
# Iterate over learning rate
# python src/train_crnn_ctc.py \ 
#   data/train/train_config/datasets=[wikitext] \ 
#   data/val/val_config/datasets=[iam,rimes,washington,saint_gall,bentham,rodrigo,icfhr_2016] \ 
#   data/test/test_config/datasets=[iam] \ 
#   data.train.train_config.img_size=[128,1024] \ 
#   data.train.train_config.batch_size=16 \ 
#   logger.wandb.name=Transformer_IAM \ 
#   trainer.max_epochs=100 \ 
#   model=transformer_big \ 
#   callbacks.early_stopping.patience=50 \ 
#   model.optimizer.lr=0.0001 \ 
#   callbacks.model_checkpoint_base.filename=transformer_big_src_wikitext \ 
#   trainer.deterministic=False \ 
#   callbacks.model_checkpoint_id.filename="${callbacks.model_checkpoint_base.filename}_ID" \ 
#   +trainer.limit_train_batches=1.0 \ 
#   +trainer.limit_val_batches=1.0 \ 
#   callbacks/heldout_targets=[iam,rimes,washington,saint_gall,bentham,rodrigo,icfhr_2016] \ 
#   logger.wandb.offline=False




# Train CRNN with real data
# Iterate over all datasets from the list, setting the data/train/train_config/datasets to one and callbacks/heldout_targets to the rest that are not the training. Set the validation always to all of them.
# Iterate over learning rate
# Iterate over image size
# Iterate over batch size
# rest_datasets=$(echo $dataset_list | sed "s/$dataset//"
for lr in $lr_list
do
  for img_size in $img_size_list
  do
    for batch_size in $batch_size_list
    do
      call_script="python src/train_crnn_ctc.py \
      data/train/train_config/datasets=[wikitext] \ 
      data/val/val_config/datasets=[iam,rimes,washington,saint_gall,bentham,rodrigo,icfhr_2016] \
      data/test/test_config/datasets=[$dataset] \ 
      data.train.train_config.img_size=[$img_size] \ 
      data.train.train_config.batch_size=$batch_size \ 
      logger.wandb.name=CRNN-puig-src-$dataset-$img_size-$batch_size-$lr \
      trainer.max_epochs=200 \ 
      model=crnn_puig \ 
      callbacks.early_stopping.patience=50 \ 
      model.optimizer.lr=$lr \ 
      callbacks.model_checkpoint_base.filename=CRNN_puig_src_wikitext-$img_size-$batch_size-$lr  \ 
      trainer.deterministic=False \ 
      callbacks.model_checkpoint_id.filename="\${callbacks.model_checkpoint_base.filename}_ID" \ 
      +trainer.limit_train_batches=1.0 \ 
      +trainer.limit_val_batches=1.0 \ 
      callbacks/heldout_targets=[iam,rimes,washington,saint_gall,bentham,rodrigo,icfhr_2016] \
      logger.wandb.offline=False"
      echo $call_script
      # eval $call_script
    done
  done
done


# Launch with wikitext for different datasets
# launch_var="python src/train_crnn_ctc.py data/train/train_config/datasets=[wikitext] data/val/val_config/datasets=[iam,rimes,washington,saint_gall,bentham,rodrigo,icfhr_2016] data/test/test_config/datasets=[iam] data.train.train_config.img_size=[128,1024] data.train.train_config.batch_size=16 logger.wandb.name=Transformer_IAM trainer.max_epochs=300 model=crnn_puig callbacks.early_stopping.patience=50 model.optimizer.lr=0.0001 callbacks.model_checkpoint_base.filename=transformer_big_src_wikitext trainer.deterministic=True callbacks.model_checkpoint_id.filename="\${callbacks.model_checkpoint_base.filename}_ID" +trainer.limit_train_batches=1.0 +trainer.limit_val_batches=1.0 callbacks/heldout_targets=[iam,rimes,washington,saint_gall,bentham,rodrigo,icfhr_2016] logger.wandb.offline=False"
# echo $launch_var
# # eval $launch_var