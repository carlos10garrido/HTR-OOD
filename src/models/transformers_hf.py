from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MinMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.text import CharErrorRate as CER
from src.utils import pylogger
from src.utils.logger import MetricLogger
log = pylogger.RankedLogger(__name__, rank_zero_only=True)
import torchvision
import os
import src
# import global variables encode and decode from htr_data_module.py
# from src.data.htr_datamodule import encode, decode

import numpy as np
# import cv2
from PIL import Image
import wandb
import matplotlib
cmap = matplotlib.cm.get_cmap('jet')
cmap.set_bad(color="k", alpha=0.0)

# PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 1, 0, 2, 3

class HTRTransformerLitModule(LightningModule):
  
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        _logger: Any,
        datasets: dict,
        visualize_attention: bool = False,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.visualize_attention = visualize_attention

        # Save datasets names in a list to index from validation_step
        self.train_datasets = list(datasets['train']['train_config']['datasets'].keys())
        self.val_datasets = list(datasets['val']['val_config']['datasets'].keys())
        self.test_datasets = list(datasets['test']['test_config']['datasets'].keys())

        print(f'self.train_datasets: {self.train_datasets}')
        print(f'self.val_datasets: {self.val_datasets}')
        print(f'self.test_datasets: {self.test_datasets}')

        # breakpoint()

        # Create metrics per dataset
        # Training metrics
        self.train_loss = MeanMetric()
        # Validation metrics
        self.val_cer = {}
        self.min_val_cer = {}
        for dataset in self.val_datasets:
            self.val_cer[dataset] = CER()
            self.min_val_cer[dataset] = MinMetric()

        # Test metrics
        self.test_cer = {}
        self.min_test_cer = {}
        for dataset in self.test_datasets:
            self.test_cer[dataset] = CER()
            self.min_test_cer[dataset] = MinMetric()

        self.net = net

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=1) # 1 is the padding token

        # self.encode = encode
        # self.decode = decode

        self.encode = src.data.htr_datamodule.encode
        self.decode = src.data.htr_datamodule.decode

        # metric objects for calculating and averaging accuracy across batches
        log.info(f'Logger in HTRTransformerLitModule: {_logger}. Keys: {list(_logger.keys())}')
        self._logger = _logger[list(_logger.keys())[0]]

        # Log train datasets, val datasets and test datasets
        self._logger.log_hyperparams({
          'train_datasets': self.train_datasets,
          'val_datasets': self.val_datasets,
          'test_datasets': self.test_datasets,
        })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        # print(f'x: {x}')
        # print(f'x.shape: {x.shape}')
        return self.net(x)

    def on_fit_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        # self.val_loss.reset()
        # Reset metrics for each dataset

        self.metric_logger = MetricLogger(
          logger=self._logger,
          train_datasets=self.train_datasets,
          val_datasets=self.val_datasets,
          test_datasets=self.test_datasets,
        )


    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        
        """
        images = batch[0]
        if self.current_epoch == 0 and self.global_step <= 1:
          str_train_datasets = f'train_' + ', '.join(self.train_datasets)
          self.metric_logger.log_images(images, str_train_datasets)

        labels = batch[1].permute(1, 0)
        labels[labels == 1] = -100
        labels = labels[:, 1:].clone().contiguous() # Shift all labels to the right
        labels = labels[:, :-1].clone().contiguous() # Remove last label (it is the <eos> token)

        
        # outputs = self.net.forward(images=batch[0], labels=labels)
        outputs = self.net(images=images, labels=labels)

        print(f'Labels[:2]: {labels[:2]}')
        print(f'outputs[:2]: {outputs.logits[:2].argmax(-1)}')


        logits = outputs.logits

        # loss = self.criterion(logits.view(-1, logits.shape[-1]), labels.view(-1))
        loss = outputs.loss
        # print(f'loss: {loss}')
        acc = (logits.argmax(dim=-1) == labels).sum() / (labels != 1).sum() # 1 is the padding token
        self.metric_logger.log_train_step(loss, acc)

        # labels[labels == -100] = 1

        if batch_idx < 10:
          for i in range(images.shape[0]):
            # images_ = self.metric_logger.log_images(images[i], f'train/training_images_{self.train_datasets[0]}')
            images_ = images[i]
            _label = labels[i].detach().cpu().numpy().tolist()
            _label = [label if label != -100 else 1 for label in _label]
            _pred = logits[i].argmax(-1).detach().cpu().numpy().tolist()
            _label, _pred = self.decode(_label), self.decode(_pred)
            _label_, _pred_ = "", ""
            for l in range(len(_label)):
              if _label[l] == '<eos>' or _label[l] == '<sos>' or _label[l] == '<pad>':
                break
              _label_ += _label[l]

            for l in range(len(_pred)):
              if _pred[l] == '<eos>' or _pred[l] == '<sos>' or _pred[l] == '<pad>':
                break
              _pred_ += _pred[l]
              
            _label, _pred = _label_, _pred_
            print(f'Label: {_label}. Pred: {_pred}')
            # self.metric_logger.log_train_step_cer(_pred, _label, self.train_datasets[0])
            cer = CER()(_pred, _label)
            self._logger.experiment.log({f'train/preds_{self.train_datasets[0]}': wandb.Image(images_, caption=f'Label: {_label} \n Pred: {_pred} \n CER: {cer} \n epoch: {self.current_epoch}')})

        # update and log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss 

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        self.metric_logger.log_train_metrics()
        self.metric_logger.update_epoch(self.current_epoch)

        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, dataloader_idx: int = None) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        dataloader_idx = 0 if len(self.val_datasets) == 1 else dataloader_idx
        dataset = self.val_datasets[dataloader_idx]
        # print(f'Calculating validation CER for each dataset for dataloader_idx: {dataloader_idx}. Dataset: {dataset}')

        # Get epoch
        epoch = self.current_epoch

        images, labels = batch[0], batch[1]
        print(f'images.shape: {images.shape}')
        labels = labels.permute(1, 0)
        labels = labels[:, 1:].clone().contiguous() # Shift all labels to the right
        # labels = labels[:, :-1].clone().contiguous() # Remove last label (it is the <eos> token)

        

        if self.current_epoch == 0 and self.global_step <= 1:
          str_train_datasets = f'val_' + ', '.join(self.train_datasets)
          self.metric_logger.log_images(images, str_train_datasets)
        
        preds = self.net.predict_greedy(images)
        # print(f'preds[:10]: {preds.sequences[:10]}')
        print(f'preds[:10]: {preds[:10]}')

        preds_str, labels_str = [], []
        for i in range(images.shape[0]):
          images_ = self.metric_logger.log_images(images[i], f'val/validation_images_{dataset}')
          _label = labels[i].detach().cpu().numpy().tolist()
          # _pred = preds.sequences[i].tolist()
          _pred = preds[i].tolist()

          _label, _pred = self.decode(_label), self.decode(_pred)

          _label_, _pred_ = "", ""
          for l in range(len(_label)):
            if _label[l] == '<eos>' or _label[l] == '<pad>':
              break
            _label_ += _label[l]

          for l in range(len(_pred)):
            if _pred[l] == '<eos>' or _pred[l] == '<pad>':
              break
            _pred_ += _pred[l]

          # Remove <sos> token from pred
          _pred = _pred_.replace('<sos>', '')
            
          _label, _pred = _label_, _pred_
          
          self.metric_logger.log_val_step_cer(_pred, _label, dataset)
          
          if batch_idx < 20:
            # print(f'Label: {_label}. Pred: {_pred}')
            print(f'VAL Label: {_label}. Pred: {_pred}')
            cer = CER()(_pred, _label)
            self._logger.experiment.log({f'val/preds_{dataset}': wandb.Image(images[i], caption=f'Label: {_label} \n Pred: {_pred} \n CER: {cer} \n epoch: {self.current_epoch}')})


        self.metric_logger.log_val_step_cer(preds_str, labels_str, dataset)

        # Calculate CER
        self.val_cer[dataset](preds_str, labels_str)
        # self.log(f"val/cer_{dataset}", step_cer, on_step=True, on_epoch=True, prog_bar=True)

        # update and log metrics
        # self.val_loss(loss)
        # self.val_acc(preds, targets)
        # self.log("val/cer", step_cer, on_step=True, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        # log.info(f'Calculating validation CER for each dataset')

        # mean_val_cer /= len(self.val_datasets)
        mean_val_cer, mean_val_wer = self.metric_logger.log_val_metrics()

        self.log(f'val/cer_epoch', mean_val_cer, sync_dist=False, prog_bar=True)
        self.log(f'val/wer_epoch', mean_val_wer, sync_dist=False, prog_bar=True)


    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, dataloader_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        dataloader_idx = 0 if len(self.test_datasets) == 1 else dataloader_idx
        dataset = self.test_datasets[dataloader_idx]
        # print(f'Calculating test CER for each dataset for dataloader_idx: {dataloader_idx}. Dataset: {dataset}')

        y = batch[1].permute(1, 0)
        # y[y == 1] = -100
        labels = y[:, 1:].clone().contiguous() # Shift all labels to the right
        
        # loss, preds, targets = self.model_step(batch)
        preds = self.net.predict_greedy(batch[0])

        preds_str, labels_str = [], []
        # Decode using self.decode
        for i in range(preds.sequences.shape[0]):
          # Convert preds and labels to list of strings to call decode removing special tokens
          _pred_str = self.decode(preds.sequences[i].tolist())
          _label_str = self.decode(labels[i].tolist())
          # Remove special tokens for each string and join the list to get a string
          _pred_str = ''.join([char.replace('<eos>', '').replace('<sos>', '').replace('<pad>', '') for char in _pred_str])
          _label_str = ''.join([char.replace('<eos>', '').replace('<sos>', '').replace('<pad>', '') for char in _label_str])
          preds_str.append(_pred_str)
          labels_str.append(_label_str)
          
        print(f'preds_str (test): {preds_str[:10]}')
        print(f'labels_str (test): {labels_str[:10]}')

        self.metric_logger.log_test_step_cer(preds_str, labels_str, dataset)

        # Calculate CER
        step_cer = self.test_cer[dataset](preds_str, labels_str)
        # self.log(f"test/cer_{dataset}", step_cer, on_step=True, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        self.metric_logger.log_test_metrics()

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        log.info(f'Optimizer: {optimizer}')
        # Number of parameters
        print(f'Number of parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}')
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/cer_epoch",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = MNISTLitModule(None, None, None, None)
