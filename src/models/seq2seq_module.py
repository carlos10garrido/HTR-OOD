from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics.text import CharErrorRate as CER
from src.utils import pylogger
from src.utils.logger import MetricLogger
log = pylogger.RankedLogger(__name__, rank_zero_only=True)
import torchvision
import os
import src
from src.data.components.tokenizers import Tokenizer

import numpy as np
# import cv2
from PIL import Image
import wandb

# PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 1, 0, 2, 3

class Seq2SeqModule(LightningModule):
  
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        _logger: Any,
        datasets: dict,
        tokenizer: Tokenizer,
        log_val_metrics: bool = True,
    ) -> None:
        """Initialize a `Seq2SeqModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=("datasets", "tokenizer", "_logger"))
        self.log_val_metrics = log_val_metrics
        print(f'Loggin val metrics: {self.log_val_metrics}')

        # Save datasets names in a list to index from validation_step
        self.train_datasets = list(datasets['train']['train_config']['datasets'].keys())
        self.val_datasets = list(datasets['val']['val_config']['datasets'].keys())
        self.test_datasets = list(datasets['test']['test_config']['datasets'].keys())

        print(f'self.train_datasets: {self.train_datasets}')
        print(f'self.val_datasets: {self.val_datasets}')
        print(f'self.test_datasets: {self.test_datasets}')
        
        self.train_cer = CER()
        self.val_cer_minus = CER()
        self.test_cer_minus = CER()
        self.net = net
        self.tokenizer = tokenizer

        # add label smoothing of 0.4 for transformers
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_id, label_smoothing=0.4) if self.net.__class__.__name__ == 'TransformerKangTorch' else torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_id)

        # metric objects for calculating and averaging accuracy across batches
        log.info(f'Logger in Seq2SeqModule: {_logger}. Keys: {list(_logger.keys())}')
        self._logger = _logger[list(_logger.keys())[0]]

        # Log train datasets, val datasets and test datasets
        self._logger.log_hyperparams({
          'train_datasets': self.train_datasets,
          'val_datasets': self.val_datasets,
          'test_datasets': self.test_datasets,
        })

        self.metric_logger = MetricLogger(
          logger=self._logger,
          tokenizer=self.tokenizer,
          train_datasets=self.train_datasets,
          val_datasets=self.val_datasets,
          test_datasets=self.test_datasets,
        )

        self.metric_logger_minusc = MetricLogger(
          logger=self._logger,
          tokenizer=self.tokenizer,
          # Append minusc to val_datasets and test_datasets
          train_datasets=[f'{train_dataset}_minusc' for train_dataset in self.train_datasets],
          val_datasets=[f'{val_dataset}_minusc' for val_dataset in self.val_datasets],
          test_datasets=[f'{test_dataset}_minusc' for test_dataset in self.test_datasets],
        )

        # Log vocab size from net
        self._logger.log_hyperparams({
          'vocab_size': self.net.vocab_size,
        })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_fit_start(self) -> None:
        """Lightning hook that is called when training begins."""


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

        labels = batch[1].permute(1, 0).type(torch.LongTensor).to(self.device)
        outputs = self.net(images, labels[:, :-1]) # Remove last label (it is the <eos> token)
        labels = labels[:, 1:].contiguous() # Shift all labels to the right as causal sequence modeling

        # Check if outputs is an instance of Hugging Face ModelOutput
        if hasattr(outputs, 'logits'):
          logits = outputs.logits
          loss = outputs.loss
        else:
          logits = outputs
          
        loss = self.criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        acc = (logits.argmax(dim=-1) == labels).sum() / (labels != 1).sum() # 1 is the padding token
        self.metric_logger.log_train_step(loss, acc)

        # update and log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Log learning rate
        self.log("train/lr_step", self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=True)

        return loss 

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        self.metric_logger.log_train_metrics()
        
        # Log the learning rate for that epoch
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        epoch = self.current_epoch
        print(f'Learning rate: {lr} for epoch: {epoch}')
        self.metric_logger.log_learning_rate(lr, epoch)

        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, dataloader_idx: int = None) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        dataloader_idx = 0 if len(self.val_datasets) == 1 else dataloader_idx
        dataset = self.val_datasets[dataloader_idx]
        
        images, labels = batch[0], batch[1]
        labels = labels.permute(1, 0)
        labels = labels[:, 1:].contiguous() # Shift all labels to the right

        total_cer_per_batch = 0.0

        if self.current_epoch == 0 and self.global_step <= 1:
          str_train_datasets = f'val_' + ', '.join(self.train_datasets)
          # self.metric_logger.log_images(images, str_train_datasets)

        preds, raw_preds = self.net.predict_greedy(images)
        preds = preds.sequences if hasattr(preds, 'sequences') else preds
        raw_preds = raw_preds.squeeze(-1)
        
        if self.log_val_metrics:
          self.metric_logger.log_val_step_confidence(raw_preds, dataset)
          self.metric_logger.log_val_step_calibration(raw_preds, labels, dataset)
          self.metric_logger.log_val_step_int_perplexity(raw_preds, dataset)

        for i in range(images.shape[0]):
          _label = labels[i].detach().cpu().numpy().tolist()
          _pred = preds[i].tolist()

          _label = [label if label != -100 else self.tokenizer.pad_id for label in _label]
          _label, _pred = self.tokenizer.detokenize(_label), self.tokenizer.detokenize(_pred)
          
          # if batch_idx < 1:
          #   print(f'VAL Label: {_label}. Pred: {_pred}')
          
          self.metric_logger.log_val_step_cer(_pred, _label, dataset)
          self.metric_logger.log_val_step_wer(_pred, _label, dataset)
          _pred_minus, _label_minus = _pred.lower(), _label.lower()

          self.metric_logger_minusc.log_val_step_cer(_pred_minus, _label_minus, f'{dataset}_minusc')
          self.metric_logger_minusc.log_val_step_wer(_pred_minus, _label_minus, f'{dataset}_minusc')

          cer = CER()(_pred, _label)
          total_cer_per_batch += cer

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        
        epoch = self.current_epoch
        
        # Change this if you want to skip validation for the first N epochs
        if epoch >= 0:
          mean_val_cer, in_domain_cer, out_of_domain_cer, heldout_domain_cers, val_cers = self.metric_logger.log_val_metrics()
          for dataset, val_cer in val_cers.items():
              self.log(f'val/val_cer_{dataset}', val_cer, sync_dist=True, prog_bar=True)
          
          print(f'mean_val_cer: {mean_val_cer}')
          self.log(f'val/mean_cer', mean_val_cer, sync_dist=True, prog_bar=True)
          self.log(f'val/in_domain_cer', in_domain_cer, sync_dist=True, prog_bar=True)
          self.log(f'val/out_of_domain_cer', out_of_domain_cer, sync_dist=True, prog_bar=True)
          
          for name, heldout_domain_cer in heldout_domain_cers.items():
            # Check if heldout_domain_cer is a tensor or list
            if isinstance(heldout_domain_cer, torch.Tensor):
              heldout_domain_cer = heldout_domain_cer.item()
            if isinstance(heldout_domain_cer, list):
              heldout_domain_cer = heldout_domain_cer[0].item()

            self.log(f'val/heldout_target_{name}', heldout_domain_cer, sync_dist=True, prog_bar=True)
          
          # Log CER minusc
          mean_val_cer_minus, in_domain_cer_minus, out_of_domain_cer_minus, heldout_domain_cer_minus, val_cers_minusc = self.metric_logger_minusc.log_val_metrics()
          for dataset, val_cer in val_cers_minusc.items():
            self.log(f'val/val_cer_minusc_{dataset}', val_cer, sync_dist=True, prog_bar=True)
            
          self.log(f'val/mean_cer_minusc', mean_val_cer_minus, sync_dist=True, prog_bar=True)
          self.log(f'val/in_domain_cer_minusc', in_domain_cer_minus, sync_dist=True, prog_bar=True)
          self.log(f'val/out_of_domain_cer_minusc', out_of_domain_cer_minus, sync_dist=True, prog_bar=True)
          # self.log(f'val/heldout_domain_cer_minusc', heldout_domain_cer_minus, sync_dist=True, prog_bar=True)
          for name, heldout_domain_cer in heldout_domain_cer_minus.items():
            # Check if heldout_domain_cer is a tensor or list
            if isinstance(heldout_domain_cer, torch.Tensor):
              heldout_domain_cer = heldout_domain_cer.item()
            if isinstance(heldout_domain_cer, list):
              heldout_domain_cer = heldout_domain_cer[0].item()
            self.log(f'val/heldout_target_{name}', heldout_domain_cer, sync_dist=True, prog_bar=True)


          self.metric_logger.update_epoch(self.current_epoch)
          self.metric_logger_minusc.update_epoch(self.current_epoch)
        else:
          print(f'Epoch: {epoch}. Skipping validation epoch end for epoch < 100. Setting all metrics to 1.0')
          for dataset in self.val_datasets:
            self.log(f'val/val_cer_{dataset}', 1.0, sync_dist=True, prog_bar=True)
            self.log(f'val/val_cer_minusc_{dataset}', 1.0, sync_dist=True, prog_bar=True)
            
          self.log(f'val/mean_cer', 1.0, sync_dist=True, prog_bar=True)
          self.log(f'val/in_domain_cer', 1.0, sync_dist=True, prog_bar=True)
          self.log(f'val/out_of_domain_cer', 1.0, sync_dist=True, prog_bar=True)
          self.log(f'val/mean_cer_minusc', 1.0, sync_dist=True, prog_bar=True)
          self.log(f'val/in_domain_cer_minusc', 1.0, sync_dist=True, prog_bar=True)
          self.log(f'val/out_of_domain_cer_minusc', 1.0, sync_dist=True, prog_bar=True)
          
          for name in self.val_datasets:
            self.log(f'val/heldout_target_{name}', 1.0, sync_dist=True, prog_bar=True)
            self.log(f'val/heldout_target_{name}_minusc', 1.0, sync_dist=True, prog_bar=True)



    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, dataloader_idx: int = None) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        dataloader_idx = 0 if len(self.test_datasets) == 1 else dataloader_idx
        dataset = self.test_datasets[dataloader_idx]
        # print(f'Calculating test CER for each dataset for dataloader_idx: {dataloader_idx}. Dataset: {dataset}')

        # Get epoch
        epoch = self.current_epoch

        images, labels = batch[0], batch[1]
        labels = labels.permute(1, 0)
        labels = labels[:, 1:].clone().contiguous() # Shift all labels to the right

        total_cer_per_batch = 0.0

        if self.current_epoch == 0 and self.global_step <= 1:
          str_train_datasets = f'test_' + ', '.join(self.train_datasets)
          # self.metric_logger.log_images(images, str_train_datasets)
        
        preds, raw_preds = self.net.predict_greedy(images)#.sequences
        preds = preds.sequences if hasattr(preds, 'sequences') else preds
        
        self.metric_logger.log_test_step_confidence(raw_preds, dataset)
        self.metric_logger.log_test_step_calibration(raw_preds, labels, dataset)
        self.metric_logger.log_test_step_int_perplexity(raw_preds, dataset)
        
        # with torch.no_grad():
        #   _preds = self.net(images, labels)
        # self.metric_logger.log_test_step_ext_perplexity(_preds[:, :-1], labels, dataset)

        preds_str, labels_str = [], []
        for i in range(images.shape[0]):
          _label = labels[i].detach().cpu().numpy().tolist()
          # _pred = preds.sequences[i].tolist()
          _pred = preds[i].tolist()

          _label = [label if label != -100 else self.tokenizer.pad_id for label in _label]
          _label, _pred = self.tokenizer.detokenize(_label), self.tokenizer.detokenize(_pred)
          
          self.metric_logger.log_test_step_cer(_pred, _label, dataset)
          self.metric_logger.log_test_step_wer(_pred, _label, dataset)

          # print(f'TEST Label: {_label}. Pred: {_pred}')
          cer = CER()(_pred, _label)
          
          # if batch_idx < 1:
          #   self._logger.experiment.log({f'test/preds_{dataset}': wandb.Image(images[i], caption=f'Label: {_label} \n Pred: {_pred} \n CER: {cer} \n epoch: {self.current_epoch}')})

          total_cer_per_batch += cer
        
        # print(f'Total CER per batch: {total_cer_per_batch/images.shape[0]}')

    def on_test_epoch_end(self) -> None:
        test_cer, test_wer = self.metric_logger.log_test_metrics()
        print(f'test_cer: {test_cer}')
        print(f'test_wer: {test_wer}')

        test_cer_minus, test_wer_minus = self.metric_logger_minusc.log_test_metrics()
        print(f'test_cer_minus: {test_cer_minus}')
        print(f'test_wer_minus: {test_wer_minus}')

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
            print(f'Using scheduler: {self.hparams.scheduler}')

            # If the class of the scheduler is SequentialLR, set the optimizer for the scheduler
            if "SequentialLR" in str(self.hparams.scheduler.func):
              print(f'Using SequentialLR scheduler')
              scheduler = self.hparams.scheduler(
                optimizer=optimizer,
                  schedulers=[
                    self.hparams.scheduler.keywords['schedulers'][i](optimizer=optimizer) for i in range(len(self.hparams.scheduler.keywords['schedulers']))
                  ]
                )
            else:
              scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/cer_epoch",
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = MNISTLitModule(None, None, None, None)
