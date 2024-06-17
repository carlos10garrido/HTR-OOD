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
from src.data.components.tokenizers import Tokenizer

import numpy as np
# import cv2
from PIL import Image
import wandb

# PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 1, 0, 2, 3

class HybridModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        _logger: Any,
        datasets: dict,
        tokenizer: Tokenizer,
    ) -> None:
        """Initialize a `HybridModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)#, ignore=("datasets"))

        # Save datasets names in a list to index from validation_step
        self.train_datasets = list(datasets['train']['train_config']['datasets'].keys())
        self.val_datasets = list(datasets['val']['val_config']['datasets'].keys())
        self.test_datasets = list(datasets['test']['test_config']['datasets'].keys())

        print(f'self.train_datasets: {self.train_datasets}')
        print(f'self.val_datasets: {self.val_datasets}')
        print(f'self.test_datasets: {self.test_datasets}')

        self.train_loss = MeanMetric()
        self.train_cer = CER()

        self.val_cer_minus = CER()
        self.test_cer_minus = CER()
        self.net = net
        self.tokenizer = tokenizer
        self.decode = self.tokenizer.detokenize

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_id) # 1 is the padding token
        self.ctc_criterion = torch.nn.CTCLoss(blank=self.tokenizer.pad_id, zero_infinity=True)

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
          train_datasets=self.train_datasets,
          val_datasets=self.val_datasets,
          test_datasets=self.test_datasets,
        )

        self.metric_logger_minusc = MetricLogger(
          logger=self._logger,
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
        images, labels, padded_cols = batch[0], batch[1], batch[2]
        if self.current_epoch == 0 and self.global_step <= 1:
          str_train_datasets = f'train_' + ', '.join(self.train_datasets)
          self.metric_logger.log_images(images, str_train_datasets)

        labels = batch[1].permute(1, 0)
        labels = labels[:, 1:].clone().contiguous() # Shift all labels to the right to remove the <bos> token

        target_lengths = torch.where(labels == self.tokenizer.eos_id)[1]
        input_lengths = (torch.ones(images.shape[0], dtype=torch.long).to(images.device) * (images.shape[-1]  // self.net.img_reduction[-1])) - (padded_cols // self.net.img_reduction[-1]).int().to(images.device)

        # print(f'input_lengths: {input_lengths}')
        # print(f'target_lengths: {target_lengths}')

        # breakpoint()
        enc_outputs, dec_outputs = self.net(x=images, y=labels)
        # print(f'enc outputs shape: {enc_outputs.shape}')
        outputs = dec_outputs[:, :-1]

        # Check if outputs is an instance of Hugging Face ModelOutput
        if hasattr(outputs, 'logits'):
          logits = outputs.logits
          loss = outputs.loss
        else:
          logits = outputs
          loss_ce = self.criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
          loss_enc = self.ctc_criterion(enc_outputs, labels, input_lengths, target_lengths)
          loss = 0.5 * loss_ce + (1 - 0.5) * loss_enc
          # loss = loss_ce
          

        output_argmax = logits.argmax(dim=-1)
        # print(f'train output_argmax[:4]: {output_argmax[:4]}')
        # print(f'train labels[:4]: {labels[:4]}')
    
        
        # print(f'loss: {loss}')
        acc = (logits.argmax(dim=-1) == labels).sum() / (labels != 1).sum() # 1 is the padding token
        # print(f'acc: {acc}')
        self.metric_logger.log_train_step(loss, acc)

        if batch_idx < 2:
          for i in range(images.shape[0]):
            # images_ = self.metric_logger.log_images(images[i], f'train/training_images_{self.train_datasets[0]}')
            # images_ = images[i]
            images_ = torchvision.transforms.ToPILImage()(images[i].detach().cpu())
            _label = labels[i].detach().cpu().numpy().tolist()
            _label = [label if label != -100 else self.tokenizer.pad_id for label in _label]
            _pred = logits[i].argmax(-1).detach().cpu().numpy().tolist()
            _label, _pred = self.tokenizer.detokenize(_label), self.tokenizer.detokenize(_pred)
            print(f'Label: {_label}. Pred: {_pred}')
            cer = CER()(_pred, _label)
            # self._logger.experiment.log({f'train/preds_{self.train_datasets[0]}': wandb.Image(images_, caption=f'Label: {_label} \n Pred: {_pred} \n CER: {cer} \n epoch: {self.current_epoch}')})

        # update and log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss 

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        self.metric_logger.log_train_metrics()
        # self.metric_logger.update_epoch(self.current_epoch)
        # self.metric_logger_minusc.update_epoch(self.current_epoch)

        # Log the learning rate for that epoch
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        epoch = self.current_epoch
        self.metric_logger.log_learning_rate(lr, epoch)
        # self.log(f'learning_rate', lr, sync_dist=True, prog_bar=True)

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
        labels = labels.permute(1, 0)
        labels = labels[:, 1:].clone().contiguous() # Shift all labels to the right
        # print(f'Labels val')
        # print(f'labels[:4]: {labels[:4]}')

        total_cer_per_batch = 0.0

        if self.current_epoch == 0 and self.global_step <= 1:
          str_train_datasets = f'val_' + ', '.join(self.train_datasets)
          self.metric_logger.log_images(images, str_train_datasets)

        
        preds = self.net.predict_greedy(images)
        # print(f'Preds val. Shape: {preds.shape}')
        # print(f'preds[:4]: {preds[:4]}')

        preds = preds[:, 1:].clone().contiguous() # Shift all labels to the right to remove the <bos> token
        
        preds_str, labels_str = [], []
        for i in range(images.shape[0]):
          images_ = self.metric_logger.log_images(images[i], f'val/validation_images_{dataset}') if self.current_epoch == 0 and batch_idx == 0 else None
          _label = labels[i].detach().cpu().numpy().tolist()
          # _pred = preds.sequences[i].tolist()
          _pred = preds[i].tolist()

          _label = [label if label != -100 else self.tokenizer.pad_id for label in _label]
          _label, _pred = self.tokenizer.detokenize(_label), self.tokenizer.detokenize(_pred)
          
          self.metric_logger.log_val_step_cer(_pred, _label, dataset)
          self.metric_logger.log_val_step_wer(_pred, _label, dataset)
          _pred_minus = _pred.lower()
          _label_minus = _label.lower()

          self.metric_logger_minusc.log_val_step_cer(_pred_minus, _label_minus, f'{dataset}_minusc')
          self.metric_logger_minusc.log_val_step_wer(_pred_minus, _label_minus, f'{dataset}_minusc')

          # print(f'VAL Label: {_label}. Pred: {_pred}')
          cer = CER()(_pred, _label)
          
          if batch_idx < 1:
            print(f'VAL Label: {_label}. Pred: {_pred}')
            # self._logger.experiment.log({f'val/preds_{dataset}': wandb.Image(images[i], caption=f'Label: {_label} \n Pred: {_pred} \n CER: {cer} \n epoch: {self.current_epoch}')})


          total_cer_per_batch += cer
        
        # print(f'Total CER per batch: {total_cer_per_batch/images.shape[0]}')



    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."

        mean_val_cer, in_domain_cer, out_of_domain_cer, heldout_domain_cers = self.metric_logger.log_val_metrics()
        print(f'mean_val_cer: {mean_val_cer}')
        self.log(f'val/mean_cer', mean_val_cer, sync_dist=True, prog_bar=True)
        self.log(f'val/in_domain_cer', in_domain_cer, sync_dist=True, prog_bar=True)
        self.log(f'val/out_of_domain_cer', out_of_domain_cer, sync_dist=True, prog_bar=True)
        # self.log(f'val/heldout_domain_cer', heldout_domain_cer, sync_dist=True, prog_bar=True)
        for name, heldout_domain_cer in heldout_domain_cers.items():
          # Check if heldout_domain_cer is a tensor or list
          if isinstance(heldout_domain_cer, torch.Tensor):
            heldout_domain_cer = heldout_domain_cer.item()
          if isinstance(heldout_domain_cer, list):
            heldout_domain_cer = heldout_domain_cer[0].item()

          self.log(f'val/heldout_target_{name}', heldout_domain_cer, sync_dist=True, prog_bar=True)
        
        # Log CER minusc
        mean_val_cer_minus, in_domain_cer_minus, out_of_domain_cer_minus, heldout_domain_cer_minus = self.metric_logger_minusc.log_val_metrics()
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
        print(f'images.shape: {images.shape}')
        labels = labels.permute(1, 0)
        labels = labels[:, 1:].clone().contiguous() # Shift all labels to the right
        # labels = labels[:, :-1].clone().contiguous() # Remove last label (it is the <eos> token)

        total_cer_per_batch = 0.0

        if self.current_epoch == 0 and self.global_step <= 1:
          str_train_datasets = f'test_' + ', '.join(self.train_datasets)
          self.metric_logger.log_images(images, str_train_datasets)
        
        preds = self.net.predict_greedy(images)#.sequences
        preds = preds[:, 1:].clone().contiguous() # Shift all labels to the right

        preds_str, labels_str = [], []
        for i in range(images.shape[0]):
          # images_ = self.metric_logger.log_images(images[i], f'test/test_images_{dataset}') if batch_idx == 0 else None
          _label = labels[i].detach().cpu().numpy().tolist()
          # _pred = preds.sequences[i].tolist()
          _pred = preds[i].tolist()

          _label = [label if label != -100 else self.tokenizer.pad_id for label in _label]
          _label, _pred = self.tokenizer.detokenize(_label), self.tokenizer.detokenize(_pred)
          
          self.metric_logger.log_test_step_cer(_pred, _label, dataset)
          self.metric_logger.log_test_step_wer(_pred, _label, dataset)

          print(f'TEST Label: {_label}. Pred: {_pred}')
          cer = CER()(_pred, _label)
          
          if batch_idx < 1:
            self._logger.experiment.log({f'test/preds_{dataset}': wandb.Image(images[i], caption=f'Label: {_label} \n Pred: {_pred} \n CER: {cer} \n epoch: {self.current_epoch}')})

          total_cer_per_batch += cer
        
        print(f'Total CER per batch: {total_cer_per_batch/images.shape[0]}')

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

            breakpoint()

            # If the class of the scheduler is SequentialLR, set the optimizer for the scheduler
            if "SequentialLR" in str(self.hparams.scheduler.func):
              scheduler = self.hparams.scheduler(
                optimizer=optimizer,
                  schedulers=[
                    # self.hparams.scheduler.keywords['schedulers'][0](optimizer=optimizer),
                    # self.hparams.scheduler.keywords['schedulers'][1](optimizer=optimizer)
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
