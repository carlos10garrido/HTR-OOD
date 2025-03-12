from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics.text import CharErrorRate as CER
from src.utils import pylogger
from src.utils.logger import MetricLogger
log = pylogger.RankedLogger(__name__, rank_zero_only=True)
# import rootutils
# rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import os 
import src
import wandb
import torchvision
from src.data.components.tokenizers import Tokenizer

class CRNN_CTC_Module(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        _logger: Any,
        datasets: dict,
        tokenizer: Tokenizer,
        log_val_metrics: bool = False,
    ) -> None:
        """Initialize a `CRNN_CTC_Module`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param compile: Whether to compile the model.
        :param _logger: The logger to use for logging metrics.
        :param datasets: The datasets to use for training, validation, and testing.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        # self.save_hyperparameters(logger=False, ignore=("datasets"))
        self.save_hyperparameters(logger=False, ignore=("datasets", "tokenizer", "_logger"))

        # Save datasets names in a list to index from validation_step
        self.train_datasets = list(datasets['train']['train_config']['datasets'].keys())
        self.val_datasets = list(datasets['val']['val_config']['datasets'].keys())
        self.test_datasets = list(datasets['test']['test_config']['datasets'].keys())
        
        self.log_val_metrics = log_val_metrics

        print(f'self.train_datasets: {self.train_datasets}')
        print(f'self.val_datasets: {self.val_datasets}')
        print(f'self.test_datasets: {self.test_datasets}')

        self.train_cer = CER()

        self.val_cer_minus = CER()
        self.test_cer_minus = CER()
        self.net = net
        self.tokenizer = tokenizer #tokenizer['tokenizers']
        self.decode = self.tokenizer.detokenize

        # loss function for regressing the number of characters in an image
        self.criterion = torch.nn.CTCLoss(blank=self.net.vocab_size, zero_infinity=True, reduction='mean')

        # metric objects for calculating and averaging accuracy across batches
        log.info(f'Logger in HTRTransformerLitModule: {_logger}. Keys: {list(_logger.keys())}')
        self._logger = _logger[list(_logger.keys())[0]]
        # self._logger = _logger

        # Log train datasets, val datasets and test datasets
        self._logger.log_hyperparams({
          'train_datasets': self.train_datasets,
          'val_datasets': self.val_datasets,
          'test_datasets': self.test_datasets,
        })

        # Log net.vocab_size
        self._logger.log_hyperparams({
          'vocab_size': self.net.vocab_size,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """

        return self.net(x)

    def on_fit_start(self) -> None:
        """Lightning hook that is called when training begins."""
        

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets."""
        
        images, labels, padded_cols = batch[0], batch[1], batch[2]
        dataset = self.train_datasets[dataloader_idx]
        # print(f'Images shape: {images.shape}')
        if self.current_epoch == 0 and self.global_step <= 1:
          str_train_datasets = f'train_' + ', '.join(self.train_datasets)
          self.metric_logger.log_images(images, str_train_datasets)

        y = batch[1].permute(1, 0)
        labels = y[:, 1:].clone().contiguous() # Shift all labels to the right
        target_lengths = torch.where(labels == self.tokenizer.eos_id)[1]

        # Calculate for CTC the length of the sequence  
        # Input_lenght: to which column in the image there is information
        preds = self.net(images).log_softmax(-1).permute(1, 0, 2)
        input_lengths = torch.LongTensor([preds.size(0)] * images.shape[0])
        loss = self.criterion(preds, labels, input_lengths, target_lengths)
        preds_ = preds.clone().permute(1, 0, 2).argmax(-1)


        # Log images and predictions
        if batch_idx < 1:
          for i in range(images.shape[0]):
            _label = labels[i].detach().cpu().numpy().tolist()
            # Remove consecutive repeated tokens
            _pred = torch.unique_consecutive(preds_[i].detach()).cpu().numpy().tolist()
            _pred = [idx for idx in _pred if idx != self.net.vocab_size] # Remove blank token            
            _pred, _label = self.tokenizer.detokenize(_pred), self.tokenizer.detokenize(_label)

            cer = CER()(_pred, _label)

        self.metric_logger.log_train_step(loss, torch.tensor([0.0]))

        # update and log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Log learning rate
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log(f'training/lr_step', lr, sync_dist=True, on_step=True, on_epoch=True, prog_bar=False)
        
        return loss 

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        self.metric_logger.log_train_metrics()
        
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        epoch = self.current_epoch
        print(f'Learning rate: {lr} for epoch: {epoch}')
        self.metric_logger.log_learning_rate(lr, epoch)

        pass


    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> None:
        """Perform a single validation step on a batch of data from the validation set."""
        images, labels = batch[0], batch[1]
        labels = labels.permute(1, 0)
        labels = labels[:, 1:].clone().contiguous() # Shift all labels to the right

        dataset = self.val_datasets[dataloader_idx]
        raw_preds = self.net(images).squeeze(-1).clone()
        
        if self.current_epoch == 0:
          str_val_dataset = f'val_' + dataset
          self.metric_logger.log_images(images, str_val_dataset)
          
        
        # Calculate confidence and perplexity
        if self.log_val_metrics:
          self.metric_logger.log_val_step_confidence(raw_preds, dataset)
        
        preds = raw_preds.clone().argmax(-1)
        
        self.metric_logger.log_val_step_confidence(raw_preds, dataset)
        self.metric_logger.log_val_step_calibration(raw_preds, labels, dataset)
          
        total_cer_per_batch = 0.0
        
        for i in range(images.shape[0]):
          _label = labels[i].detach().cpu().numpy().tolist()
          _pred = torch.unique_consecutive(preds[i].detach()).cpu().numpy().tolist()
          _pred = [idx for idx in _pred if idx != self.net.vocab_size] # Remove blank token
          _pred, _label = self.tokenizer.detokenize(_pred), self.tokenizer.detokenize(_label)

          # Calculate CER converting mayus to minus
          _label_minus = _label.lower()
          _pred_minus = _pred.lower()
          cer_minus = self.val_cer_minus.forward(_pred_minus, _label_minus)
          self.metric_logger_minusc.log_val_step_cer(_pred_minus, _label_minus, f'{dataset}_minusc')
          self.metric_logger_minusc.log_val_step_wer(_pred_minus, _label_minus, f'{dataset}_minusc')

          # Calculate CER
          cer = CER()(_pred, _label)
          
          self.metric_logger.log_val_step_cer(_pred, _label, dataset)
          self.metric_logger.log_val_step_wer(_pred, _label, dataset)
          
          total_cer_per_batch += cer

        # print(f'Total CER per batch: {total_cer_per_batch/images.shape[0]}')

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."

        mean_val_cer, in_domain_cer, out_of_domain_cer, heldout_domain_cers, val_cers = self.metric_logger.log_val_metrics()
        for dataset, val_cer in val_cers.items():
            self.log(f'val/val_cer_{dataset}', val_cer, sync_dist=True, prog_bar=True)
            
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
        mean_val_cer_minus, in_domain_cer_minus, out_of_domain_cer_minus, heldout_domain_cers_minus, val_cers_minusc = self.metric_logger_minusc.log_val_metrics()
        for dataset, val_cer in val_cers_minusc.items():
          self.log(f'val/val_cer_minusc_{dataset}', val_cer, sync_dist=True, prog_bar=True)
          
        self.log(f'val/mean_cer_minusc', mean_val_cer_minus, sync_dist=True, prog_bar=True)
        self.log(f'val/in_domain_cer_minusc', in_domain_cer_minus, sync_dist=True, prog_bar=True)
        self.log(f'val/out_of_domain_cer_minusc', out_of_domain_cer_minus, sync_dist=True, prog_bar=True)
        # self.log(f'val/heldout_domain_cer_minusc', heldout_domain_cer_minus, sync_dist=True, prog_bar=True)
        for name, heldout_domain_cer in heldout_domain_cers_minus.items():
          # Check if heldout_domain_cer is a tensor or list
          if isinstance(heldout_domain_cer, torch.Tensor):
            heldout_domain_cer = heldout_domain_cer.item()
          if isinstance(heldout_domain_cer, list):
            heldout_domain_cer = heldout_domain_cer[0].item()
          self.log(f'val/heldout_target_{name}', heldout_domain_cer, sync_dist=True, prog_bar=True)
          
        self.metric_logger.update_epoch(self.current_epoch)
        self.metric_logger_minusc.update_epoch(self.current_epoch)

        
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> None:
      """Perform a single TEST step on a batch of data from the TEST set."""
      images, labels = batch[0], batch[1]
      labels = labels.permute(1, 0)
      labels = labels[:, 1:].clone().contiguous() # Shift all labels to the right
      labels = labels[:, :-1].clone().contiguous() # Remove last label (it is the <eos> token)

      dataloader_idx = 0 if len(self.test_datasets) == 1 else dataloader_idx
      dataset = self.test_datasets[dataloader_idx]
      
      raw_preds = self.net(images).squeeze(-1).clone()
      preds = raw_preds.clone().argmax(-1)
      
      self.metric_logger.log_test_step_confidence(raw_preds, dataset)
      self.metric_logger.log_test_step_calibration(raw_preds, labels, dataset)

      total_cer_per_batch = 0.0
      
      for i in range(images.shape[0]):
        _label = labels[i].detach().cpu().numpy().tolist()
        _pred = torch.unique_consecutive(preds[i].detach()).cpu().numpy().tolist()
        _pred = [idx for idx in _pred if idx != self.net.vocab_size] # Remove blank token

        _pred, _label = self.tokenizer.detokenize(_pred), self.tokenizer.detokenize(_label)

        # print(f'Label: {_label} - Pred: {_pred}')

        # Calculate CER converting mayus to minus
        _label_minus = _label.lower()
        _pred_minus = _pred.lower()
        cer_minus = self.test_cer_minus.forward(_pred_minus, _label_minus)
        self.metric_logger_minusc.log_test_step_cer(_pred_minus, _label_minus, f'{dataset}_minusc')

        # Calculate CER
        cer = CER()(_pred, _label)
        
        self.metric_logger.log_test_step_cer(_pred, _label, dataset)
        self.metric_logger.log_test_step_wer(_pred, _label, dataset)
        
        total_cer_per_batch += cer

      # print(f'Total CER per batch: {total_cer_per_batch/images.shape[0]}')
      

    def on_test_epoch_end(self) -> None:
        test_cer, test_wer = self.metric_logger.log_test_metrics()
        print(f'test_cer: {test_cer}')
        print(f'test_wer: {test_wer}')
        
        test_cer_minus, test_wer_minus = self.metric_logger_minusc.log_test_metrics()
        print(f'test_cer_minus: {test_cer_minus}')
        print(f'test_wer_minus: {test_wer_minus}')
        
        self.metric_logger.update_epoch(self.current_epoch)
        self.metric_logger_minusc.update_epoch(self.current_epoch)

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
                    # "monitor": "val/cer_epoch",
                    "monitor": "val/in_domain_cer",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = CRNN_CTC_Module(None, None, None, None)
