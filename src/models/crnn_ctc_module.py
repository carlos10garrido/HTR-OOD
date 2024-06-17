from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MinMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.text import CharErrorRate as CER
from torchmetrics.regression import MeanAbsoluteError as MAE
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
        self.save_hyperparameters(logger=False, ignore=("datasets"))

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
        self.tokenizer = tokenizer #tokenizer['tokenizers']
        self.decode = self.tokenizer.detokenize

        # loss function for regressing the number of characters in an image
        self.criterion = torch.nn.CTCLoss(blank=self.net.vocab_size, zero_infinity=True, reduction='mean')

        # metric objects for calculating and averaging accuracy across batches
        log.info(f'Logger in HTRTransformerLitModule: {_logger}. Keys: {list(_logger.keys())}')
        self._logger = _logger[list(_logger.keys())[0]]

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
        # print(f'target_lengths: {target_lengths}')
        labels = labels[:, :-1].clone().contiguous() # Remove last label (it is the <eos> token)

        # Calculate for CTC the length of the sequence
        # Input_lenght: to which column in the image there is information
        input_lengths = (torch.ones(images.shape[0], dtype=torch.long).to(images.device) * (images.shape[-1]  // self.net.img_reduction[-1])) - (padded_cols // self.net.img_reduction[-1]).int().to(images.device)
        
        # Arguments CTC LOSS: log_probs, target, input_lenghts, target_lenghts
        # print(f'self.net(images).shape: {self.net(images).shape}')
        preds = self.net(images).log_softmax(-1).permute(1, 0, 2)
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

            # Log training image and predictions
            orig_image = torchvision.utils.make_grid(images[i].detach().cpu(), nrow=1, normalize=True)
            orig_image = torchvision.transforms.ToPILImage()(images[i].detach().cpu())
            self._logger.experiment.log({f'train/original_image_{dataset}': wandb.Image(orig_image, caption=f'Label: {_label} \n Pred: {_pred} \n CER: {cer} \n epoch: {self.current_epoch}')})

            # Write image on disk to check if it is correct
            # orig_image.save(f'./outputs/train_image_{batch_idx}_{i}.png')

        self.metric_logger.log_train_step(loss, torch.tensor([0.0]))

        # update and log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss 

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        self.metric_logger.log_train_metrics()
        # self.metric_logger.update_epoch(self.current_epoch)
        # self.metric_logger_minusc.update_epoch(self.current_epoch)
        self.train_cer_epoch = self.train_cer.compute()

        self.log("train/cer_epoch", self.train_cer_epoch, on_epoch=True, prog_bar=True)

        pass


    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> None:
        """Perform a single validation step on a batch of data from the validation set."""
        images, labels = batch[0], batch[1]
        labels = labels.permute(1, 0)
        labels = labels[:, 1:].clone().contiguous() # Shift all labels to the right
        labels = labels[:, :-1].clone().contiguous() # Remove last label (it is the <eos> token)

        dataset = self.val_datasets[dataloader_idx]

        preds = self.net(images).squeeze(-1).clone().argmax(-1)
        total_cer_per_batch = 0.0
        # print(f'---VALIDATION STEP----- ended')
        
        for i in range(images.shape[0]):
          _label = labels[i].detach().cpu().numpy().tolist()
          _pred = torch.unique_consecutive(preds[i].detach()).cpu().numpy().tolist()
          _pred = [idx for idx in _pred if idx != self.net.vocab_size] # Remove blank token
          
          # _pred, _label = self.decode_text(_pred, self.net.vocab_size), self.decode_text(_label, self.net.vocab_size)
          _pred, _label = self.tokenizer.detokenize(_pred), self.tokenizer.detokenize(_label)

          print(f'Label: {_label} - Pred: {_pred}')

          # Calculate CER converting mayus to minus
          _label_minus = _label.lower()
          _pred_minus = _pred.lower()
          cer_minus = self.val_cer_minus.forward(_pred_minus, _label_minus)
          self.metric_logger_minusc.log_val_step_cer(_pred_minus, _label_minus, f'{dataset}_minusc')
          self.metric_logger_minusc.log_val_step_wer(_pred_minus, _label_minus, f'{dataset}_minusc')

          # Calculate CER
          cer = CER()(_pred, _label)
          if batch_idx < 1:
            orig_image = torchvision.transforms.ToPILImage()(images[i].detach().cpu())
            self._logger.experiment.log({f'val/original_image_{dataset}': wandb.Image(orig_image, caption=f'Label: {_label} \n Pred: {_pred} \n CER: {cer} \n CER minus: {cer_minus} \n epoch: {self.current_epoch}')})

          
          self.metric_logger.log_val_step_cer(_pred, _label, dataset)
          self.metric_logger.log_val_step_wer(_pred, _label, dataset)
          
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
        mean_val_cer_minus, in_domain_cer_minus, out_of_domain_cer_minus, heldout_domain_cers_minus = self.metric_logger_minusc.log_val_metrics()
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
      # Convert 0, 1 and 2 to 0 (padding token) to 0 
      # labels[labels < 3] = 0

      dataloader_idx = 0 if len(self.test_datasets) == 1 else dataloader_idx
      dataset = self.test_datasets[dataloader_idx]

      preds = self.net(images).squeeze(-1)
      preds = preds.clone().argmax(-1)

      total_cer_per_batch = 0.0
      # print(f'-- TEST STEP----- ended')
      
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
        if batch_idx < 1:
          orig_image = torchvision.transforms.ToPILImage()(images[i].detach().cpu())
          # self._logger.experiment.log({f'test/preds_{dataset}': wandb.Image(image_, caption=f'Label: {_label} \n Pred: {_pred} \n CER: {cer} \n CER minus: {cer_minus} \n epoch: {self.current_epoch}')})
          self._logger.experiment.log({f'test/original_image_{dataset}': wandb.Image(orig_image, caption=f'Label: {_label} \n Pred: {_pred} \n CER: {cer} \n CER minus: {cer_minus} \n epoch: {self.current_epoch}')})

        
        self.metric_logger.log_test_step_cer(_pred, _label, dataset)
        self.metric_logger.log_test_step_wer(_pred, _label, dataset)
        
        total_cer_per_batch += cer

      # print(f'Total CER per batch: {total_cer_per_batch/images.shape[0]}')
      

    def on_test_epoch_end(self) -> None:
        test_cer, test_wer = self.metric_logger.log_test_metrics()
        print(f'test_cer: {test_cer}')
        print(f'test_wer: {test_wer}')
        # self.log(f'test/cer_epoch', test_cer, sync_dist=True, prog_bar=True)
        # self.log(f'test/wer_epoch', test_wer, sync_dist=True, prog_bar=True)

        test_cer_minus, test_wer_minus = self.metric_logger_minusc.log_test_metrics()
        print(f'test_cer_minus: {test_cer_minus}')
        print(f'test_wer_minus: {test_wer_minus}')
        # self.log(f'test/cer_epoch_minusc', test_cer_minus, sync_dist=True, prog_bar=True)
        # self.log(f'test/wer_epoch_minusc', test_wer_minus, sync_dist=True, prog_bar=True)


        # mean_test_cer, in_domain_cer, out_of_domain_cer, heldout_domain_cer = self.metric_logger.log_test_metrics()
        # print(f'mean_test_cer: {mean_test_cer}')
        # self.log(f'test/mean_cer', mean_test_cer, sync_dist=True, prog_bar=True)
        # self.log(f'test/in_domain_cer', in_domain_cer, sync_dist=True, prog_bar=True)
        # self.log(f'test/out_of_domain_cer', out_of_domain_cer, sync_dist=True, prog_bar=True)
        # self.log(f'test/heldout_domain_cer', heldout_domain_cer, sync_dist=True, prog_bar=True)
        
        # Log CER minusc
        # mean_test_cer_minus, in_domain_cer_minus, out_of_domain_cer_minus, heldout_domain_cer_minus = self.metric_logger_minusc.log_test_metrics()
        # self.log(f'test/mean_cer_minusc', mean_test_cer_minus, sync_dist=True, prog_bar=True)
        # self.log(f'test/in_domain_cer_minusc', in_domain_cer_minus, sync_dist=True, prog_bar=True)
        # self.log(f'test/out_of_domain_cer_minusc', out_of_domain_cer_minus, sync_dist=True, prog_bar=True)
        # self.log(f'test/heldout_domain_cer_minusc', heldout_domain_cer_minus, sync_dist=True, prog_bar=True)

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
                    "monitor": "val/cer_epoch",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = HTRCharClassifier(None, None, None, None)
