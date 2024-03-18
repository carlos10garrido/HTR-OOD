from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MinMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.text import CharErrorRate as CER
from torchmetrics import MeanSquaredError as MSE
from src.utils import pylogger
from src.utils.logger import MetricLogger
log = pylogger.RankedLogger(__name__, rank_zero_only=True)
import torchvision
import os
# import global variables encode and decode from htr_data_module.py
# from src.data.seg_datamodule import encode, decode

import numpy as np
# import cv2
from PIL import Image

import os 
import src
import wandb
import torchvision
# from src.models.transformers_seg import SegTransformerLitModule 

class SegTransformerMaskLitModuleE2E(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        _logger: Any,
        datasets: dict,
        visualize_attention: bool = False,
        invert_images: bool = False,
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
        self.invert_images = invert_images

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
        self.val_mse = {}
        self.min_val_mse = {}
        for dataset in self.val_datasets:
            self.val_mse[dataset] = MSE()
            self.min_val_mse[dataset] = MinMetric()

        # Test metrics
        self.test_mse = {}
        self.min_test_mse = {}
        for dataset in self.test_datasets:
            self.test_mse[dataset] = MSE
            self.min_test_mse[dataset] = MinMetric()

        self.net = net

        # loss function
        self.criterion_segmenter = torch.nn.BCELoss()
        self.criterion_classifier = torch.nn.CrossEntropyLoss(ignore_index=0)

        self.encode = src.data.char_datamodule.encode
        self.decode = src.data.char_datamodule.decode

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

        print(f'ON FIT START')


    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        """Perform a single model step on a batch of data."""
        
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion_segmenter(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        return loss, preds, y

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """

        images, images_masked, seq_lens, sequences = batch
        
        if self.current_epoch == 0 and self.global_step <= 1:
          str_train_datasets = f'train_' + ', '.join(self.train_datasets)
          # self.metric_logger.log_images(images, str_train_datasets)
          # self.metric_logger.log_images(images_masked, str_train_datasets)

        # if self.invert_images:
        #   images = 1 - images
        #   images_masked = 1 - images_masked

        sequences = sequences.permute(1, 0)
        sequences = sequences[:, 1:].clone().contiguous() # Shift all labels to the right
        sequences = sequences.unsqueeze(-1)

        labels = images_masked.flatten(2)
        labels = labels.round()

        # Create a mask with seq_lens that contains a integer with the length of the sequence 
        # for each image in the batch
        mask = torch.arange(labels.shape[1])[None, :].to(self.device) <= seq_lens[:, None]

        # breakpoint()
        outputs = self.net(images=images, labels=labels) # Returns 0 output_decoder, 1 output_classifier

        logits = outputs[0]
        logits = logits[:, :-1, :] # Removing last token that is after the <EOS> token
        # Reshape to initial shape

        logits_classifier = outputs[1]

        logits_classifier = logits_classifier.reshape(-1, logits_classifier.shape[-1])
        sequences = sequences.reshape(-1)

        # Use the mask to compute the loss
        loss = self.criterion_segmenter(logits[mask], labels[mask])
        loss_classifier = self.criterion_classifier(logits_classifier, sequences)

        preds_ = logits_classifier.argmax(dim=-1)

        total_loss = loss + loss_classifier

        preds_ = preds_.reshape(labels.shape[0], -1)
        sequences = sequences.reshape(labels.shape[0], -1)



        if batch_idx < 10:
            images_pred = logits.reshape(images_masked.shape)
            # Make a copy of the images_pred tensor
            images_pred = images_pred.clone().detach()
            images_pred = (images_pred > 0.5).float().round()
            # Reshape labels
            labels = labels.reshape(images_masked.shape)
            for i in range(5):
              print(f'Sequence: {sequences[i]}')
              print(f'Preds: {preds_[i]}')
              images_masked_i = images[i] * images_pred[i]
              _label = sequences[i].detach().cpu().numpy().tolist()
              _pred = preds_[i].detach().cpu().numpy().tolist()
              # _pred_from_segmented = preds_from_segmented[i].detach().cpu().numpy().tolist()
              print(f'_label: {_label}')
              print(f'_pred: {_pred}')

              _label = self.decode(_label)
              _pred = self.decode(_pred)

              _label_, _pred_ = "", ""
              for l in range(len(_label)):
                if _label[l] == '<eos>' or _label[l] == '<sos>' or _label[l] == '<pad>':
                  break
                _label_ += _label[l]

              for l in range(len(_pred)):
                if _pred[l] == '<eos>' or _pred[l] == '<sos>' or _pred[l] == '<pad>':
                  break
                _pred_ += _pred[l]
              

              # _label, _pred, _pred_from_segmented = _label_, _pred_, _pred_from_segmented_
              _label, _pred = _label_, _pred_

              print(f'Images masked shape: {images_masked_i.shape}')
              
              # grid_images = self.metric_logger.log_images(images[i], 'training_images')
              cer_from_segmented = CER()(_pred, _label)
              self.metric_logger.log_images(images_masked[i], 'training_images_masked')
              self._logger.experiment.log({"training_preds_from_segmented": wandb.Image(images_pred[i], caption=f'Label: {_label} \n Pred: {_pred} \n CER: {cer_from_segmented} \n epoch: {self.current_epoch}')})


        self.metric_logger.log_train_step(loss.detach().item(), 0.0)

        # update and log metrics
        self.log("train/loss_segmenter", loss.detach().item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss_classifier", loss_classifier.detach().item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss", total_loss.detach().item(), on_step=True, on_epoch=True, prog_bar=True)


        return total_loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        # self.metric_logger.log_train_metrics()
        self.metric_logger.update_epoch(self.current_epoch)

        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, dataloader_idx: int = None) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        print(f'VALIDATION STEP')
        dataloader_idx = 0 if len(self.val_datasets) == 1 else dataloader_idx
        dataset = self.val_datasets[dataloader_idx]
        # print(f'Calculating validation CER for each dataset for dataloader_idx: {dataloader_idx}. Dataset: {dataset}')

        # Get epoch
        epoch = self.current_epoch
        images, labels = batch[0], batch[1]
        labels = labels.permute(1, 0)
        labels = labels[:, 1:].clone().contiguous() # Shift all labels to the right
        labels = labels[:, :-1].clone().contiguous() # Remove last label (it is the <eos> token)
        print(f'images shape: {images.shape}')

        if self.invert_images:
          images = 1 - images

        if self.current_epoch == 0 and self.global_step <= 1:
          str_val_datasets = f'val_' + ', '.join(self.val_datasets)
          self.metric_logger.log_images(images, str_val_datasets)
        
        preds_seg, preds = self.net.predict_greedy(images)

        preds = preds.argmax(dim=-1)

        print(f'preds_seg shape: {preds_seg.shape}')
        print(f'preds shape: {preds.shape}')
        

        for i in range(images.shape[0]):
          images_pred = preds_seg.reshape(preds_seg.shape[0], preds_seg.shape[1], 1, images.shape[2], images.shape[3]) 
          images_ = self.metric_logger.log_images(images[i], f'val/validation_images_{dataset}')
          _label = labels[i].detach().cpu().numpy().tolist()
          _pred = preds[i].detach().cpu().numpy().tolist()
          # _pred_from_segmented = preds_from_segmented[i].detach().cpu().numpy().tolist()
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
          self.metric_logger.log_val_step_cer(_pred, _label, dataset)
          
          if batch_idx < 20:
            # print(f'Label: {_label}. Pred: {_pred}')
            cer_from_segmented = CER()(_pred, _label)
            self._logger.experiment.log({f'val/preds_{dataset}': wandb.Image(images_pred[i], caption=f'Label: {_label} \n Pred: {_pred} \n CER: {cer_from_segmented} \n epoch: {self.current_epoch}')})

          # self.metric_logger.log_val_step(torch.tensor([0.0]), torch.tensor([0.0]), dataset)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        # log.info(f'Calculating validation CER for each dataset')
        val_cer_epoch = self.metric_logger.log_val_metrics()
        self.log(f"val/cer_epoch", val_cer_epoch, on_step=False, on_epoch=True, prog_bar=True)


    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, dataloader_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        dataloader_idx = 0 if len(self.test_datasets) == 1 else dataloader_idx
        dataset = self.test_datasets[dataloader_idx]
        # print(f'Calculating test CER for each dataset for dataloader_idx: {dataloader_idx}. Dataset: {dataset}')

        images, images_masked = batch[0], batch[1]

        labels = images_masked.reshape(images_masked.shape[0], images_masked.shape[1], -1)
        print(f'Labels shape: {labels.shape}')

        # breakpoint()

        # y = batch[1].permute(1, 0)
        # # y[y == 1] = -100
        # labels = y[:, 1:].clone().contiguous() # Shift all labels to the right
        
        # loss, preds, targets = self.model_step(batch)
        preds = self.net.predict_greedy(batch[0])

        

        # Calculate CER
        # step_mse = self.test_mse[dataset](preds_str, labels_str)
        # self.log(f"test/cer_{dataset}", step_mse, on_step=True, on_epoch=True, prog_bar=True)

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
    _ = SegTransformerMaskLitModule(None, None, None, None)
