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
from src.models.transformers_seg import SegTransformerLitModule 



class HTRCharClassifier(LightningModule):
  
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        _logger: Any,
        datasets: dict,
        pretrained_seg: str = None,
        # datasets: dict,
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

        # Save datasets names in a list to index from validation_step
        self.train_datasets = list(datasets['train']['train_config']['datasets'].keys())
        self.val_datasets = list(datasets['val']['val_config']['datasets'].keys())
        self.test_datasets = list(datasets['test']['test_config']['datasets'].keys())

        print(f'self.train_datasets: {self.train_datasets}')
        print(f'self.val_datasets: {self.val_datasets}')
        print(f'self.test_datasets: {self.test_datasets}')

        # Load a pretrained model if the path is provided (SegTransformerLitModule)
        if pretrained_seg is not None:
          self.segmenter = SegTransformerLitModule.load_from_checkpoint(pretrained_seg)
          # Freeze the segmenter
          for param in self.segmenter.parameters():
            param.requires_grad = False
        else: 
          self.segmenter = None

        # breakpoint()

        # Create metrics per dataset
        # Training metrics
        self.train_loss = MeanMetric()
        self.train_cer = CER()
        self.train_cer_from_segmented = CER()

        self.val_cer_minus = CER()

        self.net = net
        self.table_train = None

        self.encode = src.data.char_datamodule.encode
        self.decode = src.data.char_datamodule.decode

        # loss function for regressing the number of characters in an image
        self.criterion = torch.nn.CrossEntropyLoss()

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

        self.metric_logger_minusc = MetricLogger(
          logger=self._logger,
          # Append minusc to val_datasets and test_datasets
          train_datasets=self.train_datasets,
          val_datasets=[f'{val_dataset}_minusc' for val_dataset in self.val_datasets],
          test_datasets=[f'{test_dataset}_minusc' for test_dataset in self.test_datasets],
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
        images, images_masked, lens_seqs, labels = batch
        if self.current_epoch == 0 and self.global_step <= 1:
          str_train_datasets = f'train_' + ', '.join(self.train_datasets)
          self.metric_logger.log_images(images, str_train_datasets)

        y = batch[-1].permute(1, 0)
        # y[y == 1] = -100
        labels = y[:, 1:].clone().contiguous() # Shift all labels to the right
        labels = labels[:, :-1].clone().contiguous() # Remove last label (it is the <eos> token)
        # Convert 0, 1 and 2 to 0 (padding token) to 0 
        labels[labels < 3] = 0

        # Images masked contain a character per image, so we need to flatten the tensor since it is a classification problem per character
        images_masked_ = images_masked.flatten(0, 1)
        images_masked_ = images_masked_.round()
        labels_ = labels.flatten(0, 1)

        # Predict with segmenter and get the masked images
        images_masked_segmenter = self.segmenter.net.predict_greedy(images).logits
        # Threshold the images (WARNING, CHANING THIS THRESHOLD MAY AFFECT THE PERFORMANCE OF THE MODEL)
        images_masked_segmenter = images_masked_segmenter > 0.5
        images_masked_segmenter = images_masked_segmenter.float()

        print(f'Images masked segmenter (raw) shape: {images_masked_segmenter.shape}')
        # images_masked_segmenter = images_masked_segmenter.reshape(-1, 1, 64, 64)
        images_masked_segmenter = images_masked_segmenter.reshape(-1, 1, images_masked.shape[-2], images_masked.shape[-1])

        preds_from_masked = self.net(images_masked_)

        preds = self.net(images_masked_segmenter)
        preds = preds.squeeze(-1)

        preds_from_segmented = self.net(images_masked_segmenter)
        print(f'Preds from segmented (after masking): {preds_from_segmented.shape}')
      
        # preds_from_segmented = preds_from_segmented.reshape(images.shape[0], -1, 1, 64, 64)
        preds_from_segmented = preds_from_segmented.reshape(images.shape[0], -1, preds_from_segmented.shape[-1]).softmax(-1).argmax(-1)
        print(f'Preds from segmented (after reshape): {preds_from_segmented.shape}')
        preds_ = preds_from_masked.reshape(images_masked.shape[0], -1, preds_from_masked.shape[-1]).softmax(-1).argmax(-1)

        images_masked_segmenter = images_masked_segmenter.reshape(images_masked.shape[0], -1, 1, images_masked.shape[-2], images_masked.shape[-1])

        loss = self.criterion(preds_from_masked, labels_)
        cer_batch = 0.0

        # Log images and predictions
        if batch_idx < 10:
          print(f'Preds shape: {preds_.shape}')
          # Remove tokens 0, 1, 2 from preds_
          preds_[preds_ < 3] = 0
          print(f'Preds shape: {preds_.shape}')


          for i in range(images_masked.shape[0]):
            _label = labels[i].detach().cpu().numpy().tolist()
            _pred = preds_[i].detach().cpu().numpy().tolist()
            _pred_from_segmented = preds_from_segmented[i].detach().cpu().numpy().tolist()
            
            img_grid = torchvision.utils.make_grid(images_masked[i].detach(), normalize=False)
            image_ = torchvision.transforms.ToPILImage()(img_grid)

            # Image from segmenter
            img_grid_segmenter = torchvision.utils.make_grid(images_masked_segmenter[i].detach(), normalize=False)
            image_segmenter = torchvision.transforms.ToPILImage()(img_grid_segmenter)

            _label, _pred = self.decode(_label), self.decode(_pred)
            _pred_from_segmented = self.decode(_pred_from_segmented)
            _label_, _pred_, _pred_from_segmented_ = "", "", ""
            for l in range(len(_label)):
              if _label[l] == '<eos>' or _label[l] == '<sos>':
                break
              _label_ += _label[l]

            for l in range(len(_pred)):
              if _pred[l] == '<eos>' or _pred[l] == '<sos>':
                break
              _pred_ += _pred[l]
            
            for l in range(len(_pred_from_segmented)):
              if _pred_from_segmented[l] == '<eos>' or _pred_from_segmented[l] == '<sos>':
                break
              _pred_from_segmented_ += _pred_from_segmented[l]

            _label, _pred, _pred_from_segmented = _label_, _pred_, _pred_from_segmented_
            cer_from_segmented = self.train_cer.forward(_pred_from_segmented, _label)
            cer = self.train_cer.forward(_pred, _label)
            self._logger.experiment.log({"train/preds_from_masked": wandb.Image(image_, caption=f'Label: {_label} \n Pred: {_pred} \n CER: {cer} \n epoch: {self.current_epoch}')})
            self._logger.experiment.log({"train/preds_from_segmented": wandb.Image(image_segmenter, caption=f'Label: {_label} \n Pred: {_pred_from_segmented} \n CER: {cer_from_segmented} \n epoch: {self.current_epoch}')})

        self.metric_logger.log_train_step(loss, torch.tensor([0.0]))

        # update and log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        # self.log("train/acc", 0.0, on_step=True, on_epoch=True, prog_bar=True)
        # self.log("train/cer", cer, on_step=True, on_epoch=True, prog_bar=True)

        return loss 

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        self.metric_logger.log_train_metrics()
        self.metric_logger.update_epoch(self.current_epoch)
        self.metric_logger_minusc.update_epoch(self.current_epoch)
        self.train_cer_epoch = self.train_cer.compute()

        self.log("train/cer_epoch", self.train_cer_epoch, on_epoch=True, prog_bar=True)

        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, dataloader_idx: int = None) -> None:
        """Perform a single validation step on a batch of data from the validation set."""
        images, labels = batch
        labels = labels.permute(1, 0)
        labels = labels[:, 1:].clone().contiguous() # Shift all labels to the right
        labels = labels[:, :-1].clone().contiguous() # Remove last label (it is the <eos> token)
        # Convert 0, 1 and 2 to 0 (padding token) to 0 
        labels[labels < 3] = 0

        dataloader_idx = 0 if len(self.val_datasets) == 1 else dataloader_idx
        dataset = self.val_datasets[dataloader_idx]

        # Predict with segmenter and get the masked images
        images_masked = self.segmenter.net.predict_greedy(images).logits
        # Threshold the images (WARNING, CHANING THIS THRESHOLD MAY AFFECT THE PERFORMANCE OF THE MODEL)
        images_masked = images_masked > 0.5
        images_masked = images_masked.float()
        images_masked = images_masked.reshape(-1, 1, images.shape[-2], images.shape[-1])
        # images_masked_ = images_masked.flatten(0, 1)

        print(f'Images masked shape (after flatten): {images_masked.shape}')

        preds = self.net(images_masked)
        preds = preds.reshape(images.shape[0], -1, preds.shape[-1]).softmax(-1).argmax(-1)

        images_masked = images_masked.reshape(images.shape[0], -1, 1, images.shape[-2], images.shape[-1])

        total_cer_per_batch = 0.0
        print(f'---VALIDATION STEP----- ended')

        
        for i in range(images.shape[0]):
          # Remove tokens == 0 in labels
          _label = labels[i].detach().cpu().numpy().tolist()
          _pred = preds[i].detach().cpu().numpy().tolist()
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
          
          _label, _pred = _label_, _pred_

          # Calculate CER converting mayus to minus
          _label_minus = _label.lower()
          _pred_minus = _pred.lower()
          cer_minus = self.val_cer_minus.forward(_pred_minus, _label_minus)
          self.metric_logger_minusc.log_val_step_cer(_pred_minus, _label_minus, f'{dataset}_minusc')

          # Calculate CER
          cer = CER()(_pred, _label)
          if batch_idx < 15:
            img_grid = torchvision.utils.make_grid(images_masked[i].detach(), normalize=False)
            image_ = torchvision.transforms.ToPILImage()(img_grid)
            orig_image = torchvision.transforms.ToPILImage()(images[i].detach().cpu())
            self._logger.experiment.log({f'val/preds_{dataset}': wandb.Image(image_, caption=f'Label: {_label} \n Pred: {_pred} \n CER: {cer} \n CER minus: {cer_minus} \n epoch: {self.current_epoch}')})
            self._logger.experiment.log({f'val/original_image_{dataset}': wandb.Image(orig_image, caption=f'Label: {_label} \n Pred: {_pred} \n CER: {cer} \n CER minus: {cer_minus} \n epoch: {self.current_epoch}')})

          # Save images with torchvision to disk in outputs folder with prefix val_images_masked and val_images, epoch, batch_idx and i, prediction, label and cer separated by -
          # Add prefix with the run name
          run_name = self._logger.experiment.name
            
          # Create folder outputs/{dataset} if it does not exist
          if not os.path.exists(f'outputs/{dataset}'):
            os.makedirs(f'outputs/{dataset}')

          # Write in a file the predictions and labels at the end of a .txt file
          with open(f'outputs/{dataset}/val_preds_labels_{run_name}.txt', 'a') as f:
            f.write(f'Pred: {_pred} - Label: {_label} - CER: {cer} - CER_minus: {cer_minus} \n')


          # torchvision.utils.save_image(images_masked[i], f'outputs/{dataset}/val_images_masked_{self.current_epoch}_{batch_idx}_{i}_PRED:{_pred}_LABEL:{_label}_CER:{cer}.png')
          # torchvision.utils.save_image(images[i], f'outputs/{dataset}/val_images_{self.current_epoch}_{batch_idx}_{i}_PRED:{_pred}_LABEL:{_label}_CER:{cer}.png')
            
          
          self.metric_logger.log_val_step_cer(_pred, _label, dataset)
          
          total_cer_per_batch += cer

        print(f'Total CER per batch: {total_cer_per_batch/images.shape[0]}')

        # self.metric_logger.log_val_step_cer(preds_str, labels_str, dataset)
        

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."

        val_cer_epoch = self.metric_logger.log_val_metrics()
        print(f'val/cer_epoch: {val_cer_epoch}')
        self.log(f'val/cer_epoch', val_cer_epoch, sync_dist=True, prog_bar=True)
        
        # Log CER minusc
        val_cer_minus = self.metric_logger_minusc.log_val_metrics()
        self.log(f'val/cer_minusc', val_cer_minus, sync_dist=True, prog_bar=True)
        
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, dataloader_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        images, labels = batch
        labels = labels.permute(1, 0)
        labels = labels[:, 1:].clone().contiguous() # Shift all labels to the right
        labels = labels[:, :-1].clone().contiguous() # Remove last label (it is the <eos> token)
        # Convert 0, 1 and 2 to 0 (padding token) to 0 
        labels[labels < 3] = 0

        dataloader_idx = 0 if len(self.test_datasets) == 1 else dataloader_idx
        dataset = self.test_datasets[dataloader_idx]

        # Predict with segmenter and get the masked images
        images_masked = self.segmenter.net.predict_greedy(images).logits
        images_masked = images_masked.reshape(-1, 1, 64, 64)
        # images_masked_ = images_masked.flatten(0, 1)

        # print(f'Images masked shape (after flatten): {images_masked_.shape}')

        preds = self.net(images_masked)
        preds = preds.reshape(images.shape[0], -1, preds.shape[-1]).softmax(-1).argmax(-1)

        total_cer_per_batch = 0.0
        print(f'---TEST STEP----- ended')
        # Print labels and predictions
        for i in range(images.shape[0]):
          # Remove tokens == 0 in labels
          img_grid = torchvision.utils.make_grid(images_masked[i].detach(), normalize=False)
          image_ = torchvision.transforms.ToPILImage()(img_grid)

          _label = labels[i].detach().cpu().numpy().tolist()
          _pred = preds[i].detach().cpu().numpy().tolist()
          _label = self.decode(_label)
          _pred = self.decode(_pred)
          # Rempve '<sos>'from _pred and _label
          _label = ''.join([x for x in _label if x != '<sos>'])
          _pred = ''.join([x for x in _pred if x != '<sos>'])
          # Calculate CER
          cer = CER()(_pred, _label)
          self._logger.experiment.log({f'test/preds_{dataset}': wandb.Image(image_, caption=f'Label: {_label} \n Pred: {_pred} \n CER: {cer}')})
          self.metric_logger.log_test_step_cer(_label, _pred, dataset)
          
          total_cer_per_batch += cer

        print(f'Total CER per batch: {total_cer_per_batch/images.shape[0]}')



    def on_test_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        test_cer_epoch = self.metric_logger.log_test_metrics()
        print(f'test/cer_epoch: {test_cer_epoch}')
        self.log(f'test/cer_epoch', test_cer_epoch, sync_dist=True, prog_bar=True)



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
