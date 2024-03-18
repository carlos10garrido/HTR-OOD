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

# import global variables encode and decode from htr_data_module.py
from src.data.htr_datamodule import encode, decode

class HTRResnetCharCounter(LightningModule):
  
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        _logger: Any,
        datasets: dict,
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

        # breakpoint()

        # Create metrics per dataset
        # Training metrics
        self.train_loss = MeanMetric()
        # Validation metrics
        self.val_mae = {}
        self.min_val_mae = {}
        for dataset in self.val_datasets:
            self.val_mae[dataset] = MAE()
            self.min_val_mae[dataset] = MinMetric()

        # Test metrics
        self.test_mae = {}
        self.min_test_mae = {}
        for dataset in self.test_datasets:
            self.test_mae[dataset] = MAE()
            self.min_test_mae[dataset] = MinMetric()

        self.net = net

        # loss function for regressing the number of characters in an image
        self.criterion = torch.nn.MSELoss()

        self.encode = encode
        self.decode = decode

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

        y = batch[1].permute(1, 0)
        y[y == 1] = -100
        labels = y[:, 1:].clone().contiguous() # Shift all labels to the right

        print(f'labels before counting: {labels}')

        # Labels are the number of character != 1 (padding token) and <eos> (2) token is not included
        labels = (labels > 2).sum(dim=1)
        labels = labels.squeeze(-1).float()

        print(f'labels after counting: {labels}')

        preds = self.net(images)
        preds = preds.squeeze(-1)

        print(f'preds {preds}')
        
        print(f'preds shape: {preds.shape} preds device: {preds.device}')
        print(f'labels shape: {labels.shape} labels device: {labels.device}')

        loss = self.criterion(preds, labels)

        # Calculate accuracy
        mae = MAE().to(preds.device)
        mae = mae(preds, labels)

        self.metric_logger.log_train_step(loss, mae)

        # update and log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/mae", mae, on_step=True, on_epoch=True, prog_bar=True)

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

        images = batch[0]
        y = batch[1].permute(1, 0)
        # y[y == 1] = -100
        labels = y[:, 1:].clone().contiguous() # Shift all labels to the right

        print(f'labels before counting: {labels}')

        # Labels are the number of character != 1 (padding token) and <eos> (2) token is not included
        labels = (labels > 2).sum(dim=1)
        labels = labels.squeeze(-1).float()

        print(f'labels after counting: {labels}')
        
        if self.current_epoch == 0 and self.global_step <= 1:
          str_train_datasets = f'val_' + ', '.join(self.train_datasets)
          self.metric_logger.log_images(images, str_train_datasets)
        
        # loss, preds, targets = self.model_step(batch)
        preds = self.net(images)
        preds = preds.squeeze(-1)

        print(f'preds {preds}')

        print(f'preds shape: {preds.shape} preds device: {preds.device}')
        print(f'labels shape: {labels.shape} labels device: {labels.device}')

        # Calculate MAE for dataset
        print(f'Calculating MAE for dataset: {dataset}')
        print(self.val_mae[dataset].device)
        self.val_mae[dataset].to(preds.device)(preds, labels)
        
        

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        mean_val_mae = 0.0

        for dataset in self.val_datasets:
            val_mae = self.val_mae[dataset].compute()
            mean_val_mae += val_mae
            self.min_val_mae[dataset](val_mae)
            self.log(f"val/mae_{dataset}", val_mae, sync_dist=True, prog_bar=True)
            self.log(f"val/min_val_mae_{dataset}", self.min_val_mae[dataset].compute(), sync_dist=True, prog_bar=True)

        mean_val_mae /= len(self.val_datasets)

        self.log(f'val/cer_epoch', mean_val_mae, sync_dist=True, prog_bar=True)
        self.metric_logger.log_val_metrics()
        


    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, dataloader_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        dataloader_idx = 0 if len(self.test_datasets) == 1 else dataloader_idx
        dataset = self.test_datasets[dataloader_idx]
        # print(f'Calculating test CER for each dataset for dataloader_idx: {dataloader_idx}. Dataset: {dataset}')

        images = batch[0]
        y = batch[1].permute(1, 0)
        # y[y == 1] = -100
        labels = y[:, 1:].clone().contiguous() # Shift all labels to the right

        # Labels are the number of character != 1 (padding token) and <eos> (2) token is not included
        labels = (labels > 2).sum(dim=1)
        labels = labels.squeeze(-1).float()

        # loss, preds, targets = self.model_step(batch)
        preds = self.net(images)
        preds = preds.squeeze(-1)

        # Calculate MAE for dataset
        self.test_mae[dataset](preds, labels)



    def on_test_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        mean_test_mae = 0.0

        for dataset in self.test_datasets:
            test_mae = self.test_mae[dataset].compute()
            mean_test_mae += test_mae
            self.min_test_mae[dataset](test_mae)
            self.log(f"test/mae_{dataset}", test_mae, sync_dist=True, prog_bar=True)
            self.log(f"test/min_test_mae_{dataset}", self.min_test_mae[dataset].compute(), sync_dist=True, prog_bar=True)

        mean_test_mae /= len(self.test_datasets)

        self.log(f'test/mae_epoch', mean_test_mae, sync_dist=True, prog_bar=True)
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
    _ = HTRResnetCharCounter(None, None, None, None)
