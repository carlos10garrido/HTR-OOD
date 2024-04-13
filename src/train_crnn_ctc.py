import hydra
from typing import List
import pytorch_lightning as pl
import rootutils
import lightning as L
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig
from lightning.pytorch.loggers import Logger
from typing import Any, Dict, List, Optional, Tuple
from hydra.core.config_store import ConfigStore
import os

from omegaconf import OmegaConf


# import data_config as data_config
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.data_config import DatasetConfig, DataConfig
from src.data.htr_datamodule import HTRDataModule
from src.models.crnn_ctc_module import CRNN_CTC_Module

print(f'Importing modules...')

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

from src.utils.instantiators import instantiate_data_configs 

log = RankedLogger(__name__, rank_zero_only=True)

@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, float], Dict[str, Any]]:

    # Set all seeds for reproducibility
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # Init wandb logger and project
    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    # Instantiating datamodule
    log.info("Instantiating DataModule...")
    data_configs = instantiate_data_configs(cfg.get("data"))
    log.info(f'TRAIN, VAL, TEST DATA_CONFIGS INSTANTIATED: {data_configs}')
    log.info(f'TRAIN: {data_configs["train"]}')
    log.info(f'VAL: {data_configs["val"]}')
    log.info(f'TEST: {data_configs["test"]}')

    # Update wandb logger with data config
    logger[0].experiment.config.update(
        OmegaConf.to_object(cfg.get("data").get("train"))
    )
        # cfg.get("data").get("train"))

    # Init data module
    log.info("Instantiating DataModule...")
    datamodule: LightningDataModule = HTRDataModule(
        train_config=data_configs["train"],
        val_config=data_configs["val"],
        test_config=data_configs["test"],
        seed=cfg.get("seed"),
    )
    log.info(f'DATAMODULE INSTANTIATED: {datamodule}')

    # Setup data module
    log.info("Setting up DataModule TRAIN AND VAL...")
    datamodule.setup(stage="fit")

    log.info("Setting up DataModule TEST...")
    datamodule.setup(stage="test")

    print(f'Instantiating model...')
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    print(f'MODEL INSTANTIATED: {model}')

    # Update wandb logger with model config
    logger[0].experiment.config.update(
        OmegaConf.to_object(cfg.model)
    )

    # logger[0].experiment.config.update(cfg.model)

    # Predict on test set
    log.info("Predicting on test set...")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, logger=logger, callbacks=instantiate_callbacks(cfg.get("callbacks"))
    )

    # Load a checkpoint if provided from callbacks.model_checkpoint filename
    # ckpt_path = cfg.callbacks.model_checkpoint.filename if cfg.callbacks.model_checkpoint.filename else None

    # Load a checkpoint if provided from callbacks.model_checkpoint filename
    # ckpt_path = cfg.callbacks.model_checkpoint.dirpath + cfg.callbacks.model_checkpoint.filename + '.ckpt' if cfg.callbacks.model_checkpoint.filename else None
    # # if ckpt_path exists, load the model from the checkpoint
    # if ckpt_path is not None and os.path.exists(ckpt_path):
    #     print(f'CHECKPOINT PATH EXISTS: {ckpt_path}')
    #     print(f'MODEL WILL BE LOADED FROM CHECKPOINT: {model}')
    # else:
    #     print(f'CHECKPOINT PATH DOES NOT EXIST: {ckpt_path}')
    #     print(f'MODEL WILL BE TRAINED FROM SCRATCH: {model}')
    #     ckpt_path = None

    # Load a checkpoint if provided from callbacks.model_checkpoint filename
    ckpt_path = cfg.callbacks.model_checkpoint_base.dirpath + cfg.callbacks.model_checkpoint_base.filename + '.ckpt' if cfg.callbacks.model_checkpoint.filename else None
    # if ckpt_path exists, load the model from the checkpoint
    if ckpt_path is not None and os.path.exists(ckpt_path):
        print(f'CHECKPOINT PATH EXISTS: {ckpt_path}')
        print(f'MODEL WILL BE LOADED FROM CHECKPOINT: {model}')
    else:
        print(f'CHECKPOINT PATH DOES NOT EXIST: {ckpt_path}')
        print(f'MODEL WILL BE TRAINED FROM SCRATCH: {model}')
        ckpt_path = None
      

    model = CRNN_CTC_Module.load_from_checkpoint(ckpt_path) if ckpt_path is not None else model
    # Train the model
    trainer.fit(model, datamodule.train_dataloader(), datamodule.val_dataloader())
    

    # Predict iterating over images
    # breakpoint()
    results = trainer.test(model, datamodule.test_dataloader())
    print(f'PREDICTIONS: {results}')



@hydra.main(version_base="1.3", config_path="../configs", config_name="train_htr.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """

    print(f'Main for training HTR models for HTR!')
    # train the model
    extras(cfg)

    _ = train(cfg)

    return None


if __name__ == "__main__":
    main()
