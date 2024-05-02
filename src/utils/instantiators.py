from typing import List

import hydra
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from src.data.data_config import DataConfig, DatasetConfig
from lightning import LightningDataModule

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

def instantiate_data_config(data_cfg: DictConfig) -> DataConfig:
    """Instantiates data config.

    :param data_cfg: A DictConfig object containing data configurations.
    :return: A DictConfig object containing data configurations.
    """
    if not data_cfg:
        log.warning("No data config found! Skipping...")
        return data_cfg

    if not isinstance(data_cfg, DictConfig):
        raise TypeError("Data config must be a DictConfig!")
    
    log.info(f"Instantiating ALL data config <{data_cfg}>")
    log.info(f"Instantiating data config <{data_cfg._target_}>")
    # Instantiate each inside dataset config if it exists
    if "datasets" in data_cfg:
        datasets: dict = {}
        for dataset in data_cfg.datasets:
            log.info(f"Instantiating dataset {dataset} <{data_cfg.datasets[dataset]._target_}>")
            if "_target_" in data_cfg.datasets[dataset]:
                inst_dataset = hydra.utils.instantiate(data_cfg.datasets[dataset], _recursive_=False)
                datasets[dataset] = inst_dataset
        data_cfg.datasets = datasets
        data_config = hydra.utils.instantiate(data_cfg, _convert_="partial", _recursive_=True)
    else:
        log.warning("No data config found! Skipping...")
        raise ValueError("No data config found!")
        
    return data_config


def instantiate_data_configs(data_cfg: DictConfig) -> dict:
    """Instantiates data config.

    :param data_cfg: A DictConfig object containing data configurations.
    :return: A DictConfig object containing data configurations.
    """
    if not data_cfg:
        log.warning("No data config found! Skipping...")
        return data_cfg

    if not isinstance(data_cfg, DictConfig):
        raise TypeError("Data config must be a DictConfig!")
    

    log.info(f"Instantiating data config <{data_cfg}>")

    data_configs = {}
    
    print(f'Stages in data config: {data_cfg.keys()}')
    # Instantiate data config for training, validation and testing
    for stage in data_cfg.keys():
        if stage in data_cfg:
            data_configs[stage] = instantiate_data_config(data_cfg[stage][stage+"_config"])

    return data_configs

def instantiate_sub_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates sub callbacks from config.

    :param callbacks_cfg: A DictConfig object containing callback configurations.
    :return: A list of instantiated callbacks.
    """
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating subcallback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config.

    :param callbacks_cfg: A DictConfig object containing callback configurations.
    :return: A list of instantiated callbacks.
    """
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    # Instantiate sub callbacks.heldout_targets callbacks
    if "heldout_targets" in callbacks_cfg:
        callbacks.extend(instantiate_sub_callbacks(callbacks_cfg.heldout_targets))

    return callbacks


def instantiate_tokenizers(tokenizer_cfg: DictConfig) -> LightningDataModule:
    """Instantiates tokenizer from config.

    :param tokenizer_cfg: A DictConfig object containing tokenizer configurations.
    :return: An instantiated tokenizer.
    """
    if not tokenizer_cfg:
        log.warning("No tokenizer config found! Skipping...")
        return tokenizer_cfg

    if not isinstance(tokenizer_cfg, DictConfig):
        raise TypeError("Tokenizer config must be a DictConfig!")

    print(f"TOKENIZER CFG: {tokenizer_cfg}")
    print(f"TOKENIZER CFG KEYS: {tokenizer_cfg.keys()}")

    log.info(f"Instantiating tokenizer <{tokenizer_cfg._target_}>")
    return hydra.utils.instantiate(tokenizer_cfg)


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config.

    :param logger_cfg: A DictConfig object containing logger configurations.
    :return: A list of instantiated loggers.
    """
    logger: List[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger
