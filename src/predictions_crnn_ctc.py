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
import torch.nn as nn
import torchmetrics

from omegaconf import OmegaConf

# Set precision for torch to bf16
import torch
import torchvision.transforms as v2
# torch.set_default_tensor_type(torch.BFloat16Tensor)

# import data_config as data_config
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.data_config import DatasetConfig, DataConfig
from src.data.htr_datamodule import HTRDataModule
from src.models.crnn_ctc_module import CRNN_CTC_Module

from src.utils.instantiators import instantiate_data_configs, instantiate_tokenizers

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

import src

log = RankedLogger(__name__, rank_zero_only=True)

from src.data.htr_datamodule import HTRDataModule
from src.data.htr_datamodule import HTRDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from tqdm import tqdm


class Encoder(nn.Module):
  # Convolutional Encoder for HTR
  def __init__(self):
    super(Encoder, self).__init__()
    self.conv1 = nn.Conv2d(3, 16, kernel_size=(3,3), padding=1)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=(3,3), padding=1)
    self.conv3 = nn.Conv2d(32, 64, kernel_size=(3,3), padding=1)
    self.conv4 = nn.Conv2d(64, 128, kernel_size=(3,3), padding=1)
    self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
    self.relu = nn.LeakyReLU()
    self.fc = nn.Linear(128*4*64, 512)
  
  def forward(self, x):
    # print(f'Encoder input shape: {x.shape}')
    x = self.relu(self.conv1(x))
    x = self.pool(x)
    # print(x.shape)
    x = self.relu(self.conv2(x))
    x = self.pool(x)
    # print(x.shape)
    x = self.relu(self.conv3(x))
    x = self.pool(x)
    # print(x.shape)
    x = self.relu(self.conv4(x))
    x = self.pool(x)
    # print(x.shape)
    x = x.view(x.size(0), -1) # Flatten operation
    # print(x.shape)
    x = self.fc(x)
    # print(x.shape)
    return x
  
  
class Decoder(nn.Module):
  # Convolutional Decoder for HTR
  def __init__(self):
    super(Decoder, self).__init__()
    self.fc = nn.Linear(512, 128*4*64)
    # Deconvolutions (transposed convolutions)
    self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=(3,3), stride=(2,2), padding=0)
    self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=(3,3), stride=(2,2), padding=1)
    self.deconv3 = nn.ConvTranspose2d(32, 16, kernel_size=(3,3), stride=(2,2), padding=1)
    self.deconv4 = nn.ConvTranspose2d(16, 3, kernel_size=(3,3), stride=(2,2), padding=1, output_padding=0)
    self.relu = nn.LeakyReLU()
    self.sigmoid = nn.Sigmoid()
    
  def forward(self, x):
    # print(f'Decoder input shape: {x.shape}')
    x = self.fc(x)
    # print(f'x.shape: {x.shape}')
    x = x.view(x.size(0), 128, 4, 64)
    # print(f'x.shape (after view): {x.shape}')
    x = self.relu(self.deconv1(x))
    # print(f'x.shape: {x.shape}')
    x = self.relu(self.deconv2(x))
    # print(f'x.shape: {x.shape}')
    x = self.relu(self.deconv3(x))
    # print(f'x.shape: {x.shape}')
    x = self.relu(self.deconv4(x))
    # print(f'x.shape: {x.shape}')
    
    x = torch.narrow(x, 2, 0, 64)
    x = torch.narrow(x, 3, 0, 1024)
    x = self.sigmoid(x)
    
    return x
  
  
class AE(nn.Module):
  # Complete Autoencoder (AE) trained to reconstruct images
  def __init__(self):
    super(AE, self).__init__()
    self.encoder = Encoder()
    self.decoder = Decoder()
    
  def forward(self, x):
    h = self.encoder(x)
    x = self.decoder(h)
    return x
  
  
def read_dataset(images_path, sequences_path, split_path, read_data, tokenizer, batch_size, img_size, transform=v2.Compose([v2.ToTensor()])):
  with open(split_path, "r") as f:
    setfiles = f.read().splitlines()
  
  images_paths, words = read_data(images_path, sequences_path, setfiles)
  htr_dataset = HTRDataset(images_paths, words, binarize=True, transform=transform)

  # Create dataloader for iam
  dl = torch.utils.data.DataLoader(
    htr_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=4, 
    pin_memory=True,
    collate_fn=lambda batch: src.data.data_utils.collate_fn(
      batch,
      # img_size=(128,1024),
      img_size=img_size,
      text_transform=tokenizer.prepare_text
    )
  )
  return dl

def calculate_cer(htr_output, target):
  torchmetrics.text.CharacterErrorRate()
  cer = cer(htr_output, target)
  return cer

def calculate_ppl(model, output, target, criterion):
  total_ppl = 0.0
  label = target[:, 1:].clone().contiguous()
  output = output[:, :-1].contiguous()
  loss = torch.log(criterion(output.reshape(-1, output.size(-1)), label.reshape(-1)))
  total_ppl += torch.exp(loss)

  return total_ppl


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

    # Instantiating tokenizer
    tokenizer = instantiate_tokenizers(cfg.get("tokenizer"))

    print(f'TOKENIZER: {tokenizer}')
    

    # Init data module
    log.info("Instantiating DataModule...")
    datamodule: LightningDataModule = HTRDataModule(
        train_config=data_configs["train"],
        val_config=data_configs["val"],
        test_config=data_configs["test"],
        tokenizer=tokenizer,
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
    # ckpt_path = cfg.callbacks.model_checkpoint_base.dirpath + cfg.callbacks.model_checkpoint_base.filename + '.ckpt' if cfg.callbacks.model_checkpoint.filename else None
    
    # Load from a pretrained_checkpoint
    ckpt_path = cfg.callbacks.model_checkpoint_base.dirpath + cfg.get("pretrained_checkpoint") + '.ckpt' if cfg.get("pretrained_checkpoint") else None
    
    
    
    # if ckpt_path exists, load the model from the checkpoint
    if ckpt_path is not None and os.path.exists(ckpt_path):
        print(f'CHECKPOINT PATH EXISTS: {ckpt_path}')
        print(f'MODEL WILL BE LOADED FROM CHECKPOINT: {model}')
    else:
        print(f'CHECKPOINT PATH DOES NOT EXIST: {ckpt_path}')
        print(f'MODEL WILL BE TRAINED FROM SCRATCH: {model}')
        ckpt_path = None
      
    model_htr = CRNN_CTC_Module.load_from_checkpoint(ckpt_path, datasets=cfg.get("data"), tokenizer=tokenizer) if ckpt_path is not None else model

    # Create autoencoder to get reconstructino errors
    model_AE = AE()
    if os.path.exists('checkpoints/best_ae_iam.ckpt'):
      print(f'CHECKPOINT PATH EXISTS: checkpoints/best_ae_iam.ckpt')
      print(f'MODEL WILL BE LOADED FROM CHECKPOINT: {model_AE}')
      model_AE.load_state_dict(torch.load('checkpoints/best_ae_iam.ckpt'))
    
    print(model_AE)
    
    # Load gpt2-small for predict the perplexity of the target test
    
    from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config

    config = GPT2Config(
      vocab_size=94,
      bos_token_id=0,
      eos_token_id=1, 
      n_positions=1024,
      n_layers=4,
      n_head=4,
      n_embd=128,
      dropout=0.4,
    )
    model_lm = GPT2LMHeadModel(config)
    print(model_lm)


    # Print number of parameters
    print(model_lm.num_parameters()) # 85.9M
    
    # Load gpt2 model_lm
    # # Save the model to disk
    # torch.save(model.state_dict(), "../checkpoints/lm_model_iam.ckpt")
    model_lm.load_state_dict(torch.load("checkpoints/lm_model_iam.ckpt"))
    print(model_lm)
    
    # Assure the same seed for reproducibility
    torch.manual_seed(42)
    dataloaders_tests = dict()
    dataloaders_tests["IAM"] = read_dataset("./data/htr_datasets/IAM/IAM_lines/", "./data/htr_datasets/IAM/IAM_xml/", "./data/htr_datasets/IAM/splits/test.txt", src.data.data_utils.read_data_IAM, tokenizer, 16, (128, 1024))
    dataloaders_tests["RIMES"] = read_dataset("./data/htr_datasets/RIMES/RIMES-2011-Lines/Images/", "./data/htr_datasets/RIMES/RIMES-2011-Lines/Transcriptions/", "./data/htr_datasets/RIMES/RIMES-2011-Lines/Sets/test.txt", src.data.data_utils.read_data_rimes, tokenizer, 16, (128, 1024))
    dataloaders_tests["Saint_Gall"] = read_dataset("./data/htr_datasets/saint_gall/saintgalldb-v1.0/data/line_images_normalized/", "./data/htr_datasets/saint_gall/saintgalldb-v1.0/ground_truth/", "./data/htr_datasets/saint_gall/saintgalldb-v1.0/sets/test.txt", src.data.data_utils.read_data_saint_gall, tokenizer, 16, (128, 1024))
    dataloaders_tests["Bentham"] = read_dataset("./data/htr_datasets/bentham/BenthamDatasetR0-GT/Images/Lines/", "./data/htr_datasets/bentham/BenthamDatasetR0-GT/Transcriptions/", "./data/htr_datasets/bentham/BenthamDatasetR0-GT/Partitions/test.txt", src.data.data_utils.read_data_bentham, tokenizer, 16, (128, 1024))
    dataloaders_tests["Washington"] = read_dataset("./data/htr_datasets/washington/washingtondb-v1.0/data/line_images_normalized/", "./data/htr_datasets/washington/washingtondb-v1.0/ground_truth/", "./data/htr_datasets/washington/washingtondb-v1.0/sets/cv1/test.txt", src.data.data_utils.read_data_washington, tokenizer, 16, (128, 1024))
    dataloaders_tests["ICFHR_2016"] = read_dataset("./data/htr_datasets/icfhr_2016/lines/", "./data/htr_datasets/icfhr_2016/transcriptions/", "./data/htr_datasets/icfhr_2016/partitions/test.txt", src.data.data_utils.read_data_icfhr_2016, tokenizer, 16, (128, 1024))
    dataloaders_tests["Rodrigo"] = read_dataset("./data/htr_datasets/rodrigo/Rodrigo corpus 1.0.0/images/", "./data/htr_datasets/rodrigo/Rodrigo corpus 1.0.0/text/", "./data/htr_datasets/rodrigo/Rodrigo corpus 1.0.0/partitions/test.txt", src.data.data_utils.read_data_rodrigo, tokenizer, 16, (128, 1024))
    
    # Ass
    # Same dataloaders for the autoencoder but different image size
    dataloaders_AE = dict()
    dataloaders_AE["IAM"] = read_dataset("./data/htr_datasets/IAM/IAM_lines/", "./data/htr_datasets/IAM/IAM_xml/", "./data/htr_datasets/IAM/splits/test.txt", src.data.data_utils.read_data_IAM, tokenizer, 16, (64, 1024))
    dataloaders_AE["RIMES"] = read_dataset("./data/htr_datasets/RIMES/RIMES-2011-Lines/Images/", "./data/htr_datasets/RIMES/RIMES-2011-Lines/Transcriptions/", "./data/htr_datasets/RIMES/RIMES-2011-Lines/Sets/test.txt", src.data.data_utils.read_data_rimes, tokenizer, 16, (64, 1024))
    dataloaders_AE["Saint_Gall"] = read_dataset("./data/htr_datasets/saint_gall/saintgalldb-v1.0/data/line_images_normalized/", "./data/htr_datasets/saint_gall/saintgalldb-v1.0/ground_truth/", "./data/htr_datasets/saint_gall/saintgalldb-v1.0/sets/test.txt", src.data.data_utils.read_data_saint_gall, tokenizer, 16, (64, 1024))
    dataloaders_AE["Bentham"] = read_dataset("./data/htr_datasets/bentham/BenthamDatasetR0-GT/Images/Lines/", "./data/htr_datasets/bentham/BenthamDatasetR0-GT/Transcriptions/", "./data/htr_datasets/bentham/BenthamDatasetR0-GT/Partitions/test.txt", src.data.data_utils.read_data_bentham, tokenizer, 16, (64, 1024))
    dataloaders_AE["Washington"] = read_dataset("./data/htr_datasets/washington/washingtondb-v1.0/data/line_images_normalized/", "./data/htr_datasets/washington/washingtondb-v1.0/ground_truth/", "./data/htr_datasets/washington/washingtondb-v1.0/sets/cv1/test.txt", src.data.data_utils.read_data_washington, tokenizer, 16, (64, 1024))
    dataloaders_AE["ICFHR_2016"] = read_dataset("./data/htr_datasets/icfhr_2016/lines/", "./data/htr_datasets/icfhr_2016/transcriptions/", "./data/htr_datasets/icfhr_2016/partitions/test.txt", src.data.data_utils.read_data_icfhr_2016, tokenizer, 16, (64, 1024))
    dataloaders_AE["Rodrigo"] = read_dataset("./data/htr_datasets/rodrigo/Rodrigo corpus 1.0.0/images/", "./data/htr_datasets/rodrigo/Rodrigo corpus 1.0.0/text/", "./data/htr_datasets/rodrigo/Rodrigo corpus 1.0.0/partitions/test.txt", src.data.data_utils.read_data_rodrigo, tokenizer, 16, (64, 1024))
                                               
    
    criterion_lm = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id, reduction='sum')
    criterion_rec = nn.MSELoss(reduction='sum')

    # WIP: seguir mañana desarrollando para ir imagen por imagen guardando CER, ppl y err_rec
    breakpoint()  
    model.eval()
    model.to(device)
    model_AE.eval()
    model_AE.to(device)
    model_lm.eval()
    model_lm.to(device)
    with torch.no_grad():
      errors_datasets = dict()
      for name in dataloaders_tests.keys():
        dl, dl_AE = dataloaders_tests[name], dataloaders_AE[name]
        print(f'Processing dataset: {name}')
        errors_datasets[name] = []
        for i, (batch, batch_AE) in enumerate(tqdm(zip(dl, dl_AE))):
          images, targets, _ = batch
          images_AE, targets_AE, _ = batch_AE
          targets = targets.transpose(1,0)
          targets_AE = targets_AE.transpose(1,0)
          for j, (image, target, image_AE, target_AE) in enumerate(zip(images, targets, images_AE, targets_AE)):
            # Assert if target != target_AE
            # assert target != target_AE
            image = image.to(device).unsqueeze(0)
            target = target.to(device).unsqueeze(0)
            image_AE = image_AE.to(device).unsqueeze(0)
            target_AE = target_AE.to(device).unsqueeze(0)
            _image = image_AE
            # _image = v2.Resize((64,1024))(image) #(model(images) > 0.5).float()
            output_rec_error = criterion_rec((model_AE(_image).to(device) > 0.5).float(), _image)
            # Get the prediction of the HTR model
            _pred = model_htr(image).log_softmax(-1).argmax(-1)
            _pred = torch.unique_consecutive(_pred.detach()).cpu().numpy().tolist()
            _pred = [idx for idx in _pred if idx != tokenizer.vocab_size] # Remove blank token
            _label = target[0].cpu().numpy().tolist()
            _pred, _label = tokenizer.detokenize(_pred), tokenizer.detokenize(_label)
            # Calculate CER
            cer_image = torchmetrics.text.CharErrorRate()(_pred, _label) #calculate_cer(_pred, target)
            
            # Calculate perplexity
            output_lm = model_lm(target).logits
            ppl_seq = calculate_ppl(model_lm, output_lm, target, criterion_lm)
            
            errors_datasets[name].extend([(output_rec_error.item(), ppl_seq.item(), cer_image.item() * 100, _pred, _label)])
            
      print(errors_datasets)
      
    # Write all the results to a csv file with columns: dataset, image_id, CER, PPL, Reconstruction Error
    import csv
    breakpoint()
    with open('results/all_errors_indiv.csv', 'w') as f:
      writer = csv.writer(f)
      writer.writerow(["dataset", "image_id", "Reconstruction Error", "PPL",  "CER", "Prediction", "Label"])
      for name, errors in errors_datasets.items():
        for i, error in enumerate(errors):
          writer.writerow([name, i, error[0], error[1], error[2], error[3], error[4]])
      
    
    


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
