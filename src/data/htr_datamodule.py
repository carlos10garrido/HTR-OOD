# Datamodules for each Dataset
import os
import numpy as np
import torch
import pytorch_lightning as pl
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, RandomSampler
import xml.etree.ElementTree as ET
from torch.nn.utils.rnn import pad_sequence
from typing import Iterable, List
import json
from torchvision.transforms import v2
from fontTools.ttLib import TTFont
from fontTools.unicode import Unicode
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Any, Dict, List, Optional, Tuple
from lightning import LightningDataModule
from datasets import load_dataset

# Import opencv for binarization
import cv2

import torchvision
import sklearn
from sklearn.cluster import KMeans
from unidecode import unidecode


# Import data_config
from src.data.data_config import DataConfig, DatasetConfig, SynthDatasetConfig, RandomSynthDatasetConfig

from src.utils import pylogger
log = pylogger.RankedLogger(__name__, rank_zero_only=True)

from src.data.data_utils import (
    tensor_transform,
    sequential_transforms,
    tokenize,
    collate_fn,
    has_glyph,
    generate_image,
    read_htr_fonts,
    prepare_esposalles,
    prepare_saint_gall,
)

def select_printable_font(sequence, fonts):
    font = np.random.choice(fonts)
    can_generate = False
    while can_generate is False:
      for c in sequence:
        if has_glyph(font, str(c)) is False:
          font = np.random.choice(fonts)
          # print(f'Selecting another font for generating {sequence}!. Cannot generate {c}')
          break
      can_generate = True

    return font

class HTRDatasetSynthRandom(Dataset):
    def __init__(self, vocab, total_words, max_len, fonts, transform=None):
        self.vocab = vocab
        self.max_len = max_len
        self.fonts = read_htr_fonts(fonts) # Read fonts
        self.total_words = total_words
        self.transform = transform
        self.prob_vocab = None
        # Set prob_vocab to [a-z A-Z] = 2 * [rest of vocab] if character is in [a-z A-Z]
        self.prob_vocab = np.ones(len(self.vocab))
        for idx, char in enumerate(self.vocab):
            if char in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ":
                self.prob_vocab[idx] = 2
        self.prob_vocab = self.prob_vocab / np.sum(self.prob_vocab)


    def __len__(self):
        return self.total_words #

    def generate_random_word(self, max_len):
        word = ""
        # Set a x2 higher probability to select from [a-z] and [A-Z] than from the rest of the vocab
        chars = np.random.choice(len(self.vocab), size=np.random.randint(1, max_len), p=self.prob_vocab)
        # chars = np.random.randint(len(self.vocab), size=np.random.randint(1, max_len), p=self.prob_vocab)
        for char in chars:
            word += self.vocab[char]
        return word

    def select_printable_font(self, sequence):
        font = np.random.choice(self.fonts)
        can_generate = False
        while can_generate is False:
            for c in sequence:
                if has_glyph(font, str(c)) is False:
                    font = np.random.choice(self.fonts)
                    # print(f'Selecting another font for generating {sequence}!. Cannot generate {c}')
                    break
            can_generate = True

        return font

    
    def __getitem__(self, idx):
        sequence = self.generate_random_word(self.max_len)
        generated = False # Flag to check if image is generated correctly, if not, generate again
        counter_trials = 5

        text_color = tuple(np.random.randint(0, 256, size=(1,)))
        text_color = (text_color[0], text_color[0], text_color[0])
        background_colors = tuple(np.random.randint(180, 256, size=(1,)))
        background_colors = (background_colors[0], background_colors[0], background_colors[0])

        # Generate image
        while not generated:
            try:
              font = self.select_printable_font(sequence)
              image = generate_image(sequence, font, background_colors, text_color)
              generated = True
            except Exception as e:
              # print(f'Exception {e} while generating image with word {sequence} and font {font}.')
              # sequence = self.generate_random_word(self.max_len)
              font = self.select_printable_font(sequence)
              counter_trials -= 1
              if counter_trials == 0:
                  # print(f'Cannot generate image for word {sequence}. Generating other word...')
                  sequence = self.generate_random_word(self.max_len)
                  counter_trials = 5

        if self.transform:
            image = self.transform(image)

        return image, sequence

class HTRDatasetSynth(Dataset):
    def __init__(self, sequences, sequences_distr, fonts, transform=None):
        self.sequences = sequences
        # self.sequences_distr = sequences_distr
        self.fonts = read_htr_fonts(fonts)
        self.transform = transform

    def __len__(self):
        return len(self.sequences)

    def crop_sequence(self, sequence):
      if len(sequence) > 50:
        start = np.random.randint(0, len(sequence) - 50)
        end = start + np.random.randint(1, 50)

        while sequence[start] != ' ' and start > 0: # Between a character
          start -= 1

        while sequence[end] != ' ' and end < len(sequence)-1:
          end += 1

        return sequence[start+1:end]
    
      return sequence

    def __getitem__(self, idx):
        # Get sequence and read image
        sequence = self.sequences[idx] # Generate other sequence if len(sequence) == 0:
        # print(f'Sequence: {sequence} to generate image')
        if len(sequence) == 0:
            # print(f'sequence {sequence} has length 0. Generating other sequence...')
            sequence = np.random.choice(self.sequences)

        if len(sequence) > 50:
            sequence = self.crop_sequence(sequence)
            # print(f'Sequence too long. Selecting window of 50 characters: {sequence}')

        sequence = sequence.replace("\n", "")
        sequence = sequence.strip()
        sequence = unidecode(sequence)

            # print(f'Sequence too long. Selecting window of 50 characters: {sequence}')
        
        # font = np.random.choice(self.fonts)
        font = select_printable_font(sequence, self.fonts)
        generated = False # Flag to check if image is generated correctly, if not, generate again

        # text_color = tuple(np.random.randint(0, 256, size=(1,)))
        # text_color = (text_color[0], text_color[0], text_color[0])
        # background_colors = tuple(np.random.randint(180, 256, size=(1,)))
        # background_colors = (background_colors[0], background_colors[0], background_colors[0])
        text_color = (0,0,0)
        background_colors = (255,255,255)

        while not generated:
            try:
                image = generate_image(sequence, font, background_colors, text_color)
                # print(f'GENERATED IMAGE WITH SEQUENCE {sequence} AND FONT {font}')
                if self.transform is not None:
                    image = self.transform(image)

                generated = True
            except Exception as e:
                # print(f'Exception {e} while generating image with sequence {sequence} and font {font}.')
                sequence = np.random.choice(self.sequences)
                sequence = sequence.replace("\n", "")
                sequence = unidecode(sequence)
                if len(sequence) > 50:
                  sequence = self.crop_sequence(sequence)

                font = np.random.choice(self.fonts)


        # print(f'GENERATED IMAGE WITH SEQUENCE {sequence} AND FONT {font}')

        # Binarize image using Opencv
        if len(image.shape) == 3:
          # Convert to numpy array
          # image = image.permute(1, 2, 0).numpy()
          # print(f'Image shape: {image.shape}')
          # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
          # Binarize image with opencv Otsu algorithm
          # _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
          # Convert to grayscale image using torchvision
          image = torchvision.transforms.Grayscale()(image)


                
        return image, sequence


class HTRDataset(Dataset):
    def __init__(self, paths_images, words, binarize=True, transform=None):
        self.paths_images = paths_images
        self.words = words
        self.transform = transform
        self.binarize = binarize

    def __len__(self):
        return len(self.paths_images)

    def binarize_image(self, image, channels=3):
      image = image.numpy()
      image = image/image.reshape(-1).max()

      # Change Nan values to 0
      image[np.isnan(image)] = 0.0

      try:
        kmeans_image = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(image.reshape(channels, -1).transpose(1, 0))
        image_bin = kmeans_image.predict(image.reshape(channels, -1).transpose(1, 0))
        vector_diff = kmeans_image.cluster_centers_[0] - kmeans_image.cluster_centers_[1]
        image_bin = image_bin.reshape(image.shape[-2], image.shape[-1])

        if np.sum(vector_diff) > 0.0: # Cluster 0 assignations should be 1 and Cluster 1 assignations should be 0
          image_bin = 1 - image_bin

        image_bin = image_bin.reshape(1, image.shape[-2], image.shape[-1])
      except Exception as e:
        print(f'Exception {e} while binarizing image. Returning image as is...')
        image_bin = image

      return image_bin


    def __getitem__(self, idx):
        sequece = self.words[idx]

        # Read image with opencv
        image = cv2.imread(self.paths_images[idx])

        if self.binarize:  
          # Convert to grayscale if image is not grayscale
          if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
          # Binarize image with opencv Otsu algorithm
          _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Convert to PIL image
        image = Image.fromarray(image)

        if self.transform:
          image = self.transform(image)
  
        return image, sequece


class HTRDataModule(pl.LightningDataModule):
    def __init__(
      self, 
      train_config: DataConfig,
      val_config: DataConfig,
      test_config: DataConfig,
      seed: int = 42
      ):
        super().__init__()
        self.train_config = train_config
        self.val_config = val_config
        self.test_config = test_config
        self.seed = seed
        # Seed everything with Pytorch Lightning
        pl.seed_everything(self.seed, workers=True)
        torch.manual_seed(self.seed)

        # Setting up vocab and text transforms
        vocab = open(self.train_config.vocab_path, 'r').read().split('\n')
        vocab = sorted(list(vocab))

        # Add special tokens to vocab ('et' for Saint-Gall, 'ç' for Esposalles)
        # vocab = ['<pad>', '<sos>', '<eos>', '<unk>'] + vocab + ['et'] + ['ç']
        vocab = ['<sos>', '<pad>','<eos>', '<unk>'] + vocab + ['et'] + ['ç']
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        log.info(f'VOCAB: {self.vocab}')
        log.info(f'VOCAB SIZE: {self.vocab_size}')

        # Set encoding and decoding functions
        stoi = {s: i for i, s in enumerate(self.vocab)}
        itos = {i: s for i, s in enumerate(self.vocab)}

        global encode
        global decode 

        encode = lambda s: [stoi[token] for token in s]
        decode = lambda x: [itos[i] for i in x]
        self.encode = encode
        self.decode = decode

        # Make self.encode and self.decode global functions
               
        
        
        print(f'Constructing HTRDataModule with vocab {self.vocab}')
        print(f'VOCAB SIZE {self.vocab_size}')

        self.vocab_transform = self.encode
        self.text_transform = sequential_transforms(
            tokenize, # Tokenization (split into characters)
            self.vocab_transform, # Numericalization (from chars to ints)
            tensor_transform # Add BOS/EOS and create tensor of ints
        )

        self.save_hyperparameters(logger=False)
        log.info(f'HYPERPARAMETERS: {self.hparams}')

        
    def prepare_data(self):
        # download, split, tokenize, etc...
        # Prepare dataset
        pass
        # if 'saint-gall' in self.train_config.datasets.keys() \
        #     or 'saint-gall' in self.val_config.datasets.keys() \
        #     or 'saint-gall' in self.test_config.datasets.keys():
        #     prepare_saint_gall(data_dir=self.train_config.data_dir)
        # if 'esposalles' in self.train_config.datasets.keys() \
        #     or 'esposalles' in self.val_config.datasets.keys() \
        #     or 'esposalles' in self.test_config.datasets.keys():
        #     prepare_esposalles(data_dir=self.train_config.data_dir)
        # if self.dataset == "saint-gall":
        #   prepare_saint_gall(data_dir=self.data_dir)
        # elif self.dataset == "esposalles":
        #   prepare_esposalles(data_dir=self.data_dir)
        # elif self.dataset == "synth":
        #     pass
        # else:
        #     pass

    def setup(self, stage: str):
        print(f'Print train_config transforms: {self.train_config.transforms[0]}')
        self.stage = stage
        print(f'Setting up stage {stage}...')

        if stage == "fit" or stage is None:
            _stages, configs = ["train", "val"], {"train": self.train_config, "val": self.val_config}
            stage_datasets = dict({"train": [], "val": []})
        elif stage == "test" or stage == "predict":
            _stages, configs = ["test"], {"test": self.test_config}
            stage_datasets = dict({"test": []})

        # Create datasets for each stage
        for _stage in _stages:
            log.info(f'Setting up stage {_stage}')
            self.__setattr__(_stage + "_sampler", None)
            
            for dataset in configs[_stage].datasets:
              ds = configs[_stage].datasets[dataset]
              
              print(f'DATASET: {ds}')

              # Check type of instance
              if isinstance(ds, DatasetConfig): # Real dataset
                  read_data = hydra.utils.get_method(ds.read_data)

                  # Check if splits_paths corresponds to stage separated in two lines
                  assert ds.splits_path.split("/")[-1] == _stage + ".txt", \
                    f'File {ds.splits_path} does not correspond to stage {_stage}'
                  
                  with open(ds.splits_path, "r") as f:
                      setfiles = f.read().splitlines()
                  images_paths, words = read_data(ds.images_path, ds.labels_path, setfiles)
                  print(f'Binarize: {configs[_stage].binarize}')
                  htr_dataset = HTRDataset(images_paths, words, binarize=configs[_stage].binarize, transform=configs[_stage].transforms[0])

              elif isinstance(ds, SynthDatasetConfig):
                  # Read words from json file
                  # with open(ds.words_path, "r") as f:
                  #     words = json.load(f)
                  #     words, distr = words["words"], [1]*len(words["words"])

                  #     words_distr, real_distr = [], []
                  #     for idx, word in enumerate(words):
                  #         words_distr += [word] * distr[idx]
                  #         real_distr += [1/distr[idx]] * distr[idx]
                  #     words, distr = words_distr, real_distr
                  # Load wikitext-2 dataset from huggingface 
                  # tokenizer = AutoTokenizer.from_pretrained("wikitext-2")
                  # dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
                  # sequences = dataset["train"]["text"]
                  # real_distr = [1/len(sequences)] * len(sequences) # Balancing not used. Uniform distribution
                  # distr = real_distr
                  # Read all sequences from dataset path
                  sequences = open(ds.words_path, "r").read().split("\n")
                  print(f'Number of sequences in dataset {dataset}: {len(sequences)} in stage {_stage}')
                  real_distr = [1/len(sequences)] * len(sequences) # Balancing not used. Uniform distribution
                  distr = real_distr
                  

                  # ds.stage_sampler = WeightedRandomSampler( # Replacement = False if real distribution is used, True if not (Uniform distribution)
                  #         weights=real_distr, 
                  #         num_samples=len(real_distr), 
                  #         replacement=False if ds.distr else True
                  # )

                  # self.__setattr__(_stage + "_sampler", ds.stage_sampler)
                  # print(f'Sampler {self.__getattribute__(_stage + "_sampler")} added to datamodule as {self.__getattribute__(_stage + "_sampler")} for stage {_stage}')

                  htr_dataset = HTRDatasetSynth(sequences, distr, ds.fonts_path, transform=configs[_stage].transforms[0])
              
              elif isinstance(ds, RandomSynthDatasetConfig):
                  htr_dataset =  HTRDatasetSynthRandom(self.vocab[5:], ds.words_to_generate, ds.max_len, ds.fonts_path, transform=configs[_stage].transforms[0])
                  print(f'Generating data with vocab {self.vocab[5:]}') # Remove special tokens from vocab and whitespace

              print(f'Number of samples in dataset {dataset}: {len(htr_dataset)} in stage {_stage}')

              stage_datasets[_stage].append(htr_dataset)

            # Concatenate datasets if more than one
            if _stage == "train" and len(stage_datasets[_stage]) > 1:
                stage_datasets[_stage] = torch.utils.data.ConcatDataset(stage_datasets[_stage])

                # Create weighted sampler for each sample in ConcatDataset.
                # Each sample has a weight equal to the number of samples in the dataset it belongs to.
                # This is done to balance the training of the model on different datasets.
                weights = []
                for dataset in stage_datasets[_stage]:
                    weights += [len(dataset)] * len(dataset)
                weights = torch.DoubleTensor(weights)
                weights = 1. / weights
                weighted_sampler = WeightedRandomSampler(weights, len(stage_datasets[_stage]), replacement=True)
                self.__setattr__(_stage + "_sampler", weighted_sampler)
                print(f'Sampler {self.__getattribute__(_stage + "_sampler")} added to datamodule as {self.__getattribute__(_stage + "_sampler")} for stage {_stage}')

            elif _stage == "train" and len(stage_datasets[_stage]) == 1:
                stage_datasets[_stage] = stage_datasets[_stage][0]  
                
            self.__setattr__(_stage + "_dataset", stage_datasets[_stage])
            print(f'Object {stage_datasets[_stage]} of type {type(stage_datasets[_stage])} added to datamodule as {self.__getattribute__(_stage + "_dataset")} for stage {_stage}')

    def train_dataloader(self):
      return DataLoader(
        self.train_dataset,
        batch_size=self.train_config.batch_size,
        shuffle=False,# if self.train_sampler is None else False,
        drop_last=True,
        num_workers=self.train_config.num_workers,
        pin_memory=self.train_config.pin_memory,
        collate_fn=lambda batch: collate_fn(batch, img_size=self.train_config.img_size, text_transform=self.text_transform),
        sampler=self.train_sampler if self.train_sampler is not None else RandomSampler(self.train_dataset)
,       )

    def val_dataloader(self):
        return [DataLoader(
            dataset,
            batch_size=self.val_config.batch_size,
            shuffle=False,
            num_workers=self.val_config.num_workers,
            pin_memory=self.val_config.pin_memory,
            collate_fn=lambda batch: collate_fn(batch, img_size=self.train_config.img_size, text_transform=self.text_transform),
            sampler=self.val_sampler if self.val_sampler is not None else None
        ) for dataset in self.val_dataset]

    def test_dataloader(self):
        return [DataLoader(
            dataset,
            batch_size=self.test_config.batch_size,
            shuffle=False,
            num_workers=self.test_config.num_workers,
            pin_memory=self.test_config.pin_memory,
            collate_fn=lambda batch: collate_fn(batch, img_size=self.train_config.img_size, text_transform=self.text_transform),
            sampler=self.test_sampler if self.test_sampler is not None else None
        ) for dataset in self.test_dataset]
    
    
if __name__ == "__main__":
    _ = HTRDataModule()