# %%
import os
import numpy as np
import PIL
from PIL import ImageFont, Image, ImageDraw
import matplotlib.pyplot as plt

import sys
sys.path.append("..") # Adds higher directory to python modules path.

# %%
fonts_path = '../data/synth/final_fonts_rendered/'
fonts = [fonts_path + f for f in os.listdir(fonts_path)]

# %%

from src.data.htr_datamodule import HTRDataModule
from src.data.htr_datamodule import HTRDataset
import src
import unidecode

from torchvision.transforms import v2
tokenizer = src.data.components.tokenizers.CharTokenizer(model_name="char_tokenizer", vocab_file="../data/vocab.txt")


def get_texts(images_path, sequences_path, split_path, read_data, tokenizer, transform=v2.Compose([v2.ToTensor()])):
    with open(split_path, "r") as f:
        setfiles = f.read().splitlines()
    
    images_paths, sentences = read_data(images_path, sequences_path, setfiles)
    return sentences

# %%
train_datasets = dict({
  'iam': get_texts("../data/htr_datasets/IAM/IAM_lines/", "../data/htr_datasets/IAM/IAM_xml/", "../data/htr_datasets/IAM/splits/train.txt", src.data.data_utils.read_data_IAM, tokenizer),
  'rimes': get_texts("../data/htr_datasets/RIMES/RIMES-2011-Lines/Images/", "../data/htr_datasets/RIMES/RIMES-2011-Lines/Transcriptions/", "../data/htr_datasets/RIMES/RIMES-2011-Lines/Sets/train.txt", src.data.data_utils.read_data_rimes, tokenizer),
  'washington': get_texts("../data/htr_datasets/washington/washingtondb-v1.0/data/line_images_normalized/", "../data/htr_datasets/washington/washingtondb-v1.0/ground_truth/", "../data/htr_datasets/washington/washingtondb-v1.0/sets/cv1/train.txt", src.data.data_utils.read_data_washington, tokenizer),
  'saint_gall': get_texts("../data/htr_datasets/saint_gall/saintgalldb-v1.0/data/line_images_normalized/", "../data/htr_datasets/saint_gall/saintgalldb-v1.0/ground_truth/", "../data/htr_datasets/saint_gall/saintgalldb-v1.0/sets/train.txt", src.data.data_utils.read_data_saint_gall, tokenizer),
  'bentham': get_texts("../data/htr_datasets/bentham/BenthamDatasetR0-GT/Images/Lines/", "../data/htr_datasets/bentham/BenthamDatasetR0-GT/Transcriptions/", "../data/htr_datasets/bentham/BenthamDatasetR0-GT/Partitions/train.txt", src.data.data_utils.read_data_bentham, tokenizer),
  'rodrigo': get_texts("../data/htr_datasets/rodrigo/Rodrigo corpus 1.0.0/images/", "../data/htr_datasets/rodrigo/Rodrigo corpus 1.0.0/text/", "../data/htr_datasets/rodrigo/Rodrigo corpus 1.0.0/partitions/train.txt", src.data.data_utils.read_data_rodrigo, tokenizer),
  'icfhr_2016': get_texts("../data/htr_datasets/icfhr_2016/lines/", "../data/htr_datasets/icfhr_2016/transcriptions/", "../data/htr_datasets/icfhr_2016/partitions/train.txt", src.data.data_utils.read_data_icfhr_2016, tokenizer),
})

val_datasets = dict({
  'iam': get_texts("../data/htr_datasets/IAM/IAM_lines/", "../data/htr_datasets/IAM/IAM_xml/", "../data/htr_datasets/IAM/splits/val.txt", src.data.data_utils.read_data_IAM, tokenizer),
  'rimes': get_texts("../data/htr_datasets/RIMES/RIMES-2011-Lines/Images/", "../data/htr_datasets/RIMES/RIMES-2011-Lines/Transcriptions/", "../data/htr_datasets/RIMES/RIMES-2011-Lines/Sets/val.txt", src.data.data_utils.read_data_rimes, tokenizer),
  'washington': get_texts("../data/htr_datasets/washington/washingtondb-v1.0/data/line_images_normalized/", "../data/htr_datasets/washington/washingtondb-v1.0/ground_truth/", "../data/htr_datasets/washington/washingtondb-v1.0/sets/cv1/val.txt", src.data.data_utils.read_data_washington, tokenizer),
  'saint_gall': get_texts("../data/htr_datasets/saint_gall/saintgalldb-v1.0/data/line_images_normalized/", "../data/htr_datasets/saint_gall/saintgalldb-v1.0/ground_truth/", "../data/htr_datasets/saint_gall/saintgalldb-v1.0/sets/val.txt", src.data.data_utils.read_data_saint_gall, tokenizer),
  'bentham': get_texts("../data/htr_datasets/bentham/BenthamDatasetR0-GT/Images/Lines/", "../data/htr_datasets/bentham/BenthamDatasetR0-GT/Transcriptions/", "../data/htr_datasets/bentham/BenthamDatasetR0-GT/Partitions/val.txt", src.data.data_utils.read_data_bentham, tokenizer),
  'rodrigo': get_texts("../data/htr_datasets/rodrigo/Rodrigo corpus 1.0.0/images/", "../data/htr_datasets/rodrigo/Rodrigo corpus 1.0.0/text/", "../data/htr_datasets/rodrigo/Rodrigo corpus 1.0.0/partitions/val.txt", src.data.data_utils.read_data_rodrigo, tokenizer),
  'icfhr_2016': get_texts("../data/htr_datasets/icfhr_2016/lines/", "../data/htr_datasets/icfhr_2016/transcriptions/", "../data/htr_datasets/icfhr_2016/partitions/val.txt", src.data.data_utils.read_data_icfhr_2016, tokenizer),
})

test_datasets = dict({
  'iam': get_texts("../data/htr_datasets/IAM/IAM_lines/", "../data/htr_datasets/IAM/IAM_xml/", "../data/htr_datasets/IAM/splits/test.txt", src.data.data_utils.read_data_IAM, tokenizer),
  'rimes': get_texts("../data/htr_datasets/RIMES/RIMES-2011-Lines/Images/", "../data/htr_datasets/RIMES/RIMES-2011-Lines/Transcriptions/", "../data/htr_datasets/RIMES/RIMES-2011-Lines/Sets/test.txt", src.data.data_utils.read_data_rimes, tokenizer),
  'washington': get_texts("../data/htr_datasets/washington/washingtondb-v1.0/data/line_images_normalized/", "../data/htr_datasets/washington/washingtondb-v1.0/ground_truth/", "../data/htr_datasets/washington/washingtondb-v1.0/sets/cv1/test.txt", src.data.data_utils.read_data_washington, tokenizer),
  'saint_gall': get_texts("../data/htr_datasets/saint_gall/saintgalldb-v1.0/data/line_images_normalized/", "../data/htr_datasets/saint_gall/saintgalldb-v1.0/ground_truth/", "../data/htr_datasets/saint_gall/saintgalldb-v1.0/sets/test.txt", src.data.data_utils.read_data_saint_gall, tokenizer),
  'bentham': get_texts("../data/htr_datasets/bentham/BenthamDatasetR0-GT/Images/Lines/", "../data/htr_datasets/bentham/BenthamDatasetR0-GT/Transcriptions/", "../data/htr_datasets/bentham/BenthamDatasetR0-GT/Partitions/test.txt", src.data.data_utils.read_data_bentham, tokenizer),
  'rodrigo': get_texts("../data/htr_datasets/rodrigo/Rodrigo corpus 1.0.0/images/", "../data/htr_datasets/rodrigo/Rodrigo corpus 1.0.0/text/", "../data/htr_datasets/rodrigo/Rodrigo corpus 1.0.0/partitions/test.txt", src.data.data_utils.read_data_rodrigo, tokenizer),
  'icfhr_2016': get_texts("../data/htr_datasets/icfhr_2016/lines/", "../data/htr_datasets/icfhr_2016/transcriptions/", "../data/htr_datasets/icfhr_2016/partitions/test.txt", src.data.data_utils.read_data_icfhr_2016, tokenizer),
})

# %%
from unidecode import unidecode

# Read vocab and get the frequency for each dataset
with open('../data/vocab.txt', 'r') as f:
  vocab = f.read().splitlines()
  
print(f'Vocab: {vocab}')

vocabs_datasets = dict({
  "train": dict(),
  "val": dict(),
  "test": dict()
})

for train_ds, val_ds, test_ds in zip(train_datasets, val_datasets, test_datasets):
  print(f"Processing {train_ds} dataset")
  print(f"Processing {val_ds} dataset")
  print(f"Processing {test_ds} dataset")
  # Training
  train_vocab = {char: 0 for char in vocab}
  for sentence in train_datasets[train_ds]:
    sentence = tokenizer.pre_tokenize(sentence)
    for char in sentence:
      train_vocab[char] += 1
  vocabs_datasets["train"][train_ds] = train_vocab
  
  # Validation
  val_vocab = {char: 0 for char in vocab}
  for sentence in val_datasets[val_ds]:
    sentence = tokenizer.pre_tokenize(sentence)
    for char in sentence:
      val_vocab[char] += 1
  vocabs_datasets["val"][val_ds] = val_vocab
  
  # Test
  test_vocab = {char: 0 for char in vocab}
  for sentence in test_datasets[test_ds]:
    sentence = tokenizer.pre_tokenize(sentence)
    for char in sentence:
      test_vocab[char] += 1
      
  vocabs_datasets["test"][test_ds] = test_vocab

# %%
# For each dataset display a heatmap with normalized frequenct for each character

# fig, ax = plt.subplots(len(vocabs_datasets["train"]), 3, figsize=(200, 200))

# Plot a heatmap only for train IAM dataset
train_ds = 'iam'
train_vocab = vocabs_datasets["train"][train_ds]
train_vocab = {k: v for k, v in sorted(train_vocab.items())}
train_vocab = {k: v for k, v in train_vocab.items() if v > 0}
train_vocab = {k: v/sum(train_vocab.values()) * 100 for k, v in train_vocab.items()}
# Create a df with sorted keys 




# %% [markdown]
# # Generate custom phrases

# %%
import random as rnd
from PIL import ImageColor
from typing import Tuple

def get_text_width(image_font: ImageFont, text: str) -> int:
    """
    Get the width of a string when rendered with a given font
    """
    return round(image_font.getlength(text) + 2)


def get_text_height(image_font: ImageFont, text: str) -> int:
    """
    Get the height of a string when rendered with a given font
    """
    left, top, right, bottom = image_font.getbbox(text)
    # print(f'Top: {top}, Bottom: {bottom}')
    return bottom
  
  
def get_max_height(image_font: ImageFont, text: str) -> int:
    """
    Get the height of a string when rendered with a given font
    """
    left, top, right, bottom = image_font.getbbox(text)
    # print(f'Top: {top}, Bottom: {bottom}')
    return round(int(bottom) - int(top))
  
def get_bboxes(image: Image) -> Tuple:
    """
    Get the bounding boxes for the image at a pixel level
    """
    image = image.convert('L') # Convert to grayscale
    image = np.array(image)
    # Binarize image with a threshold of 128. If the pixel value is greater than 128, set it to 255
    image = np.where(image > 128, 255, 0)
    
    # print(f'Image shape: {image.shape}')
    # Get the bounding box for the image
    # print(f'Image shape: {image.shape}')
    # bbox = np.where(image <= 255)
    bbox = np.where(image < 255)
    
    x_min, x_max = np.min(bbox[1]), np.max(bbox[1])
    y_min, y_max = np.min(bbox[0]), np.max(bbox[0])
    
    bbox = (x_min, y_min, x_max, y_max) # (left, top, right, bottom)
    
    return bbox



def generate_line(font, text, font_size, stroke_width=0, stroke_fill="#000000"):
    # font = ImageFont.FreeTypeFont(font, font_size)
    font = ImageFont.truetype(font, font_size)
    bbox = font.getbbox(text)
    
    img_size = (int((bbox[2] - bbox[0]) * 2.7), int((bbox[3] - bbox[1]) * 2.7))
    
    # Check that img_size is > 0 in both dimensions
    # print(f'Img size: {img_size}')
    assert img_size[0] > 0 and img_size[1] > 0, f'Image size: {img_size} with font size: {font_size} and font: {font}'
    
    img = Image.new('RGB', img_size, color = (255,255,255))
    draw = ImageDraw.Draw(img)

    draw.text((img_size[0]//10, img_size[1]//10), text, font=font, fill=(0, 0, 0), stroke_width=stroke_width, stroke_fill=stroke_fill)
    
    # print(f'Bbox with numpy: {get_bboxes(img)}')
    
    bbox = get_bboxes(img)
    image = img.crop(bbox).convert('L')
    
    # Generate white image    
    bboxes_chars, generated_chars = [], []
    max_width, max_height, min_width, min_height = 0, 0, 10000, 10000
    
    for char in text:
      img = Image.new("RGB", (img_size[0], img_size[1]), color=(255,255,255))
      draw = ImageDraw.Draw(img)
      draw.text((img_size[0]//10, img_size[1]//10), char, font=font, fill=(0,0,0), stroke_width=stroke_width, stroke_fill=stroke_fill)
      
      if char != ' ':
        bbox = get_bboxes(img)
        
        # print(f'Bbox: {bbox} for char: {char}')
        # Check that bbox is not empty
        assert bbox != (0, 0, 0, 0), f'Bbox is empty for char: {char}'
        # Check that all values are positive and that the width and height are greater than 0
        assert bbox[0] >= 0 and bbox[1] >= 0 and bbox[2] >= 0 and bbox[3] >= 0, f'Bbox is negative for char: {char} and font: {font}'
        # Check that the width and height are greater than 0
        assert bbox[2] - bbox[0] > 0 and bbox[3] - bbox[1] > 0, f'Bbox is negative for char: {char} and font: {font}'
        bboxes_chars.append(bbox)

        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        # print(f'Width, height: {w}, {h}')
        max_width = max(max_width, w)
        max_height = max(max_height, bbox[3])
        min_width = min(min_width, w)
        min_height = min(min_height, bbox[1])
        data = np.array(img)
        generated_chars.append(data)
      else:
        bbox = (0, 0, 0, 0)
        bboxes_chars.append(bbox)
        data = np.array(img)
        generated_chars.append(data)
      
    # Iterate and reescale each character according to the max height and max width
    reescaled_chars = []
    
    for gen_char, bbox in zip(generated_chars, bboxes_chars):
      if bbox == (0, 0, 0, 0):
        gen_char = Image.fromarray(gen_char)
        gen_char = gen_char.resize((64, 64))
        reescaled_chars.append(gen_char)
        continue
        
        
      char = Image.fromarray(gen_char)
      bbox_x = (bbox[0] + (bbox[2] - bbox[0]) / 2) - max_width / 2, (bbox[0] + (bbox[2] - bbox[0]) / 2) + max_width / 2
      bbox_y = (bbox[3] - max_height, bbox[3])
      bbox_y = (min_height, max_height)

      bbox = (bbox_x[0], bbox_y[0], bbox_x[1], bbox_y[1])
      
      # Bbox is left, top, right, bottom
      char = char.crop(bbox)
      char = char.resize((64, 64)) # WARNING, CHECK OTHER RESIZE SIZE
      char = char.convert('L')
      assert char.size == (64, 64), f'Char size: {char.size}'
      
      # Check number of dims == 2
      assert len(np.array(char).shape) == 2, f'Char shape: {np.array(char).shape}'
      reescaled_chars.append(char)
      
    
    # print(f'Max width, max height: {max_width},{max_height}')
      
    return image, reescaled_chars


# %%
# Lets read the IAM by words and plot the width and height of each word
from tqdm import tqdm
# Read the IAM dataset from data/htr_datasets/IAM_words/words
# Find all images with extension .png inside the folder
all_paths = []

for root, dirs, files in os.walk("../data/htr_datasets/IAM_words/words"):
    for file in files:
        if file.endswith(".png"):
            all_paths.append(os.path.join(root, file))
            
print(f'Number of images: {len(all_paths)}')

# Read all images and get the width and height of each word

widths, heights = [], []

for path in tqdm(all_paths):
    try: 
      img = Image.open(path)
      img = img.convert('L')
      img = np.array(img)
      # print(f'Image shape: {img.shape}')
      width, height = img.shape[1], img.shape[0]
      widths.append(width)
      heights.append(height)
    except Exception as e:
      print(f'Error with image: {path}. Error message: {e}')
      
widths = np.array(widths)
heights = np.array(heights)


# Get the width and height of each word
# Plot the width and height of each word


# %%
# From data/htr_datasets/IAM_words/xml read all the word id 'text' and save all the words in a list
# Read the xml files and get the text for each word
import xml.etree.ElementTree as ET

path = '../data/htr_datasets/IAM_words/xml/'
def read_xml(path):
    tree = ET.parse(path)
    root = tree.getroot()
    
    words = []
    
    for word in root.iter('word'):
        words.append(word.attrib['text'])
        
    return words
  
words, lengths = [], []
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".xml"):
            words.extend(read_xml(os.path.join(root, file)))
            lengths.extend([len(word) for word in words])
            
            
print(f'Number of words: {len(words)}')
lengths = np.array(lengths)

for word in words:
    if len(word) > 15:
        print(f'Word: {word}')

iam_words = words
print(f'Number of words: {len(iam_words)}')
print(iam_words[:10])

# %%
# Make a segmenter architecture
import torch
import torch.nn as nn 
import numpy as np
# import torch.nn.functional as F

# import rearrange from

def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)

class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding1D, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # print(f'X shape: {x.shape}')
        # Have into account the batch size
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)
      
      
class PositionalEncoding2D(nn.Module):
    def __init__(self, channels, dtype_override=None):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        :param dtype_override: If set, overrides the dtype of the output embedding.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)
        self.dtype_override = dtype_override
        self.channels = channels

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_y = torch.arange(y, device=tensor.device, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1)
        emb_y = get_emb(sin_inp_y)
        emb = torch.zeros(
            (x, y, self.channels * 2),
            device=tensor.device,
            dtype=(
                self.dtype_override if self.dtype_override is not None else tensor.dtype
            ),
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        self.cached_penc = emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1)
        return self.cached_penc


class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels, dtype_override=None):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels, dtype_override)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)

    @property
    def org_channels(self):
        return self.penc.org_channels

class Segmenter(nn.Module):
  def __init__(self, 
    patch_size: int = 2,
    output_size: int = 64*64,
    in_channels: int = 1,
    encoder_layers: int = 3,
    decoder_layers: int = 3,
    d_model: int = 256,
    hidden_dim: int = 256,
    nheads: int = 4,
    dropout: float = 0.1,
    activation: str = 'relu',
  ) -> None:
    super(Segmenter, self).__init__()
    
    # Convolutional block encoder
    self.conv_encoder = nn.Sequential(
      nn.Conv2d(in_channels, 8, kernel_size=(3,3), stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=(2,2), stride=2),
      nn.Conv2d(8, 16, kernel_size=(3,3), stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=(2,2), stride=2),
      nn.Conv2d(16, 32, kernel_size=(3,3), stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=(2,2), stride=2),
      # nn.MaxPool2d(kernel_size=(2,2), stride=2),
    )
    
    self.patch_size = patch_size
    self.output_size = output_size
    self.in_channels = in_channels
    self.encoder_layers = encoder_layers
    self.decoder_layers = decoder_layers
    self.d_model = d_model
    self.hidden_dim = hidden_dim
    self.nheads = nheads
    self.dropout = dropout
    self.activation = activation
    self.patchify = nn.Conv2d(32, d_model, kernel_size=(patch_size, patch_size), stride=self.patch_size)
    # self.positional_encoding = PositionalEncodingPermute2D(channels=d_model)
    self.positional_encoding = PositionalEncoding1D(d_model=d_model, dropout=dropout)
    
    # Transformer encoder
    encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nheads, dim_feedforward=hidden_dim, dropout=dropout, activation=activation, batch_first=True, norm_first=True)
    self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
    
    # Convolutional block decoder (for character segmented)
    self.conv_decoder = nn.Sequential(
      nn.Conv2d(1, 4, kernel_size=(3,3), stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=(2,2), stride=2),
      nn.Conv2d(4, 4, kernel_size=(3,3), stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=(2,2), stride=2),
      nn.Conv2d(4, 4, kernel_size=(3,3), stride=1, padding=1),
      nn.ReLU(),
    )
    
    self.proj_decoder = nn.Linear(4*16*16, d_model)
    self.positional_encoding_decoder = PositionalEncoding1D(d_model=d_model, dropout=dropout)
    
    # Transformer decoder
    decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nheads, dim_feedforward=hidden_dim, dropout=dropout, activation=activation, batch_first=True, norm_first=True)
    self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
    
    # self.upconv = nn.Sequential( # From 4, 16, 16 to 1, 64, 64
    #   nn.ConvTranspose2d(4, 1, kernel_size=(3,3), stride=2, padding=1, output_padding=1),
    #   nn.ReLU(),
    #   nn.ConvTranspose2d(1, 1, kernel_size=(3,3), stride=2, padding=1, output_padding=1),
    #   # nn.Sigmoid()
    #   nn.ReLU()
    # )
    self.upconv = nn.Sequential(
      nn.Linear(4*16*16, 4*64*64),
      nn.ReLU(),
      nn.Linear(4*64*64, 1*64*64),
      # nn.Sigmoid()
      nn.ReLU()
    )
    
    # Initialize to Xavier
    for p in self.parameters():
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    
    
  def forward(self, x: torch.Tensor, chars_segmented: torch.Tensor, char_lengths: torch.Tensor) -> torch.Tensor:
    # Encoder
    x = self.conv_encoder(x)
    x = self.patchify(x)
    # x = self.positional_encoding(x)
    
    # Rearrange from [B, C, H, W] to [B, H*W, C]
    x = x.permute(0, 2, 3, 1)
    B, H, W, C = x.shape
    x = x.view(B, H*W, C)
    x = self.positional_encoding(x) # This works instead of 2D positional encoding
    x = self.encoder(x)
    # print(f'X shape: {x.shape} after encoder')
    
    memory = x
    
    # Decoder
    # chars_segmented = torch.cat([torch.zeros(chars_segmented.shape[0], 1, 1, 64, 64).to(chars_segmented.device), chars_segmented], dim=1)
    tok_decoder = chars_segmented.reshape(chars_segmented.shape[0]*chars_segmented.shape[1], 1, 64, 64)
    # Convolution downsmapling + projection
    tok_decoder = self.conv_decoder(tok_decoder)
    tok_decoder = self.proj_decoder(tok_decoder.view(tok_decoder.shape[0], -1)).contiguous()
    input_decoder = tok_decoder.reshape(chars_segmented.shape[0], chars_segmented.shape[1], -1).contiguous()
    # Positional encoding for the decoder
    input_decoder = self.positional_encoding_decoder(input_decoder)

    mask = torch.ones(input_decoder.shape[0], input_decoder.shape[1], input_decoder.shape[1])
    mask = torch.triu(mask, diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    
    tgt_key_pad_mask  = torch.zeros(chars_segmented.shape[0], chars_segmented.shape[1])
    for i, length in enumerate(char_lengths):
      tgt_key_pad_mask[i, length+1:] = 1
      
    tgt_key_pad_mask = tgt_key_pad_mask.bool()
    
    # Add the padding mask to the mask
    for i, length in enumerate(char_lengths):
      mask[i, :, length+1:] = float('-inf') # +1 because we added a zero token at the beginning
      
    # print(f'Mask shape: {mask.shape}')
    # print(f'Mask: {mask}')
    
    # print(f'Input decoder shape: {input_decoder.shape}')
    # print(f'Memory shape: {memory.shape}')
    # print(f'Mask shape: {mask.shape}')
    # print(f'Tgt key pad mask shape: {tgt_key_pad_mask.shape}')
    
    
    output = self.decoder(tgt=input_decoder, memory=memory, tgt_mask=mask, tgt_is_causal=True)#, tgt_key_padding_mask=tgt_key_pad_mask)
    
    # AFTER
    output = self.upconv(output)
    output = output.view(chars_segmented.shape[0], chars_segmented.shape[1], 1, 64, 64).contiguous()
    
    return output
  
  def greedy_decoding(self, image: torch.Tensor, max_length: int) -> torch.Tensor:
    """
    Greedy decoding for the transformer
    """
    # Encoder
    x = self.conv_encoder(image)
    x = self.patchify(x)
    # x = self.positional_encoding(x)
    
    # Rearrange from [B, C, H, W] to [B, H*W, C]
    x = x.permute(0, 2, 3, 1)
    B, H, W, C = x.shape
    x = x.view(B, H*W, C)
    x = self.positional_encoding(x)
    x = self.encoder(x)
    
    memory = x
    
    # Decoder
    # Initialize the output tensor for the decoder
    output = torch.ones(image.shape[0], 1, 1, 64, 64).to(image.device)
    
    # Decoder
    
    for i in range(max_length):
      print(f'----- Iteration: {i} -----')
      input_decoder = output.reshape(output.shape[0]*output.shape[1], 1, 64, 64)
      # Convolution downsmapling + projection
      input_decoder = self.conv_decoder(input_decoder)
      input_decoder = self.proj_decoder(input_decoder.view(input_decoder.shape[0], -1))
      input_decoder = input_decoder.reshape(output.shape[0], output.shape[1], -1).contiguous()
      # Positional encoding for the decoder
      input_decoder = self.positional_encoding_decoder(input_decoder)
      
      # Predict the next token
      output_ = self.decoder(tgt=input_decoder, memory=memory)
      print(f'Output shape (after decoder): {output_.shape}')
      output_ = self.upconv(output_)
      output_ = output_.view(output.shape[0], -1, 1, 64, 64).contiguous()
      # Select only last prediction
      output_ = output_[:, -1].unsqueeze(1)
      # output_ = torch.where(output_ > 0.5, 1, 0)
      print(f'Output shape after threshold: {output_.shape} and output shape: {output.shape}')
      # Reshape output original shape
      output = torch.cat([output, output_], dim=1)
      print(f'Output shape after cat: {output.shape}')
      
    return output




# %%
# Create a segmenter model
segmenter = Segmenter(patch_size=2, d_model=1024, hidden_dim=256, nheads=4, encoder_layers=4, decoder_layers=4, dropout=0.0, activation='relu')
# print(segmenter)

# Calculate the number of parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
  
print(f'The model has {count_parameters(segmenter):,} trainable parameters')

# Generate a random image
image = torch.randn(32, 1, 128, 1024)
print(image.shape)

segmented_chars = torch.randn(32, 100, 1, 64, 64)

# Pass the image through the model
output = segmenter(image, segmented_chars, torch.tensor([10]*32))


# %%
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor

class SyntheticDataset(Dataset):
    def __init__(self, words, fonts, transforms=v2.Compose([v2.ToTensor()])):
        self.words = [word for word in words if len(word) > 1 and word != '' and word != ' ']
        self.fonts = fonts
        self.transforms = transforms
        # self.num_exceptions = 0
        # self.correct_generated = 0
        
    def __len__(self):
        return len(self.words)
    
    def __getitem__(self, idx):
        word = self.words[idx]
        font = np.random.choice(self.fonts)
        
        # Generate the word and the segmented characters
        generated = False
        while not generated:
          try: 
            # img, chars = generate_line(font=font, text=word, font_size=np.random.choice([200]), stroke_width=0, stroke_fill='#000000')
            img, chars = generate_line(font=font, text=word, font_size=200, stroke_width=0, stroke_fill='#000000')
            generated = True
            # Check if h and w are greater than 0
            h, w = img.size
            if h <= 0 or w <= 0:
              generated = False
            
            # print(f'Correctly generated: {self.correct_generated}')
          except Exception as e:
            # self.num_exceptions += 1
            # print(f'Error with word: \'{word}\' and font {font}. Error message: {e}. Number of exceptions: {self.num_exceptions}. % of correct generated: {self.correct_generated/(self.correct_generated+self.num_exceptions)}')
            # print(f'Error with word: \'{word}\' and font {font}. Error message: {e}')
            # raise e
            word = self.words[np.random.randint(0, len(self.words))]
            font = np.random.choice(self.fonts)
        
        # img = torchvision.transforms.ToTensor()(img)
        # Apply self.transforms to the image
        if self.transforms:
          img = self.transforms(img)
          
        
        chars = [v2.ToTensor()(char) for char in chars]
        
        # self.correct_generated += 1
        
        return img, word, chars
        
def collate_fn_synth(batch, img_size):
    # Batch contains img, word, chars
    # Resize images to the same size with padding and pad the segmented characters
    # up to the maximum number of characters
    images, words, chars = zip(*batch)
    assert len(images) == len(words) == len(chars)
    max_chars = max([len(char) for char in chars])
    char_lengths = [len(char) for char in chars]
    images_shapes = torch.tensor([img.shape for img in images])
    height_ratios = (images_shapes[:, 1] / img_size[0])
    width_ratios_reescaled = images_shapes[:, 2] / height_ratios
    max_width = img_size[1]
    
    images_batch = torch.ones(len(images), 1, img_size[0], img_size[1])
    segmented_batch = torch.ones(len(chars), max_chars+1, 1, 64, 64) # +1 for blank token
    
    for i, (img, word, segmented_chars) in enumerate(zip(images, words, chars)):
      height, width = img_size[0], width_ratios_reescaled[i].int().item()
      
      if width > max_width:
        width = max_width
        
      image_resized = v2.Resize((height, width), antialias=True)(img)
      images_batch[i, :, :, :image_resized.shape[2]] = image_resized
      # segmented_batch[i, :len(word), :, :, :] = torch.stack(segmented_chars)
      for j, char in enumerate(segmented_chars):
        # check that n_dims == 2 and get the first channel if it has more than 1
        if len(char.shape) == 3:
          char = char[0, :, :]
        segmented_batch[i, j+1, :, :, :] = char
      
    return images_batch, segmented_batch, char_lengths

    
    

# %%
# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create a synthetic dataset
synthetic_dataset = SyntheticDataset(
  iam_words, 
  fonts, 
  transforms=v2.Compose( # Apply padding of 5 pixels and convert to tensor
    [v2.Pad(20, fill=255), v2.ToTensor()]
  )
)
batch_size = 32
synthetic_loader = DataLoader(synthetic_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: collate_fn_synth(batch, (128, 256)))

# Iterate over the synthetic loader
for i, (images, segmented_chars, lengths) in enumerate(synthetic_loader):
    print(f'Images shape: {images.shape}')
    print(f'Segmented chars shape: {segmented_chars.shape}')
    
    # Plot the images and the segmented characters
    fig, ax = plt.subplots(batch_size, segmented_chars.shape[1]+1, figsize=(20, 20))
    
    for i in range(batch_size):
        ax[i, 0].imshow(images[i, 0, :, :], cmap='gray')
        # ax[i, 0].axis('off')
        
        for j in range(segmented_chars.shape[1]):
            ax[i, j+1].imshow(segmented_chars[i, j, 0, :, :], cmap='gray', vmin=0, vmax=1)
            ax[i, j+1].axis('on')
    
    break

# %%
print(torch.cuda.is_available())

# %%
# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

synthetic_dataset = SyntheticDataset(
  iam_words, 
  fonts, 
  transforms=v2.Compose( # Apply padding of 5 pixels and convert to tensor
    [v2.Pad(20, fill=255), v2.ToTensor()]
  )
)

batch_size = 32
synthetic_loader = DataLoader(
  synthetic_dataset, 
  batch_size=batch_size, 
  shuffle=True, 
  collate_fn=lambda batch: collate_fn_synth(batch, (128, 256)), 
  num_workers=16, 
  # pin_memory=True, 
  persistent_workers=False,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Segmenter(patch_size=2, d_model=1024, hidden_dim=1024, nheads=8, encoder_layers=4, decoder_layers=4, dropout=0.1, activation='relu')
# model = torch.compile(model)
print(model)


# Check if path exists and load the model
if os.path.exists('segmenter.ckpt'):
  model.load_state_dict(torch.load('segmenter.ckpt'))
  print('Model loaded')
else:
  print('Model not loaded')

model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.MSELoss(reduction='mean').to(device)

print(f'The model has {count_parameters(model):,} trainable parameters')
print(f'Device: {device}')
print(f'Len dataset: {len(synthetic_loader)}')

best_loss = 1e9

# Training loop
for epoch in range(1000):
  total_loss = 0.0
  for i, (images, segmented_chars, lengths) in tqdm(enumerate(synthetic_loader), total=len(synthetic_loader), desc=f'Epoch {epoch}'):
  #enumerate(synthetic_loader):
      images = images.to(device)
      segmented_chars = segmented_chars.to(device)
      
      optimizer.zero_grad()
      
      output = model(images, segmented_chars[:, :-1], lengths).to(device) # Forward generates the blank token
      segmented_chars = segmented_chars[:, 1:, :, :, :] # Remove first segment since it is a blank token
      
      # Generate a mask for the loss (mask the padding with the lengths)
      mask = torch.ones(segmented_chars.shape[0], segmented_chars.shape[1])
      for j, length in enumerate(lengths):
        mask[j, length+1:] = 0 # +1 to account for the blank token

      # Cast to bool
      mask = mask.bool()
      mask = mask.to(device)
      
      # print(f'Segmented chars shape: {segmented_chars.shape}')
      # print(f'Mask shape: {mask.shape}')
      # print(f'Output shape: {output.shape}')
      
      
      # Calculate the loss
      loss = criterion(output, segmented_chars).to(device)
      loss_masked = torch.masked_select(loss, mask).mean()

      # print(f'{i}. Loss masked: {loss_masked}')
      
      total_loss += loss_masked.item()
      loss_masked.backward()
      
      if i % 100 == 0:
        print(f'{i}. Loss masked: {loss_masked}')
        
        grads = [
          param.grad.detach().flatten()
          for param in model.parameters()
          if param.grad is not None
        ]
        norm = torch.cat(grads).norm()
        print(f'Norm of the gradients: {norm}')
      
      
      # grads = [
      #   param.grad.detach().flatten()
      #   for param in model.parameters()
      #   if param.grad is not None
      # ]
      # norm = torch.cat(grads).norm()
      # print(f'Norm of the gradients: {norm}')      

      # # Get mean values of gradients
      # print(f'Mean of the gradients: {torch.cat(grads).mean()}')
      
      # Get the norm per each layer of the model
      # grad_layers = []

      # for name, param in model.named_parameters():
      #   if param.grad is not None:
      #     grad_layers.append(param.grad.norm())
      #     print(f'Layer: {name}. Norm: {param.grad.norm()}')

      optimizer.step()
      
  print(f'Epoch: {epoch}. Total loss epoch: {total_loss / len(synthetic_loader)}')
  
  # Save the model every epoch if mean loss is less than best loss
  total_loss_epoch = total_loss / len(synthetic_loader)
  
  if total_loss_epoch < best_loss:
    best_loss = total_loss_epoch
    torch.save(model.state_dict(), 'segmenter.ckpt')
    with open("log_losses_segmenter.txt", "a") as log_file:
        log_file.write(f'Loss epoch {epoch}: {best_loss}\n')
    print(f'Model saved with loss: {best_loss}')

# %%
# model = model.to(device)
# del model


