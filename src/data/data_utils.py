# Datamodules for each Dataset
import torch
import os
import numpy as np
import torchvision
import pytorch_lightning as pl
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import xml.etree.ElementTree as ET
from torch.nn.utils.rnn import pad_sequence
from typing import Iterable, List
from fontTools.ttLib import TTFont
import cv2
import re
import random as rnd
from typing import Tuple


from unidecode import unidecode

PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 2, 0, 1, 3


def read_htr_fonts(fonts_path):
  fonts = []
  for font in os.listdir(fonts_path):
    fonts.append(fonts_path + font)
  return fonts
    
def collate_fn(batch, img_size, text_transform):
    sequences_batch = []
    images_shapes = torch.tensor([image_sample.shape for image_sample, seq_sample in batch]) # Get shapes of images in batch [B, C, H, W]
    all_height_ratios = (images_shapes[:, 1] / img_size[0]) # Get height ratios for all images in batch
    all_width_reescaled = images_shapes[:, 2] / all_height_ratios # Get width reescaled for all images in batch
    assert all_height_ratios.shape[0] == all_width_reescaled.shape[0], 'All height ratios and all width reescaled must have the same length'
    max_width = img_size[1] # Get max width ratio
    
    images_batch = torch.ones(len(batch), 3, img_size[0], max_width) # Reescaled height and width and white background
    padded_columns = torch.zeros(len(batch)) # Padded columns for each image
  

    for i, (image_sample, seq_sample) in enumerate(batch):
      # Resize image to fixed height
      height, width = img_size[0], all_width_reescaled[i].int()

      if all_width_reescaled[i] > max_width:
        width = max_width

      image_resized_height = torchvision.transforms.Resize((height, width), antialias=True)(image_sample)
      images_batch[i, :, :, :image_resized_height.shape[2]] = image_resized_height 

      # Calculate padding
      padding_added = max_width - image_resized_height.shape[2]
      # padded_columns.append(padding_added)
      padded_columns[i] = padding_added

      # print(f'Seq sample: {seq_sample}')
      sequences_batch.append(text_transform(seq_sample))

    sequences_batch = pad_sequence(sequences_batch, padding_value=PAD_IDX)
    assert images_batch.shape[0] == sequences_batch.shape[1] == padded_columns.shape[0], "Batch size of images and sequences should be equal"
    
    return images_batch, sequences_batch, padded_columns
  
def get_bboxes(image: Image) -> Tuple:
    """
    Get the bounding boxes for the image at a pixel level
    """
    image = image.convert('L')
    image = np.array(image)
    # print(f'Image shape: {image.shape}')
    # Get the bounding box for the image
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
        bboxes_chars.append(bbox)

        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
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
      
    return image, reescaled_chars


def has_glyph(font, glyph):
    # print(font['cmap'])
    font = TTFont(font)
    for table in font['cmap'].tables:
        if ord(glyph) in table.cmap.keys():
            return True
    return False

def generate_image(sequence, font, background_color=(255, 255, 255), text_color=(0, 0, 0)):
    txt = sequence
    font_name = font
    font = ImageFont.truetype(font, 50)
    img_size = (1500, 1000)

    # Check if a font can generate all the characters
    for char in txt:
      if has_glyph(font_name, str(char)) is False:
        raise Exception(f'Font {font} cannot generate char {char}')


    # Generate white image
    img = Image.new("RGB", (img_size[0], img_size[1]), background_color)
    draw = ImageDraw.Draw(img)
    draw.text((img_size[1]//10, img_size[0]//4), txt, font=font, fill=text_color)
    text_bbox = draw.textbbox((img_size[1]//10, img_size[0]//4), txt, font=font)
    img = img.crop(text_bbox)

    # Check if image shapes are zero
    assert img.size[0] != 0 and img.size[1] != 0, f'Image shape is zero. Image shape: {img.size}. Sequence: {sequence}'

    return img

def read_data_IAM(images_path, lines_path, files):
    # Read lines from XML files
    images_paths, lines = [], []
    # print(f'Files {files}')
    for file in files:
      filepath = lines_path + file + ".xml"
      tree = ET.parse(filepath)
      root = tree.getroot()
      root_image = file.split("-")[0] + '/'

      # Get the lines
      for line in root.iter("line"):
        text, image_id = line.attrib.get("text"), line.attrib.get("id")
        # Replace &quot with "
        text = text.replace("&quot;", "\"")
        image_path = images_path + '/' + root_image + "-".join(image_id.split("-")[:2]) + "/" + image_id + ".png"

          # Check if image_path exists
        if os.path.exists(image_path):
            images_paths.append(image_path)
            lines.append(text)

    return images_paths, lines

def read_data_rimes(images_path, lines_path, files):
  # Read files from .txt files
  images_paths, lines = [], []

  for file in files:
    image_path = images_path + file + ".jpg"
    line_path = lines_path + file + ".txt"

    # Read lines from .txt files
    with open(line_path, "r") as f:
      line = f.read().replace("\n", "")
      line = line.replace("°", ".")
      line = line.replace("œ", "oe")
      line = line.replace("¤", "")
      line = line.replace(" €", "€")
      
      lines.append(line)
      images_paths.append(image_path)
    
  return images_paths, lines

def read_data_bentham(images_path, lines_path, files):
  images_paths, lines = [], []

  for file in files:
    image_path = images_path + file + ".png"
    line_path = lines_path + file + ".txt"

    # Read lines from .txt files
    with open(line_path, "r") as f:
      line = f.read().replace("\n", "")
      line = line.replace("§", "S")
      line = line.replace("|", " ")
      
      # Regex for bentham
      line = re.sub(r'(\w)\s([,\.\!\:;\?])', '\g<1>\g<2>', line)  # noqa
      line = re.sub(r'(["\'\(\[<])\s(\w+)', '\g<1>\g<2>', line)  # noqa
      line = re.sub(r'(\w+)\s([\)\]>])', '\g<1>\g<2>', line)  # noqa
      line = re.sub(r'\s+', ' ', line)  # noqa
      
      lines.append(line)
      images_paths.append(image_path)

  return images_paths, lines

def convert_text_washington(text):
    text = text.replace("-", "").replace("|", " ")
    text = text.replace("s_pt", ".").replace("s_cm", ",")
    text = text.replace("s_mi", "-").replace("s_qo", ":")
    text = text.replace("s_sq", ";").replace("s_et", "V")
    text = text.replace("s_bl", "(").replace("s_br", ")")
    text = text.replace("s_qt", "'").replace("s_GW", "G.W.")
    text = text.replace("s_", "")
    return text

def read_data_washington(images_path, lines_paths, files):
  images_paths, lines = [], []

  # Convert files list to a set
  files = set(files)

  # Read words from word_labels.txt
  with open(lines_paths + "transcription.txt", "r") as f:
    for line in f:
      image_id, text = line.split(" ")

      # Remove \n if exists in the text
      text = text.replace("\n", "")
      file = image_id 
      # file = "-".join(image_id_split[:2])

      # Check if first_part is in files
      if file in files:
        image_path = images_path + image_id + ".png"
        text = convert_text_washington(text)

        # Check if image_path exists
        if os.path.exists(image_path):
            images_paths.append(image_path)
            lines.append(text)

  return images_paths, lines

def read_data_saint_gall(images_path, lines_paths, files):
  images_paths, lines = [], []

  # Convert files list to a set
  files = set(files)

  # Read words from transcriptions.txt
  with open(lines_paths + "transcription.txt", "r") as f:
    for line in f:
      image_id, text = line.split(" ")[0], line.split(" ")[1]
      file = '-'.join(image_id.split('-')[:2]) # Get first two parts of the image_id that corresponds to a line
      image_path = images_path + image_id + ".png"
      text = text.replace("\n", "")
      text = text.replace("-", "").replace("|", " ")
      text = text.replace("s_pt", ".").replace("s_cm", ",")

      # Check if folder is in files 
      if file in files:
        images_paths.append(image_path)
        lines.append(text)

  return images_paths, lines

def read_data_rodrigo(images_path, lines_paths, files):
    images_paths, lines = [], []

    # Convert files list to a set
    files = set(files)

    # Read words from word_labels.txt
    with open(lines_paths + "transcriptions.txt", "r") as f:
      for line in f:
        # Rodrigo_00006_00 blablabla
        image_id, text = line[:16], line[17:]

        # Remove \n if exists in the text
        text = text.replace("\n", "")
        text = text.replace("♦", "")
        text = text.replace("\\", "")
        text = text.replace("|", "")
        text = text.replace("þ", "p")
        text = text.replace("Þ", "p")
        text = text.replace("¶", "C")
        text = text.replace("Ⴒ", "p")
        text = text.replace("ք", "p")
        text = text.replace("℣", "v")
        
        file = image_id

        # Check if first_part is in files
        if file in files:
          image_path = images_path + image_id + ".png"

          # Check if image_path exists
          if os.path.exists(image_path):
            images_paths.append(image_path)
            lines.append(text)

    return images_paths, lines

def read_data_icfhr_2016(images_path, lines_paths, files):
  images_paths, lines = [], []

  print(f'Files {files}')

  # Convert files list to a set
  files = set(files)

  # Read words from word_labels.txt
  with open(lines_paths + "transcriptions.txt", "r") as f:
    for line in f:
      image_id, text = line.split(" ", 1)

      # Remove \n if exists in the text
      text = text.replace("\n", "")
      text = text.replace("¾", "3/4")
      text = text.replace("ß", "B")
      text = text.replace("—", "-")
      
      file = image_id 
      # file = "-".join(image_id_split[:2])

      # Check if first_part is in files
      if file.split("_")[0] in files:
        image_path = images_path + image_id + ".png"

        # Check if image_path exists
        if os.path.exists(image_path):
            images_paths.append(image_path)
            lines.append(text)

  return images_paths, lines

def read_data_esposalles(images_path, words_path, files):
  images_paths, words = [], []
  # print(f'Files {files}')
  for file in files:
    folder_path = words_path + file
    record_path = folder_path + '/words/'

    transcription_path = record_path + file.split('/')[-1] + '_' + 'transcription.txt'
    stage = "train"

    if "test" in file:
      stage = "test"
      csv_path = words_path + file.split("/")[-2] + '/gt/' + file.split("/")[-1] + '_output.csv'
      transcription_path = csv_path

    # Read words/transcription.txt
    with open(transcription_path, 'r') as f:
      for line in f:
        if stage == "test":
          folder, word = line.split(",")[0], line.split(",")[1]
        else:
          folder, word = line.split(":")[0], line.split(":")[1]

        image_path = record_path + folder + '.png'
        word = word.replace("\n", "")

        # Check if folder is in files
        if os.path.exists(image_path):
            images_paths.append(image_path)
            words.append(word)


  return images_paths, words


# Dilation class for transform using opencv
class Dilation(object):
  def __init__(self, kernel_size=3, iterations=1):
    self.kernel = np.ones((kernel_size, kernel_size), np.uint8)
    self.iterations = iterations

  def __call__(self, image):
    # First invert the image
    image = cv2.bitwise_not(np.array(image))
    image = cv2.dilate(image, self.kernel, iterations=self.iterations)
    image =  cv2.bitwise_not(image)
    image = Image.fromarray(image)
    return image

# Erosion class for transform using opencv
class Erosion(object):
  def __init__(self, kernel_size=3, iterations=1):
    self.kernel = np.ones((kernel_size, kernel_size), np.uint8)
    self.iterations = iterations

  def __call__(self, image):
    # First invert the image
    image = cv2.bitwise_not(np.array(image))
    image = cv2.erode(image, self.kernel, iterations=self.iterations)
    image = cv2.bitwise_not(image)
    image = Image.fromarray(image)
    return image
    
    # return cv2.erode(np.array(image), self.kernel, iterations=self.iterations)

class Binarization(object):
  def __init__(self):
    pass

  def __call__(self, image):
    if len(image.shape) == 3:
      image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      # Binarize image with opencv Otsu algorithm
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return image

class Degradations(object):
  def __init__(self, ink_colors: List[str], paths_backgrounds: str):
    # Colors come in a list of #RRGGBB strings in hexadecimal
    # conver to tuple of (R, G, B) for each color
    self.colors = [tuple(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) for color in ink_colors]
    self.paths_backgrounds = paths_backgrounds
    # self.files_backgrounds = !find $paths_backgrounds -type f -name "*.png"
    # Same function but in python with os without using bash
    extensions = ['.png', '.jpg', '.jpeg']
    self.files_backgrounds = [os.path.join(dp, f) for dp, dn, filenames in os.walk(paths_backgrounds) for f in filenames if os.path.splitext(f)[1] in extensions]
    # print(f'Files {self.files_backgrounds}')

  def __call__(self, image):
    # Binarize with opencv to obtain a mask of the pixels
    image_thres = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(image_thres, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    ink_image = Image.new('RGB', image.size, color=self.colors[np.random.randint(0, len(self.colors))])
    ink_image = ink_image.resize(image.size)
    image = Image.composite(image, ink_image, Image.fromarray(mask))

    background = cv2.imread(self.files_backgrounds[np.random.randint(0, len(self.files_backgrounds))])
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB) # Convert to RGB
    # Apply median filter to remove text
    background = cv2.medianBlur(background, 51)

    # Convert to PIL
    img_pil_background = Image.fromarray(background)
    img_pil_background_resized = img_pil_background.resize(image.size)

    # Composite image with background without pixels of the text (using mask)
    # invert mask to composite the image with the background
    image = Image.composite(img_pil_background_resized, image, Image.fromarray(mask))
    
    return image
    
    