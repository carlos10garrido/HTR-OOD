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

from unidecode import unidecode

PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 2, 0, 1, 3


def read_htr_fonts(fonts_path):
  fonts = []
  for font in os.listdir(fonts_path):
    fonts.append(fonts_path + font)
  return fonts
    
def collate_fn(batch, img_size, text_transform):
    images_batch, sequences_batch, padded_columns = [], [], []
    images_shapes = torch.tensor([image_sample.shape for image_sample, seq_sample in batch]) # Get shapes of images in batch [B, C, H, W]
    all_height_ratios = (images_shapes[:, 1] / img_size[0]) # Get height ratios for all images in batch
    all_width_reescaled = images_shapes[:, 2] / all_height_ratios # Get width reescaled for all images in batch
    assert all_height_ratios.shape[0] == all_width_reescaled.shape[0], 'All height ratios and all width reescaled must have the same length'
    max_width = img_size[1] # Get max width ratio

    for i, (image_sample, seq_sample) in enumerate(batch):
      if i == 0:
        images_batch = torch.ones(len(batch), 3, img_size[0], max_width) # Reescaled height and width and white background
      
      # Resize image to fixed height
      height, width = img_size[0], all_width_reescaled[i].int()

      if all_width_reescaled[i] > max_width:
        width = max_width

      image_resized_height = torchvision.transforms.Resize((height, width), antialias=True)(image_sample)
      images_batch[i, :, :, :image_resized_height.shape[2]] = image_resized_height 

      # Calculate padding
      padding_added = max_width - image_resized_height.shape[2]
      padded_columns.append(padding_added)

      # print(f'Seq sample: {seq_sample}')
      sequences_batch.append(text_transform(seq_sample))

    sequences_batch = pad_sequence(sequences_batch, padding_value=PAD_IDX)
    padded_columns = torch.tensor(padded_columns)

    assert images_batch.shape[0] == sequences_batch.shape[1] == padded_columns.shape[0], "Batch size of images and sequences should be equal"
    
    return images_batch, sequences_batch, padded_columns

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

    # TODO: check new way of cropping image!

    # # Apply transformations here? (before cropping)

    # # Crop image to fit the text
    # data = np.asarray(img)
    # data = np.mean(data, -1)
    # min_col = np.argmin(data, 0)
    # min_row = np.argmin(data, 1)

    # first_col = np.nonzero(min_col)[0][0] - 1
    # last_col = np.nonzero(min_col)[0][-1] + 1

    # first_row = np.nonzero(min_row)[0][0] - 1
    # last_row = np.nonzero(min_row)[0][-1] + 1

    # cropped_image = img.crop((first_col, first_row, last_col, last_row))
    # # Add padding to image with PIL

    # # img = cropped_image.resize((256, 64))
    # img = cropped_image

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
      # print(f'Image id-{image_id}.Text-{text}')

      # Remove \n if exists in the text
      text = text.replace("\n", "")
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

def read_data_parzival(images_path, words_path, files):
  images_paths, words = [], []

  # Convert files list to a set
  files = set(files)
  print(f'Files {files}')

  # Read words from word_labels.txt
  with open(words_path + "word_labels.txt", "r") as f:
    for line in f:
      image_id, text = line.split(" ")
      image_id_split = image_id.split("-")

      # Remove \n if exists in the text
      text = text.replace("\n", "")
      file = "-".join(image_id_split[:2])

      # Check if first_part is in files
      if file in files:
        image_path = images_path + image_id + ".png"
        text = text.replace("-", "").replace("|", " ")
        text = text.replace("s_pt", ".").replace("s_cm", ",")
        text = text.replace("s_mi", "-").replace("s_qo", ":")
        text = text.replace("s_sq", ";").replace("s_et", "V")
        text = text.replace("s_bl", "(").replace("s_br", ")")
        text = text.replace("s_qt", "'").replace("s_GW", "G.W.")
        text = text.replace("s_", "")

        # Check if image_path exists
        if os.path.exists(image_path):
            images_paths.append(image_path)
            words.append(text)

  
  return images_paths, words

def prepare_esposalles(data_dir):
  print(f'Preparing esposalles dataset')

  # Check if exists esposalles/splits and create it if not
  if not os.path.exists(data_dir + "/splits"):
    print(f'Creating esposalles/splits')
    os.makedirs(data_dir + "/splits", exist_ok=True)

    # Put IEHHR_training_part1 + IEHHR_training_part2 folders in train.txt
    with open(data_dir + "/splits/train.txt", "w") as f:
      for file in os.listdir(data_dir + "/IEHHR_training_part1"):
        f.write("/IEHHR_training_part1/" + file + "\n")
      for file in os.listdir(data_dir + "/IEHHR_training_part2"):
        f.write("/IEHHR_training_part2/" + file + "\n")

    # Put IEHHR_training_part3 + IEHHR_training_part2 folders in validation.txt
    with open(data_dir + "/splits/validation.txt", "w") as f:
      for file in os.listdir(data_dir + "/IEHHR_training_part3"):
        f.write("/IEHHR_training_part3/" + file + "\n")

    # Put IEHHR_test folders in test.txt
    with open(data_dir + "/splits/test.txt", "w") as f:
      for file in os.listdir(data_dir + "/IEHHR_test"):
        f.write("/IEHHR_test/Records/" + file + "\n")
  else:
    print(f'esposalles/splits already exists')

def prepare_saint_gall(data_dir):
  print(f'Preparing saint_gall dataset')

  # Check if exists saint_gaill/data/words_images_normalized or is empty
  if not os.path.exists(data_dir + "/data/words_images_normalized") \
    or len(os.listdir(data_dir + "/data/words_images_normalized")) == 0:
    print(f'Creating saint_gall/data/words_images_normalized')
    os.makedirs(data_dir + "/data/words_images_normalized", exist_ok=True)

    # Read words_location.txt and for each image crop it and save it in saint_gaill/data/words_images_normalized
    with open(data_dir + "/ground_truth/word_location.txt", "r") as f:
      for line in f:
        image_path, _, locations = line.split(" ")
        image_path_full = data_dir + '/data/line_images_normalized/' + image_path + ".png"
        image = Image.open(image_path_full)

        # Crop all locations images
        for idx, location in enumerate(locations.split("|")):
            x0, x1 = int(location.split("-")[0]), int(location.split("-")[1])

            # Crop image
            cropped_image = image.crop((x0, 0, x1, image.size[1]))

            # Save image in words_images_normalized
            cropped_image_path = data_dir + "/data/words_images_normalized/" + image_path + "-" + str(idx) + ".png"
            print(f'Saving image in {cropped_image_path}')
            cropped_image.save(cropped_image_path)

    print(f'Parsed images from lines')

  else:
    print(f'Images from lines already parsed')   

    # Set splits from random_split if not exists in splits folder
    random_list_path = splits_path + "random_list.txt"

    if os.path.exists(random_list_path):
      print(f'Random list already exists')

      # Divide random_list.txt in train, validation and test and rewrite
      with open(random_list_path, "r") as f:
          random_list = f.read().splitlines()

          # Divide random_list in train, validation and test [0.6, 0.2, 0.2] and rewrite
          n_train = int(len(random_list) * 0.6)
          n_val = int(len(random_list) * 0.2)

          # Write lines in train.txt, val.txt and test.txt
          with open(splits_path + "train.txt", "w") as f:
              f.write("\n".join(random_list[:n_train]))
          with open(splits_path + "validation.txt", "w") as f:
              f.write("\n".join(random_list[n_train:n_train+n_val]))
          with open(splits_path + "test.txt", "w") as f:
              f.write("\n".join(random_list[n_train+n_val:]))
    else:
        raise Exception(f'Random list not exists in {random_list_path}!')


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
    # return cv2.dilate(np.array(image), self.kernel, iterations=self.iterations)

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
    # Convert image to PIL image
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
    
    # print(f'Returning image with degradation')
    return image
    
# class Background(object):
#   def __init__(self, paths):
#     self.paths = paths
#     self.files = !find $paths -type f -name "*.png"
#     # Same function but in python with os without using bash
#     self.files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(paths) for f in filenames if os.path.splitext(f)[1] == '.png']
#     print(f'Files {self.files}')

#   def __call__(self, image):
#     # Read image
    
    
#     # Add text to image
#     paths_fonts = !find ../data/synth/final_fonts_rendered -type f -name '*.ttf'
#     font = np.random.choice(paths_fonts)
#     print(f'Font: {font}')
#     font = ImageFont.truetype(font, 50)
#     txt = 'As he pointed out, "the world is round".'
    
#     text_color = (0, 0, 0)
    
    
    