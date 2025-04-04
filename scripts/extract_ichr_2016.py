# Python script to extract icfhr_2016 dataset
import torchvision
import torch
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import re
import numpy as np
import sys
import os
import PIL
from tqdm import tqdm

print(f'SCRIPT TO EXTRACT ICFHR 2016 DATASET')

# Read arguments to get the path to the dataset
path_data = sys.argv[1]
print(f'Path to the dataset: {path_data}')

# Find all files with extension *.xml
list_xml_icfhr_2016 = []
for root, dirs, files in os.walk(path_data):
  for file in files:
    if file.endswith(".xml"):
      list_xml_icfhr_2016.append(os.path.join(root, file))

images_icfhr, gt_icfhr = [], []

for file in list_xml_icfhr_2016:
  if "Test" not in file:
    image_path = file[:-4] + '.JPG'
  else:
    image_path = str(file[:-4]).replace("/page", "") + '.JPG'
  
  images_icfhr.append(image_path)
  gt_icfhr.append(file)


assert len(images_icfhr) == len(gt_icfhr), \
  f'Number of images and ground truth files are different: {len(images_icfhr)} != {len(gt_icfhr)}'

images_icfhr = sorted(images_icfhr)
gt_icfhr = sorted(gt_icfhr)


path_images_to_save = f'./{path_data}/lines/'
path_transcriptions_to_save = f'./{path_data}/transcriptions/'

print(f'Path images to save: {path_images_to_save}')
print(f'Path transcriptions to save: {path_transcriptions_to_save}')

def parse_xml_auto_namespace(xml_file):
    # Parse the XML file to extract the namespace map name
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for items_root in root.items():
      ns_val = items_root[1].split(" ")[0]

    ns = {namespace_map_name: ns_val}

    return ns_val

def get_bbox(coords):
  xs, ys = [], []
  for coord in coords.split(" "):
    x, y = coord.split(",")
    xs.append(int(x))
    ys.append(int(y))

  xmin = np.array(xs).min()
  xmax = np.array(xs).max()
  ymin = np.array(ys).min()
  ymax = np.array(ys).max()

  return xmin, ymin, xmax, ymax

def parse_textline_regions(xml_file, namespace_map_name):
    # print(xml_file)
    # Construct the namespace map using the provided name
    namespace_map_name = parse_xml_auto_namespace(xml_file) 
    ns = {'ns': namespace_map_name}

    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Find all TextLine elements
    text_lines = root.findall('.//ns:TextLine', ns)
    data = []

    # Iterate over each TextLine
    for text_line in text_lines:
        coords = text_line.find('./ns:Coords', ns).attrib['points']
        baseline = text_line.find('./ns:Baseline', ns).attrib['points']
        text_equiv = text_line.find('./ns:TextEquiv', ns)
        unicode_label = text_equiv.find('./ns:Unicode', ns).text if text_equiv is not None else None

        # Append the data to the list
        data.append({
            'Coords': coords,
            'Baseline': baseline,
            'Unicode': unicode_label
        })

    return data


# Usage
xml_file = file
namespace_map_name = None #'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'

print(f'Parsing ICFHR-2016')
for id_image, (xml_file, path_image) in tqdm(enumerate(zip(gt_icfhr, images_icfhr)), total=len(gt_icfhr)):
  path_xml_file = xml_file
  xml_file = ''.join(xml_file.split("/")[-1])
  path_image = path_image.replace("/page/", "/Images/")
  image = PIL.Image.open(f'{path_image}')
  try:
    parsed_data = parse_textline_regions(path_xml_file, namespace_map_name)
  except Exception as e:
    print(f'Exception {e} in file {xml_file}')
    continue

  for idx, item in enumerate(parsed_data):
      text = item['Unicode']
      bboxes = get_bbox(item['Coords'])
      text = text.replace("¬", "") if text is not None else ""
      
      image_pil = torchvision.transforms.functional.to_pil_image(image) if isinstance(image, torch.Tensor) else image
      cropped = image_pil.crop(bboxes)
      path_line_image = f'{path_images_to_save}{xml_file[:-4]}_{idx}.png'
      path_line_transcription = f'{path_transcriptions_to_save}{xml_file[:-4]}_{idx}.txt'
      cropped.save(path_line_image, 'PNG')
      
      # Append to the end of transcriptions.txt the id_image <space> transcription
      with open(f'{path_transcriptions_to_save}transcriptions.txt', 'a') as f_transcription:
        f_transcription.write(f'{xml_file[:-4]}_{idx} {text}\n')