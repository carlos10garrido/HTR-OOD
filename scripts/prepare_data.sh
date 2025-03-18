#!/bin/bash

# Prepare and organize data for training, validation and testing for each dataset
# It is assumed that there is a folder called data/$dataset with the raw data. 
# It will create a split file for each dataset and each stage (train, val, test)
folder_name=$1 # It is the original folder name in which original files are stored
datasets="iam rimes washington saint_gall bentham rodrigo icfhr_2016"

for dataset in $datasets
  # Create the dataset folder inside data
  mkdir -p data/prepare_$dataset

  # Prepare the dataset
  # IAM
  if [ $dataset == "iam" ]
  then
    # Descompress the dataset (lines.tar, xml.tar and splits.zip)
    tar -xvf data/$folder_name/lines.tar -C data/prepare_iam
    tar -xvf data/$folder_name/xml.tar -C data/prepare_iam
    unzip data/$folder_name/splits.zip -d data/prepare_iam

    # Prepare the dataset using bash
    # if splits folder does not exist, create it
    if [ ! -d data/$folder_name/splits ]
    then
      mkdir data/$folder_name/splits
    fi
    # Create a train.txt, val.txt and test.txt file and paste the content of train.uttlist, validation.uttlist and test.uttlist
    cat data/$folder_name/train.uttlist > data/prepare_iam/splits/train.txt
    cat data/$folder_name/validation.uttlist > data/prepare_iam/splits/val.txt
    cat data/$folder_name/test.uttlist > data/prepare_iam/splits/test.txt
  fi

  # Rimes
  if [ $dataset == "rimes" ]
  then
    # Descompress the dataset (RIMES-2011-Lines.zip) -> Contains Images, Sets, Transcriptions
    unzip data/$folder_name/RIMES-2011-Lines.zip -d data/prepare_rimes
    # Create a train.txt, val.txt and test.txt file and paste the content of TrainLines.txt, ValidationLines.txt and TestLines.txt
    cat data/$folder_name/TrainLines.txt > data/prepare_rimes/splits/train.txt
    cat data/$folder_name/ValidationLines.txt > data/prepare_rimes/splits/val.txt
    cat data/$folder_name/TestLines.txt > data/prepare_rimes/splits/test.txt
  fi

  # Washington
  if [ $dataset == "washington" ]
  then
    # Descompress the dataset (washingtondb-v1.0) -> Contains data/, ground_truth/, sets/
    unzip data/$folder_name/washingtondb-v1.0.zip -d data/prepare_washington
    
    # We will use cv1 for the splits. Just rename sets/cv1/valid.txt to sets/cv1/val.txt
    cat data/prepare_washington/sets/cv1/valid.txt > data/prepare_washington/sets/cv1/val.txt
  fi

  # Bentham
  if [ $dataset == "bentham" ]
  then
    # Descompress the dataset (BenthamDatasetR0-GT.tbz, BenthamDatasetR0-Images.tbz)
    tar -xvf data/$folder_name/BenthamDatasetR0-GT.tbz -C data/prepare_bentham
    tar -xvf data/$folder_name/BenthamDatasetR0-Images.tbz -C data/prepare_bentham

    # Create a train.txt, val.txt and test.txt file and paste the content of TrainLines.lst, ValidationLines.lst and TestLines.lst
    cat data/$folder_name/TrainLines.lst > data/prepare_bentham/splits/train.txt
    cat data/$folder_name/ValidationLines.lst > data/prepare_bentham/splits/val.txt
    cat data/$folder_name/TestLines.lst > data/prepare_bentham/splits/test.txt
  fi

  # Saint Gall
  if [ $dataset == "saint_gall" ]
  then
    # Descompress the dataset (saintgalldb-v1.0) -> Contains data/, ground_truth/, sets/
    unzip data/$folder_name/saintgalldb-v1.0.zip -d data/prepare_saint_gall

    # Just rename sets/cv1/valid.txt to sets/cv1/val.txt
    cat data/prepare_saint_gall/sets/cv1/valid.txt > data/prepare_saint_gall/sets/cv1/val.txt
  fi

  # Rodrigo
  if [ $dataset == "rodrigo" ]
  then
    # Descompress the dataset (Rodrigo corpus 1.0.0.tar) -> Contains images/, text/, partitions/
    tar -xvf data/$folder_name/Rodrigo_corpus_1.0.0.tar -C data/prepare_rodrigo

    # Just rename partitions/validation.txt to partitions/val.txt
    cat data/prepare_rodrigo/partitions/validation.txt > data/prepare_rodrigo/partitions/val.txt
  fi
  
  # ICFHR 2016
  if [ $dataset == "icfhr_2016" ]
  then
    # Descompress the dataset (Train-And-Val-ICFHR-2016.tar) -> Contains PublicData/Training, PublicData/Validation. 
    tar -xvf data/$folder_name/Train-And-Val-ICFHR-2016.tar -C data/prepare_icfhr_2016
    
    # Descompress the test dataset (Test-ICFHR-2016.tar) -> Contains the images (page level!)
    tar -xvf data/$folder_name/Test-ICFHR-2016.tar -C data/prepare_icfhr_2016

    # Create a lines, transcriptions and splits folder
    mkdir -p data/prepare_icfhr_2016/lines
    mkdir -p data/prepare_icfhr_2016/transcriptions
    mkdir -p data/prepare_icfhr_2016/splits
    # Create a train.txt, val.txt from copying the list in Training/page/list and Validation/page/list (removing the final .xml)
    cat data/prepare_icfhr_2016/PublicData/Training/page/list | sed 's/.xml//g' > data/prepare_icfhr_2016/splits/train.txt
    cat data/prepare_icfhr_2016/PublicData/Validation/page/list | sed 's/.xml//g' > data/prepare_icfhr_2016/splits/val.txt
    # Create a test.txt from listing the Test-ICFHR-2016/page
    ls data/prepare_icfhr_2016/Test-ICFHR-2016/page | sed 's/.xml//g' > data/prepare_icfhr_2016/splits/test.txt

    # Execute a python script to extract the info from the xml files and create the lines and transcriptions folders
    # TODO: Implement the python script
    
  fi


