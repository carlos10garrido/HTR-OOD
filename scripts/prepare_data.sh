#!/bin/bash

# Prepare and organize data for training, validation and testing for each dataset
# It is assumed that there is a folder called $folder_name/$dataset with the raw data. 
# It will create (or modify) a split file for each dataset and each stage (train, val, test)
folder_name=$1 # It is the original folder name in which original files are stored
datasets="iam rimes washington bentham saint_gall rodrigo icfhr_2016"

mkdir -p $folder_name/htr_datasets # Real dataset folder
mkdir -p $folder_name/synth/ # Synth dataset folder

for dataset in $datasets
do
  echo "Preparing $dataset dataset"
  mkdir -p $folder_name/$dataset

  # IAM
  if [ $dataset == "iam" ]
  then
    # Descompress the dataset (lines.tar, xml.tar and splits.zip) 
    mkdir -p $folder_name/$dataset/iam/
    mkdir -p $folder_name/$dataset/iam/lines
    mkdir -p $folder_name/$dataset/iam/xml
    mkdir -p $folder_name/$dataset/iam/splits
    # Mantain the folder structure
    tar -xf $folder_name/lines.tar -C $folder_name/$dataset/iam/lines
    tar -xf $folder_name/xml.tar -C $folder_name/$dataset/iam/xml
    unzip -q $folder_name/splits.zip -d $folder_name/$dataset/iam/

    # Create a train.txt, val.txt and test.txt file and paste the content of train.uttlist, validation.uttlist and test.uttlist
    mv $folder_name/$dataset/iam/splits/train.uttlist $folder_name/$dataset/iam/splits/train.txt
    mv $folder_name/$dataset/iam/splits/validation.uttlist $folder_name/$dataset/iam/splits/val.txt
    mv $folder_name/$dataset/iam/splits/test.uttlist $folder_name/$dataset/iam/splits/test.txt

  fi
  
  # Rimes
  if [ $dataset == "rimes" ]
  then
    # Descompress the dataset (RIMES-2011-Lines.zip) -> Contains Images, Sets, Transcriptions
    unzip -q $folder_name/RIMES-2011-Lines.zip -d $folder_name/$dataset 

    # Just rename the files with the correct name since they exist
    mv $folder_name/$dataset/RIMES-2011-Lines/Sets/TrainLines.txt $folder_name/$dataset/RIMES-2011-Lines/Sets/train.txt
    mv $folder_name/$dataset/RIMES-2011-Lines/Sets/ValidationLines.txt $folder_name/$dataset/RIMES-2011-Lines/Sets/val.txt
    mv $folder_name/$dataset/RIMES-2011-Lines/Sets/TestLines.txt $folder_name/$dataset/RIMES-2011-Lines/Sets/test.txt

  fi

  # Washington
  if [ $dataset == "washington" ]
  then
    # Descompress the dataset (washingtondb-v1.0) -> Contains $folder_name/, ground_truth/, sets/
    unzip -q $folder_name/washingtondb-v1-3.0.zip -d $folder_name/$dataset
    
    # We will use cv1 for the splits. Just rename sets/cv1/valid.txt to sets/cv1/val.txt
    mv $folder_name/$dataset/washingtondb-v1-3.0/sets/cv1/valid.txt $folder_name/$dataset/washingtondb-v1-3.0/sets/cv1/val.txt
  fi

  # Bentham
  if [ $dataset == "bentham" ]
  then
    echo "This one will take a while..."
    # Descompress the dataset (BenthamDatasetR0-GT.tbz, BenthamDatasetR0-Images.tbz)
    tar -xf $folder_name/BenthamDatasetR0-GT.tbz -C $folder_name/$dataset
    echo "Done!"

    # Create a train.txt, val.txt and test.txt file and paste the content of TrainLines.lst, ValidationLines.lst and TestLines.lst
    # Just rename the files with the correct name since they exist
    mv $folder_name/$dataset/BenthamDatasetR0-GT/Partitions/TrainLines.lst $folder_name/$dataset/BenthamDatasetR0-GT/Partitions/train.txt
    mv $folder_name/$dataset/BenthamDatasetR0-GT/Partitions/ValidationLines.lst $folder_name/$dataset/BenthamDatasetR0-GT/Partitions/val.txt
    mv $folder_name/$dataset/BenthamDatasetR0-GT/Partitions/TestLines.lst $folder_name/$dataset/BenthamDatasetR0-GT/Partitions/test.txt

    # diff -r -q $folder_name/$dataset $folder_name/htr_datasets/bentham
  fi

  # Saint Gall
  if [ $dataset == "saint_gall" ]
  then
    # Descompress the dataset (saintgalldb-v1.0) -> Contains $folder_name/, ground_truth/, sets/
    unzip -q $folder_name/saintgalldb-v1.0.zip -d $folder_name/$dataset

    # Just rename sets/cv1/valid.txt to sets/cv1/val.txt
    cat $folder_name/$dataset/saintgalldb-v1.0/sets/valid.txt > $folder_name/$dataset/saintgalldb-v1.0/sets/val.txt

    # diff -r -q $folder_name/$dataset $folder_name/htr_datasets/saint_gall
  fi

  # Rodrigo
  if [ $dataset == "rodrigo" ]
  then
    # Descompress the dataset (Rodrigo corpus 1.0.0.tar) -> Contains images/, text/, partitions/
    mkdir -p $folder_name/$dataset/'Rodrigo corpus 1.0.0'
    tar -xf $folder_name/'Rodrigo corpus 1.0.0.tar' -C $folder_name/$dataset/'Rodrigo corpus 1.0.0'

    # Just rename the files with the correct name since they exist
    mv $folder_name/$dataset/'Rodrigo corpus 1.0.0'/partitions/validation.txt $folder_name/$dataset/'Rodrigo corpus 1.0.0'/partitions/val.txt

    # diff -r -q $folder_name/$dataset $folder_name/htr_datasets/rodrigo
  fi
  
  # ICFHR 2016
  if [ $dataset == "icfhr_2016" ]
  then
    # Descompress the dataset (Train-And-Val-ICFHR-2016.tar) -> Contains Public$folder_name/Training, Public$folder_name/Validation. 
    tar -xf $folder_name/Train-And-Val-ICFHR-2016.tar -C $folder_name/$dataset
    
    # Descompress the test dataset (Test-ICFHR-2016.tar) -> Contains the images (page level!)
    tar -xf $folder_name/Test-ICFHR-2016.tar -C $folder_name/$dataset

    # Create the new structure since it is a little bit clunky. It does not have the data by lines!!! 
    # We need to extract the lines from the xml files (at the end)
    # Create a lines, transcriptions and splits folder
    mkdir -p $folder_name/$dataset/lines
    mkdir -p $folder_name/$dataset/transcriptions
    mkdir -p $folder_name/$dataset/splits
    # Create a train.txt, val.txt from copying the list in Training/page/list and Validation/page/list (removing the final .xml)
    cat $folder_name/$dataset/PublicData/Training/page/list | sed 's/.xml//g' > $folder_name/$dataset/splits/train.txt
    cat $folder_name/$dataset/PublicData/Validation/page/list | sed 's/.xml//g' > $folder_name/$dataset/splits/val.txt
    # Create a test.txt from listing the Test-ICFHR-2016/page
    ls $folder_name/$dataset/Test-ICFHR-2016/page | sed 's/.xml//g' > $folder_name/$dataset/splits/test.txt

    # Execute a python script to extract the info from the xml files and create the lines and transcriptions folders
    script_python="python ./scripts/extract_ichr_2016.py $folder_name/$dataset" 
    echo $script_python
    eval $script_python

    # diff -r -q $folder_name/$dataset $folder_name/htr_datasets/icfhr_2016
  fi

  # Remove all paths that contain __MACOSX in $folder_name/$datasettaset
  find $folder_name/$dataset/ -name "__MACOSX" -type d -exec rm -rf {} \;

  # Move all to the real dataset folder
  mv $folder_name/$dataset/ $folder_name/htr_datasets

done


