#!/bin/bash

base_files=$1

# Get files througth grep 
files=$(ls checkpoints | grep $base_files)

# Move them having as base folder checkpoints/

for x in $files; do
    # echo mv $x `echo $x | cut -c 7-`
    # Moving but having as base folder checkpoints/
    mv checkpoints/$x checkpoints/`echo $x | cut -c 5-`
done