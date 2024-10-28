#!/bin/bash

# Exit immediately if any command fails
set -e

# Function to handle errors
handle_error() {
    echo "ERROR: There was an issue with training using ${trainer_name}"
    exit 1
}

# Trap errors to call the error handler function
trap 'handle_error' ERR

# Directory containing the trainer scripts
TRAINER_DIR="./"  # Adjust this if your scripts are in a different directory

# Iterate over all trainer scripts in the directory
for trainer_script in "${TRAINER_DIR}"nnUNetTrainer*.py; do
    # Extract the trainer name by removing the directory path and ".py" extension
    trainer_name=$(basename "${trainer_script}" .py)
    
    # Print and execute the training command
    echo "nnUNetv2_train 999 3d_fullres 0 -tr ${trainer_name}"
    nnUNetv2_train 999 3d_fullres 0 -tr "${trainer_name}"
done
