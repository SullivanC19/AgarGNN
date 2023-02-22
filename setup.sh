#!/bin/bash
# This script is used to setup the conda environment for the project
conda env create --force -f environment.yml
conda activate simple-agar-gnn