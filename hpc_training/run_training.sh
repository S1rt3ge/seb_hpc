#!/bin/bash
module load anaconda/conda-24.9.2
module load python/3.9.19
cd SEB-susliki/
cd hpc_training/
pip3 install -r requirements.txt
python3 fast_training.py
