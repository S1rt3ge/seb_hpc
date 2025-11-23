#!/bin/sh
#PBS -N Susliki-SEB
#PBS -q batch
#PBS -l nodes=1:ppn=16:gpus=4
#PBS -j oe

cd $PBS_O_WORKDIR

echo "Job started: $(date)"
echo "Working dir: $(pwd)"

module purge
module load anaconda/conda-24.9.2
module load python/3.9.19
module load AI/pytorch-1.13.1-gpu-conda

pip install --user -q torch==2.0.1
pip install --user -q pandas==1.5.3
pip install --user -q numpy==1.25.2
pip install --user -q scikit-learn==1.3.2
pip install --user -q statsmodels==0.14.1
pip install --user -q joblib==1.3.2

python train_final.py

echo "Job finished: $(date)"