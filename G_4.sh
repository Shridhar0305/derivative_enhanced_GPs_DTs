#!/bin/bash -l

###############################
# SLURM RESOURCE SETUP
#SBATCH --account=hochhalter-np     
#SBATCH --partition=hochhalter-shared-np 
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5                   
#SBATCH --gres=gpu:0                         
#SBATCH --mem=32gb
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shridhar.vashishtha@utah.edu
#SBATCH --output=./G_4.txt
###############################


# FIXED COMMANDS
# For Miniconda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /uufs/chpc.utah.edu/common/home/u1589024/miniconda3/envs/main_a100_py39
NOTEBOOK="G_4"
jupyter nbconvert --to script --output ${NOTEBOOK} ${NOTEBOOK}.ipynb

###############################

python -u ${NOTEBOOK}.py
rm ${NOTEBOOK}.py
