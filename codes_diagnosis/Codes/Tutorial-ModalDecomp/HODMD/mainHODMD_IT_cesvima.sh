#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --partition=standard
#SBATCH --job-name=mainHODMD_IT_cesvima
#SBATCH --nodes=1-1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=160:00:00
#SBATCH --output=output-mainHODMD_IT_cesvima-%j.log
#SBATCH --error=error-mainHODMD_IT_cesvima-%j.log
##------------------------ End job description ------------------------

module purge && module load intel && module load Python/3.10.8-GCCcore-12.2.0 && module load intelcuda && module load CUDA/11.3.1 && module load cuDNN/8.2.1.32-CUDA-11.3.1

source venv/bin/activate

torchrun --rdzv-endpoint=0.0.0.0:29707 HODMD/mainHODMD_IT_cesvima.py
