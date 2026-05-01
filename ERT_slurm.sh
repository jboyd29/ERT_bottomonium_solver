#!/bin/bash
#SBATCH --job-name=ERT_ENSEMBLE
#SBATCH --output=ert_%j.out
#SBATCH --error=ert_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=206
#SBATCH --mem=356G
#SBATCH --time=24:00:00
#SBATCH --nodelist=quark

# --- Environment Setup ---
# Initialize Conda and activate your qcd environment
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate qcd

# --- Performance Tuning ---
# CRITICAL: Prevent sub-libraries (NumPy/OpenBLAS) from spawning 
# their own thread pools, which would cause "thread thrashing"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# --- Execution ---
echo "Job started on $(hostname) at $(date)"
#echo "Processing 300 trajectories with 64 workers..."

# Run with unbuffered output so you can tail the .out file in real-time
python3 -u ERT_multi_script.py

echo "Job finished at $(date)"
