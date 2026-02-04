#!/bin/bash
#SBATCH --job-name=threshold_calc
#SBATCH --output=logs/threshold_calc_%j.out
#SBATCH --error=logs/threshold_calc_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# Threshold Calculation - Phase 1
# Computes P90 and P99 percentile thresholds from daily precipitation data

# Exit on error
set -e

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate conda environment
# Adjust the path to your conda installation if needed
source "${CONDA_PREFIX:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate threshold-calc

# Run threshold calculation
python -m src.climatology config/config.yaml

echo "Threshold calculation complete"
