#!/bin/bash
#SBATCH --job-name=2fast2detections
#SBATCH --partition=boost_usr_prod
#SBATCH --account=iscrc_mnlp26
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --output=logs/optuna_heads_%j.out
#SBATCH --error=logs/optuna_heads_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Load the Python module available on Leonardo
module load python/3.11.7

export HF_HUB_OFFLINE=1

# Activate the virtual environment
source /leonardo/home/userexternal/asathyan/.cvenv/bin/activate

# Prepend virtual environment's nvidia package libraries to LD_LIBRARY_PATH
# to avoid cluster-wide CUDA version mismatches
for dir in /leonardo/home/userexternal/asathyan/.cvenv/lib/python3.11/site-packages/nvidia/*/lib; do
    if [ -d "$dir" ]; then
        export LD_LIBRARY_PATH="$dir:$LD_LIBRARY_PATH"
    fi
done

# Print job information for debugging
echo "============================================="
# Make sure we run from the correct directory
cd /leonardo/home/userexternal/asathyan/2Fast2Detections/code

echo "Job started at: $(date)"
echo "Running on node: $SLURM_NODENAME"
echo "Current directory: $(pwd)"
echo "Python location: $(which python)"
echo "Allocated GPU Info:"
nvidia-smi
echo "============================================="

# Execute the main script
python optuna_heads.py --trials 60 --checkpoint ../checkpoints/final_multitask_model.pth

echo "============================================="
echo "Job finished at: $(date)"
echo "============================================="
