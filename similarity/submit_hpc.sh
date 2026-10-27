#!/bin/bash
# =============================================================
# SLURM Job Script — Similarity-Based GNN Training
# =============================================================
# Adjust the SBATCH directives to match your cluster's partitions.
# Submit with: sbatch submit_hpc.sh
# =============================================================

#SBATCH --job-name=sim_gnn
#SBATCH --output=slurm_%j.out          # stdout + stderr → slurm_<jobid>.out
#SBATCH --error=slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8              # DataLoader workers
#SBATCH --mem=64G                      # 20k graphs × ~200kB each ≈ 4GB + overhead
#SBATCH --time=24:00:00               # 24h wall time
#SBATCH --partition=gpu                # change to your GPU partition name
#SBATCH --gres=gpu:1                   # 1 GPU; set gpu:4 for DataParallel

# ----- Optional: email notification -----
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=your_email@university.edu

# =============================================================
# Environment setup
# =============================================================
echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "Working directory: $SLURM_SUBMIT_DIR"

# Load required modules (edit for your cluster's module system)
module load python/3.11
module load cuda/12.1          # match your PyTorch CUDA build

# Activate virtual/conda environment
# Option A: conda
# conda activate femgnn

# Option B: venv
source $HOME/envs/femgnn/bin/activate

# =============================================================
# Paths — edit these
# =============================================================
SIMILARITY_DIR=$SLURM_SUBMIT_DIR     # directory containing this script
DATA_ROOT=/scratch/$USER/generated_cases_20k   # HPC data path
RUN_DIR=/scratch/$USER/runs/sim_gnn_v1

# =============================================================
# Run training
# =============================================================
cd $SIMILARITY_DIR

python scripts/run_sim_training.py \
    --config configs/sim_gnn.yaml \
    --data.root      "$DATA_ROOT" \
    --output.run_dir "$RUN_DIR" \
    --training.device cuda \
    --training.batch_size 8 \
    --training.epochs 300

echo "Training finished at $(date)"
echo "Results in $RUN_DIR"

# =============================================================
# Optional: run test inference after training
# =============================================================
# python scripts/run_inference.py \
#     --checkpoint "$RUN_DIR/best.pt" \
#     --norm_stats  "$RUN_DIR/norm_stats.pt" \
#     --data_root   "$DATA_ROOT" \
#     --output_dir  "$RUN_DIR/predictions" \
#     --device      cuda
