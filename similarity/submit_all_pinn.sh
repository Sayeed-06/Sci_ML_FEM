#!/bin/bash
# =============================================================
# SLURM Job Script — Submit ALL Similarity-Based PINN Models
# =============================================================
# Run with: bash submit_all_pinn.sh
# =============================================================

SIMILARITY_DIR=/iitgn/homedirs/24110320/projects/srip_re/similarity
DATA_ROOT=$SIMILARITY_DIR/generated_cases   # HPC data path
LOSS_TYPE="pinn_biharmonic"                        # Options: pinn_stress or pinn_biharmonic

echo "Submitting PINN training jobs to SLURM..."

# 1. SimGNN PINN
RUN_DIR=/scratch/$USER/runs/sim_gnn_pinn
sbatch \
  --job-name=sim_gnn_pinn \
  --output=slurm_%j_sim_gnn_pinn.out \
  --error=slurm_%j_sim_gnn_pinn.err \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G \
  --time=24:00:00 --partition=v100gpu --gres=gpu:1 \
  --wrap="
module load python/3.11
module load cuda/12.1
source \$HOME/envs/femgnn/bin/activate
cd $SIMILARITY_DIR
python scripts/run_sim_training.py \
    --config configs/sim_gnn.yaml \
    --data.root \"$DATA_ROOT\" \
    --output.run_dir \"$RUN_DIR\" \
    --training.device cuda \
    --training.batch_size 2 \
    --training.epochs 300 \
    --training.loss_type \"$LOSS_TYPE\"
"
echo "  [✓] Submitted SimGNN PINN"

# 2. Graph Transformer PINN
RUN_DIR=/scratch/$USER/runs/graph_transformer_pinn
sbatch \
  --job-name=sim_gt_pinn \
  --output=slurm_%j_sim_gt_pinn.out \
  --error=slurm_%j_sim_gt_pinn.err \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G \
  --time=24:00:00 --partition=v100gpu --gres=gpu:1 \
  --wrap="
module load python/3.11
module load cuda/12.1
source \$HOME/envs/femgnn/bin/activate
cd $SIMILARITY_DIR
python scripts/run_sim_training.py \
    --config configs/sim_transformer.yaml \
    --data.root \"$DATA_ROOT\" \
    --output.run_dir \"$RUN_DIR\" \
    --training.device cuda \
    --training.batch_size 2 \
    --training.epochs 300 \
    --training.loss_type \"$LOSS_TYPE\"
"
echo "  [✓] Submitted Graph Transformer PINN"

# 3. DeepONet PINN
RUN_DIR=/scratch/$USER/runs/sim_deeponet_pinn
sbatch \
  --job-name=sim_don_pinn \
  --output=slurm_%j_sim_don_pinn.out \
  --error=slurm_%j_sim_don_pinn.err \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G \
  --time=24:00:00 --partition=v100gpu --gres=gpu:1 \
  --wrap="
module load python/3.11
module load cuda/12.1
source \$HOME/envs/femgnn/bin/activate
cd $SIMILARITY_DIR
python scripts/run_sim_training.py \
    --config configs/sim_deeponet.yaml \
    --data.root \"$DATA_ROOT\" \
    --output.run_dir \"$RUN_DIR\" \
    --training.device cuda \
    --training.batch_size 2 \
    --training.epochs 300 \
    --training.loss_type \"$LOSS_TYPE\"
"
echo "  [✓] Submitted DeepONet PINN"

echo ""
echo "All 3 PINN models have been submitted to the cluster queue!"
echo "Check their status with: squeue -u \$USER"
