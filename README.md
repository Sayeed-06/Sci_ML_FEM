# Scientific Machine Learning Surrogates for FEM Plate Mechanics

This repository develops scientific machine-learning surrogates for finite-element simulations of concrete plates on Winkler foundations. The models learn full nodal fields—vertical deflection and two bottom-surface normal stresses—from plate geometry, material properties, thermal loading, foundation stiffness, and applied pressure.

The work has two connected parts:

1. **Physical-feature surrogates (`project/`)** compare graph neural networks, graph transformers, and a graph-conditioned DeepONet, trained with data-only and physics-informed objectives.
2. **Similarity-coordinate surrogates (`similarity/`)** encode classical plate similarity quantities and test whether physics-based nondimensionalisation improves learning and generalisation.

The repository also contains a global Sobol sensitivity study for spatial mean, median, 95th-percentile, and maximum absolute deflection.

## What was done

### 1. FEM cases were converted into graphs

Each ISLAB-style case contains `input.in2` and `input.ou2` files. The parsers recover:

- grid coordinates and plate geometry;
- elastic modulus `E`, Poisson ratio, thickness, thermal expansion and temperature difference;
- Winkler foundation modulus and pressure/load footprint;
- nodal deflection `w` and bottom-surface stresses `sigma_x`, `sigma_y`.

Every FEM node becomes a graph node. Horizontal, vertical, and diagonal grid connections become bidirectional graph edges. Targets remain nodal fields in physical units after denormalisation.

The original physical-feature graph uses 11 node features. The expanded DeepONet workflow adds thermal-expansion coefficient and pressure magnitude, producing 13 features without overwriting the legacy cache.

### 2. Multiple surrogate architectures were compared

- **GNN:** GINE-style message passing with edge features.
- **Graph Transformer:** attention-based graph propagation.
- **DeepONet:** a graph branch network plus a coordinate trunk network with Fourier features and cross-attention.
- **PINN variants:** data loss augmented with plate-equation, boundary, and stress-consistency terms.

All models predict three quantities per node:

```text
[w, sigma_x, sigma_y]
```

### 3. A similarity representation was implemented

The similarity pipeline transforms the raw plate problem into:

- spatial coordinates `x/l` and `y/l`;
- radius of relative stiffness `l`;
- nondimensional thermal-gradient parameter `phi`;
- pressure-to-self-weight ratio `P/Q`.

Global similarity variables condition the graph network through FiLM layers. Physics-consistent augmentation applies edge dropout and linear load scaling. Architectural decisions and their motivation are documented in [`similarity/ARCHITECTURE_DECISIONS.md`](similarity/ARCHITECTURE_DECISIONS.md).

### 4. Surrogates were validated against held-out FEM cases

The baseline model comparison produced the following nodal deflection results:

| Model | Deflection R2 | Deflection RMSE |
|---|---:|---:|
| GNN + MSE | 0.504 | 0.0721 |
| GNN + PINN | 0.449 | 0.0759 |
| Graph Transformer + MSE | 0.749 | 0.0512 |
| Graph Transformer + PINN | 0.891 | 0.0337 |
| DeepONet + MSE | **0.932** | **0.0268** |
| DeepONet + PINN | 0.930 | 0.0270 |

The expanded physical-feature DeepONet was trained on a reproducible 6,000-case subset and evaluated on 900 held-out cases. Its nodal deflection R2 was **0.952**. For the Sobol quantities of interest, held-out R2 values were:

| Spatial statistic of absolute deflection | Expanded DeepONet R2 | Similarity DeepONet R2 |
|---|---:|---:|
| Mean | 0.980 | 0.897 |
| Median | 0.972 | 0.909 |
| 95th percentile | 0.870 | 0.702 |
| Maximum | 0.833 | 0.647 |

The similarity model used the full 20,000-case split with 3,000 held-out test cases. Mean and median conclusions are more reliable than peak-response conclusions, particularly for the similarity model.

### 5. Global Sobol sensitivity analysis was performed

Six independent physical inputs were varied over their observed ranges:

```text
E, thickness, k_winkler, delta_T, alpha_T, pressure_q
```

The plate geometry and load footprint were fixed at a representative `24 x 45` FEM grid (1,080 nodes). For each surrogate, scrambled Sobol samples were evaluated at base sizes `N = 256, 512, 1024`. With six variables and second-order indices disabled, this required:

```text
N(D + 2) = 8N
```

or 14,336 surrogate evaluations per model across the three convergence levels. Five hundred bootstrap resamples were used for 95% uncertainty intervals.

Main finding: **Winkler foundation stiffness dominates deflection variance in both representations**. Thickness is the main secondary factor for global response, while pressure becomes more visible for peak response. The two models differ in the magnitude assigned to pressure and interactions, which is reported as model-form uncertainty rather than averaged away.

See:

- [`project/SOBOL_SENSITIVITY_GUIDE.md`](project/SOBOL_SENSITIVITY_GUIDE.md)
- [`project/runs/sobol_expanded/SOBOL_RESULTS_REPORT.md`](project/runs/sobol_expanded/SOBOL_RESULTS_REPORT.md)
- [`similarity/runs/sobol_similarity/SOBOL_SIMILARITY_RESULTS_REPORT.md`](similarity/runs/sobol_similarity/SOBOL_SIMILARITY_RESULTS_REPORT.md)
- [`similarity/runs/sobol_similarity/project_similarity_comparison.png`](similarity/runs/sobol_similarity/project_similarity_comparison.png)

## Repository structure

```text
.
|-- project/
|   |-- configs/             # physical-feature model configurations
|   |-- dataset/             # FEM-to-graph construction and normalisation
|   |-- models/              # GNN, graph transformer and DeepONet
|   |-- losses/              # MSE and PINN objectives
|   |-- training/            # training loop and checkpoint handling
|   |-- evaluation/          # physical-unit metrics
|   |-- scripts/             # training, inference, Sobol and plotting CLIs
|   `-- runs/sobol_expanded/ # reviewed Sobol outputs
|-- similarity/
|   |-- configs/             # similarity-model configurations
|   |-- dataset/             # similarity-coordinate graph pipeline
|   |-- models/              # FiLM GNN, transformer and DeepONet
|   |-- training/            # similarity training and PINN loss
|   |-- evaluation/          # physical-unit evaluation
|   |-- scripts/             # training, inference and Sobol CLIs
|   `-- runs/sobol_similarity/
`-- README.md
```

## Installation

Python 3.11 or 3.12 is recommended.

```bash
git clone https://github.com/Sayeed-06/Sci_ML_FEM.git
cd Sci_ML_FEM
python -m venv .venv
```

Activate the environment, then install the requirements for the workflow you want:

```bash
pip install -r project/requirements.txt
# or
pip install -r similarity/requirements.txt
```

Install a PyTorch/PyG build compatible with your CPU or CUDA version. RTX 50-series GPUs require a recent CUDA 12.8-compatible PyTorch build.

## Dataset layout

Generated FEM data are intentionally not committed because the full dataset is many gigabytes. Place cases in `project/generated_cases/` or override `data.root` in a configuration file.

```text
generated_cases/
|-- case_00001_.../
|   |-- input.in2
|   `-- input.ou2
|-- case_00002_.../
|   |-- input.in2
|   `-- input.ou2
`-- ...
```

## Training

From `project/`:

```bash
python scripts/run_training.py --config configs/gnn_mse.yaml
python scripts/run_training.py --config configs/transformer_pinn.yaml
python scripts/run_training.py --config configs/deeponet_mse.yaml
```

Expanded DeepONet:

```bash
python scripts/run_training.py --config configs/deeponet_mse_expanded.yaml
```

From `similarity/`:

```bash
python scripts/run_sim_training.py --config configs/sim_gnn.yaml
python scripts/run_sim_training.py --config configs/sim_deeponet.yaml
```

Configuration values can be overridden at the command line, for example:

```bash
python scripts/run_training.py --config configs/gnn_mse.yaml --training.device cuda --training.epochs 100
```

## Sobol analysis

Physical-feature model, from `project/`:

```bash
python scripts/evaluate_expanded_qoi.py
python scripts/run_sobol_qoi_analysis.py
python scripts/plot_sobol_bootstrap_summary.py \
  --input runs/sobol_expanded/sobol_bootstrap_resamples.csv \
  --output runs/sobol_expanded/sobol_bootstrap_summary.png
python scripts/plot_sobol_convergence.py \
  --input runs/sobol_expanded/sobol_convergence_summary.csv \
  --output runs/sobol_expanded/sobol_convergence.png
```

Similarity model, from `similarity/`:

```bash
python scripts/evaluate_similarity_qoi.py
python scripts/run_sobol_qoi_analysis.py
python scripts/plot_project_similarity_comparison.py \
  --project-dir ../project/runs/sobol_expanded \
  --similarity-dir runs/sobol_similarity \
  --output runs/sobol_similarity/project_similarity_comparison.png
```

## Reproducibility notes

- Dataset splits and Sobol sampling use seed `42`.
- Standard split: 70% training, 15% validation, 15% testing.
- Targets are normalised using training-set statistics and reported in physical units after denormalisation.
- Sobol input ranges, representative geometry, sample sizes and bootstrap settings are stored with the output metadata.
- Checkpoints and normalisation tensors are excluded from Git because of their size. Re-run training to reproduce them.

## Limitations

- The Sobol study establishes sampling convergence, not FEM mesh convergence; only one fixed `24 x 45` mesh was used for sensitivity evaluation.
- Sobol indices are conditional on the selected parameter ranges, independent-uniform distributions, fixed geometry and trained surrogate.
- Peak deflection has larger surrogate error than mean or median deflection.
- The existing similarity transformation has a documented unit-convention concern for unit weight and Winkler stiffness. The published similarity Sobol outputs describe the as-trained checkpoint; verify the ISLAB unit convention before treating them as final physical evidence.
- Second-order Sobol indices were not computed. `ST - S1` indicates interaction involvement but does not identify the interacting partner.

## Large artifacts excluded from Git

The local workspace also contains generated FEM cases, cached graph tensors, virtual environments, CUDA/PyTorch installers, training checkpoints and compressed backups. These are excluded by `.gitignore` because they are reproducible, machine-specific or exceed GitHub's file-size limits.

