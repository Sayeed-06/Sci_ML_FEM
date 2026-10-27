# Architecture & Design Decisions
**Project:** Similarity-Based GNN Surrogate for Plate Mechanics

This document tracks the major deviations, improvements, and architectural decisions made compared to the original baseline idea, along with the physical and mathematical rationale for each change.

---

### 1. Introduction of Similarity Coordinates & FiLM Conditioning
**Change:** 
Instead of feeding absolute physical coordinates (`x, y`) and global parameters (`E, h, q`) directly as node features into a standard GNN, we transformed the inputs into **Similarity Coordinates** (`x/l, y/l`) normalised by the radius of relative stiffness ($l$). We then injected the global similarity parameters ($l$, $\phi$, $P/Q$) into the network using a **FiLM (Feature-wise Linear Modulation)** layer.

**Why:**
- **Physics Inductive Bias:** Solid mechanics dictates that plates with the same non-dimensional geometry ($L/l$, $W/l$) behave identically up to a scaling factor. By using similarity coordinates, the GNN doesn't have to "re-learn" basic scaling laws from scratch.
- **Role Separation:** The message-passing layers focus exclusively on local PDE integration (local field gradients), while the FiLM layer handles the global non-linear scaling of the entire solution.

---

### 2. Abandoning Graph Transformers for Deep GINE (Message Passing)
**Change:** 
We implemented and tested a `SimGraphTransformer` (using PyG `TransformerConv`) but ultimately abandoned it in favor of a deeper standard GNN (`SimGNN` using `GINEConv` with `num_layers=8` and `hidden_dim=256`).

**Why:**
- **Overfitting & Capacity:** The Transformer expanded the parameter count to ~2.58 million and severely overfit the training data (memorizing cases instead of learning physics).
- **The "Stencil" Effect:** Attention mechanisms are designed to figure out "which distant neighbor is important" (like in a sentence). However, on a uniform FEM grid solving a PDE, **all immediate neighbors are equally important** purely based on their distance. The GNN's simple sum-aggregation mathematically mirrors a finite-difference/finite-element solver perfectly. The Transformer wasted capacity learning attention weights that the physics already provided for free.

---

### 3. Physics-Consistent Data Augmentation
**Change:** 
We introduced a custom `AugmentedSimDataset` wrapper in the training loop with two dynamic augmentations:
1. **DropEdge (10%)**: Randomly dropping edges during training.
2. **Linear Load Scaling ($\pm 20\%$)**: Multiplying the input load (`q_local`, `PQ`) and output targets ($w, \sigma_x, \sigma_y$) by a random factor $\alpha \in [0.8, 1.2]$ on-the-fly, before normalisation.

**Why:**
- **Over-smoothing Prevention:** DropEdge forces the deeper 8-layer GNN to learn robust pathways for stress propagation and prevents node representations from becoming indistinguishable.
- **Infinite Valid Data:** Because the problem domain is **Linear Elasticity**, scaling the load by $\alpha$ mathematically guarantees that the displacements and stresses scale by exactly $\alpha$. This provides infinite synthetic data points that perfectly obey the physics, completely eliminating the model's ability to just memorize the training set.

---

### 4. Robust Dataset Parsing (Case Insensitivity)
**Change:** 
Modified the PyG dataset parser to match folder prefixes case-insensitively (`.lower().startswith("case_")`).

**Why:**
- **HPC Compatibility:** The generated FEM datasets on the HPC environment used lowercase `case_20000_...` structures, whereas the local code expected uppercase `Case_`. This ensures the code runs seamlessly across environments without requiring massive file renaming operations.
