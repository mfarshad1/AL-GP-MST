# Efficient Free-Energy Landscape Construction Using Graph-Based Active Path Sampling

This repository contains code and analysis workflows supporting the paper:

**“Efficient Free-Energy Landscape Construction Using Graph-Based Active Path Sampling”**  
Mohsen Farshad and Gaurav Arya

**Acknowledgments:** Şafak Çallıoğlu and Yilong Zhou are acknowledged for their contributions, consistent with the manuscript.

## 🧠 Overview
This repository implements a graph-constrained active-learning workflow to reconstruct many-body potentials of mean force (PMFs) from sparse, expensive labels in collective-variable space. The workflow combines analytical toy landscapes, a realistic polymer-grafted nanoparticle (PGNP) triplet benchmark represented by a smooth oracle surface, Gaussian Process (GP) regression, uncertainty-guided sequential sampling, and order-independent cost analysis based on the minimum spanning tree (MST) of the sampled set.

The analytical toy and oracle-based PGNP benchmarks use:
- direct oracle-energy labels (`label_source=oracle_paid`)
- GP learning of the anchored many-body PMF (`target_mode=umb`)
- plain composite GP kernel (`kernel_mode=plain`)
- no symmetry enforcement for the toy cases (`symmetry_mode=none`)
- sorted-distance symmetry for the realistic PGNP benchmark (`symmetry_mode=sorted`)

The codebase also contains additional options such as LSQR labels, path integration, residual learning, invariant features, and structured kernels. The revised manuscript additionally includes an explicit-MD alanine dipeptide benchmark using integrated mean-force labels (`pathint`).

## 📈 Method Flow
- Discretize the feasible configuration space as a graph in interparticle-distance coordinates.
- Define oracle landscapes:
  - 2D toy diagonal slice
  - 3D toy landscape with pair and nonadditive three-body structure
  - realistic 3D PGNP benchmark from a smooth fitted oracle
- Initialize a connected sampled set on the graph.
- Train a GP surrogate on the current labeled set.
- Rank candidate configurations using predictive uncertainty with spacing constraints.
- Select new queried points and add intermediate path-paid nodes to preserve connectivity.
- Evaluate reconstruction quality using RMSE, parity plots, and MST-based sampling cost.

## 🗂️ Analysis pipeline used in this project

Use the versioned scripts below for the revised benchmark implementations. The older `GP-*.py` files are retained for historical reference and should not be used for the revised 2D validation analysis. The corrected 2D script excludes held-out nodes from both queries and connecting paths, checks training/validation disjointness during learning, and exports `validation_nodes.csv`.

### 1) Two-dimensional toy landscape
Run the 2D diagonal-slice active-learning workflow:

```bash
python Stage_2_GP-toy2-2D-v18-v19_STRICTVAL.py \
  --start_mode warm \
  --target_mode umb \
  --symmetry_mode none \
  --kernel_mode plain \
  --label_source oracle_paid

```

### 2) Three-dimensional toy landscape
Run the 3D analytical active-learning workflow:

```bash
python Stage_2_GP-toy2-3D-v18-v17.py \
  --start_mode warm \
  --target_mode umb \
  --symmetry_mode none \
  --kernel_mode plain \
  --label_source oracle_paid

```

### 3) Three-dimensional MD-PIP landscape
Run the 3D PGNP active-learning workflow:

```bash
python Stage_2_GP-Umb-3D-v18-v17.py \
  --start_mode cold \
  --target_mode umb \
  --symmetry_mode sorted \
  --kernel_mode plain \
  --label_source oracle_paid
```

### 4) Alanine dipeptide (periodic two-dimensional CVs)

The versioned production controller is [`Stage_2_GP-alanine-2D-v18-v18.py`](Stage_2_GP-alanine-2D-v18-v18.py). It uses fixed-CV CP2K dynamics for mean-force observations, connected path integration, and a periodic 50-by-50 grid. The benchmark starts with 50 nodes on the periodic seam, uses five queries per active-learning step, and runs up to 50 steps with one replica per sampled node (10 ps equilibration and 40 ps production).

This controller is the original CRC-specific production source. It requires adapting the storage paths and scheduler/module settings to the local HPC environment, as well as providing the staged inputs: `alanine_config.json`, the validated CP2K model/topology and campaign configuration in `ALANINE_CP2K_PATHINT_FINAL`, the FFT/static-transfer validation reports in `GAPS_CP2K_AUTOMATED_RUN`, and `simulations_essentials`. These input assets are not yet included in this repository; the script alone is not a complete standalone reproduction package. Inspect and configure the environment-specific settings before execution.
