# Graph-based active learning for landscape reconstruction with minimum-spanning-tree cost

This repository contains code and analysis workflows supporting the paper:

**“Graph-based active learning for landscape reconstruction with minimum-spanning-tree cost”**  
Mohsen Farshad, Şafak Çallıoğlu, Yilong Zhou, Gaurav Arya

## 🧠 Overview
This repository implements a graph-constrained active-learning workflow to reconstruct many-body potentials of mean force (PMFs) from sparse, expensive labels in collective-variable space. The workflow combines analytical toy landscapes, a realistic polymer-grafted nanoparticle (PGNP) triplet benchmark represented by a smooth oracle surface, Gaussian Process (GP) regression, uncertainty-guided sequential sampling, and order-independent cost analysis based on the minimum spanning tree (MST) of the sampled set.

The main paper focuses on:
- direct oracle-energy labels (`label_source=oracle_paid`)
- GP learning of the anchored many-body PMF (`target_mode=umb`)
- plain composite GP kernel (`kernel_mode=plain`)
- no symmetry enforcement for the toy cases (`symmetry_mode=none`)
- sorted-distance symmetry for the realistic PGNP benchmark (`symmetry_mode=sorted`)

The codebase also contains additional options such as LSQR labels, residual learning, invariant features, and structured kernels, but those are not part of the main results reported in the current paper.

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

### 1) Two-dimensional toy landscape
Run the 2D diagonal-slice active-learning workflow:

```bash
python Stage_2_GP-toy2-2D-v18-v14.py \
  --start_mode warm \
  --target_mode umb \
  --symmetry_mode none \
  --kernel_mode plain \
  --label_source oracle_paid
