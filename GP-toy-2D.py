#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# Stage-2 GP Umb 2D  (SIBLING OF THE 3D SCRIPT)
#
# Results:
#   - Oracle diag slice 2D/3D
#   - Oracle W3 diag slice 2D/3D
#   - Optional full-grid LSQR diagnostic slice
#   - Optional per-step LSQR paid-vs-full-grid parity
#   - Per-step: student ¼ diag slice 2D/3D
#   - Per-step: student Ã diag slice 2D/3D
#   - Per-step: absolute error |pred - true| diag slice 2D/3D
#   - Per-step: extracted W3 diag slice 2D/3D
#   - Per-step: parity (train+val), RMSE trace, MST trace
#   - Summary RMSE vs iter, MST vs iter
# =============================================================================

import os
import tempfile
from pathlib import Path

def _setup_mpl_tex_cache_to_tmp():
    root = Path(tempfile.mkdtemp(prefix="mpl_tex_", dir="/tmp"))
    os.environ["MPLCONFIGDIR"]   = str(root / "mplconfig")
    os.environ["XDG_CACHE_HOME"] = str(root / "xdg_cache")
    os.environ["XDG_CONFIG_HOME"]= str(root / "xdg_config")
    os.environ["TEXMFVAR"]       = str(root / "texmfvar")
    os.environ["TEXMFCONFIG"]    = str(root / "texmfconfig")
    os.environ["TEXMFCACHE"]     = str(root / "texmfcache")
    os.environ["TMPDIR"]         = "/tmp"
    for k in ("MPLCONFIGDIR","XDG_CACHE_HOME","XDG_CONFIG_HOME","TEXMFVAR","TEXMFCONFIG","TEXMFCACHE"):
        Path(os.environ[k]).mkdir(parents=True, exist_ok=True)

_setup_mpl_tex_cache_to_tmp()

# =============================================================================
# Imports
# =============================================================================
import argparse
import csv
import heapq
import numpy as np
import matplotlib as mpl
# Match the latest toy/PIP plotting style: use the narrow hyphen-minus in numeric ticks.
mpl.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

import tensorflow as tf
import gpflow
from gpflow import utilities as gpf_util

from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.linalg import lsqr
from matplotlib.ticker import MaxNLocator, FuncFormatter

# =============================================================================
# CLI options
# =============================================================================
DEFAULT_START_MODE    = "warm"         # warm/cold
DEFAULT_TARGET_MODE   = "umb"          # umb/residual
DEFAULT_SYMM_MODE     = "none"         # sorted/invariants/none
DEFAULT_KERNEL_MODE   = "plain"        # structured/plain
DEFAULT_LABEL_SOURCE  = "oracle_paid"  # pathint|lsqr|oracle_paid|oracle_queried

RUN_TAG = "Toy2UmbDirect_StrictVal100"


def parse_cli():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--start_mode",
        choices=["warm", "cold"],
        default=DEFAULT_START_MODE,
        help="warm=copies previous GP params; cold=rebuild from scratch each step",
    )
    p.add_argument(
        "--target_mode",
        choices=["umb", "residual"],
        default=DEFAULT_TARGET_MODE,
        help="umb=GP trains on anchored Umb; residual=GP trains on Umb-(W2sum-W2_anchor)",
    )
    p.add_argument(
        "--symmetry_mode",
        choices=["sorted", "invariants", "none"],
        default=DEFAULT_SYMM_MODE,
        help="Permutation-invariant inputs on virtual triplet (d13,d23,d12) with d23=d13",
    )
    p.add_argument(
        "--kernel_mode",
        choices=["structured", "plain"],
        default=DEFAULT_KERNEL_MODE,
        help="structured=(1D+1D+1D)+3D, plain=use KERNEL_SPEC",
    )
    p.add_argument(
        "--label_source",
        choices=["pathint", "lsqr", "oracle_paid", "oracle_queried"],
        default=DEFAULT_LABEL_SOURCE,
        help=(
            "Label source for GP training: "
            "pathint=on-the-fly path integration of exact forces on paid graph; "
            "lsqr=LSQR from forces on paid graph; "
            "oracle_paid=direct oracle energies on ALL paid nodes; "
            "oracle_queried=direct oracle energies on QUERIED nodes only "
            "(still pays path nodes, but excludes them from training)."
        ),
    )
    return p.parse_args()

# =============================================================================
# TOY ORACLE PARAMETERS
# =============================================================================
GRAFT = "toy2"

D12_MIN, D12_MAX = 6.0, 11.0
D13_MIN, D13_MAX = 6.0, 11.0
D23_MIN, D23_MAX = D13_MIN, D13_MAX  # kept for sibling symmetry

GRID_N = 121
SEED = 12345

DO_FULLGRID_LSQR_DIAGNOSTIC = True
PLOT_FULLGRID_LSQR_DIAGNOSTIC = False

LSQR_PRINT_EQUATION_STATS = True
DW_CLIP = 40.0

# Continuous edge-integration controls.
# By default, LSQR uses Gauss-Legendre points along each paid edge instead of
# endpoint-averaged forces. Set LSQR_INTEGRATION_MODE=endpoint to recover the old
# behavior. PATHINT keeps the same quadrature control.
PATHINT_QUAD_ORDER = int(os.environ.get("PATHINT_QUAD_ORDER", 16))
LSQR_INTEGRATION_MODE = os.environ.get("LSQR_INTEGRATION_MODE", "continuous").lower()
LSQR_EDGE_QUAD_ORDER = int(os.environ.get("LSQR_EDGE_QUAD_ORDER", PATHINT_QUAD_ORDER))

# Pair LJ
LJ_EPS = 0.50
LJ_SIG = 7.00

# 3-body toy term
THETA0_DEG = 100.0
COS_TH0 = float(np.cos(np.deg2rad(THETA0_DEG)))
LAMBDA_3B = 2.00
GAMMA_3B = 0.60
R0_3B = LJ_SIG

# Landscape bumps
LANDSCALE = 20.0

# =============================================================================
# ORACLE FUNCTIONS
# =============================================================================
def lj_12_6(r, eps=LJ_EPS, sig=LJ_SIG):
    r = np.maximum(np.asarray(r, float), 1e-12)
    x = sig / r
    x6 = x**6
    return 4.0 * eps * (x6**2 - x6)


def dlj_12_6_dr(r, eps=LJ_EPS, sig=LJ_SIG):
    r = np.maximum(np.asarray(r, float), 1e-12)
    x = sig / r
    x6 = x**6
    return 24.0 * eps * (-2.0 * (x6**2) + x6) / r


def W2(d, p2=None):
    return lj_12_6(d)


def toy_three_body(
    d12,
    d13,
    d23,
    lam=LAMBDA_3B,
    cos_th0=COS_TH0,
    gamma=GAMMA_3B,
    r0=R0_3B,
):
    d12 = np.asarray(d12, float)
    d13 = np.asarray(d13, float)
    d23 = np.asarray(d23, float)
    denom = 2.0 * np.maximum(d12 * d13, 1e-12)
    cos_th = (d12**2 + d13**2 - d23**2) / denom
    cos_th = np.clip(cos_th, -1.0, 1.0)
    env = np.exp(-gamma * ((d12 + d13 + d23) - 3.0 * r0))
    return lam * (cos_th - cos_th0) ** 2 * env


def dU3_ddists(
    d12,
    d13,
    d23,
    lam=LAMBDA_3B,
    cos_th0=COS_TH0,
    gamma=GAMMA_3B,
    r0=R0_3B,
):
    eps = 1e-12
    d12 = np.asarray(d12, float)
    d13 = np.asarray(d13, float)
    d23 = np.asarray(d23, float)

    d12c = np.maximum(d12, eps)
    d13c = np.maximum(d13, eps)

    denom = 2.0 * d12c * d13c
    A = d12**2 + d13**2 - d23**2
    c = A / np.maximum(denom, eps)
    c = np.clip(c, -1.0, 1.0)

    env = np.exp(-gamma * ((d12 + d13 + d23) - 3.0 * r0))
    dc = c - cos_th0

    dA_dd12 = 2.0 * d12
    dA_dd13 = 2.0 * d13
    dA_dd23 = -2.0 * d23

    dden_dd12 = 2.0 * d13c
    dden_dd13 = 2.0 * d12c

    denom2 = np.maximum(denom**2, eps)
    dc_dd12 = (dA_dd12 * denom - A * dden_dd12) / denom2
    dc_dd13 = (dA_dd13 * denom - A * dden_dd13) / denom2
    dc_dd23 = (dA_dd23 * denom) / denom2

    denv = -gamma * env
    pref = lam
    dU_dd12 = pref * (2.0 * dc * dc_dd12 * env + (dc**2) * denv)
    dU_dd13 = pref * (2.0 * dc * dc_dd13 * env + (dc**2) * denv)
    dU_dd23 = pref * (2.0 * dc * dc_dd23 * env + (dc**2) * denv)
    return dU_dd12, dU_dd13, dU_dd23


def landscape_gaussians(d12, d13, d23):
    d12 = np.asarray(d12, float)
    d13 = np.asarray(d13, float)
    d23 = np.asarray(d23, float)
    terms = [
        (-2.2, 8.3 - 1.4, 8.15 + 0.55, 8.15 + 0.55, 0.50),
        (-2.2, 8.3 + 1.4, 8.15 - 0.55, 8.15 - 0.55, 0.50),
        (+2.1, 8.3 - 0.9, 8.15 - 0.75, 8.15 - 0.75, 0.50),
        (+2.1, 8.3 + 0.9, 8.15 + 0.75, 8.15 + 0.75, 0.50),
    ]
    U = 0.0
    for A, c12, c13, c23, w in terms:
        r2 = (d12 - c12) ** 2 + (d13 - c13) ** 2 + (d23 - c23) ** 2
        U += A * np.exp(-0.5 * r2 / (w**2))
    return U


def d_landscape_gaussians_ddists(d12, d13, d23):
    d12 = np.asarray(d12, float)
    d13 = np.asarray(d13, float)
    d23 = np.asarray(d23, float)
    terms = [
        (-2.2, 8.3 - 1.4, 8.15 + 0.55, 8.15 + 0.55, 0.50),
        (-2.2, 8.3 + 1.4, 8.15 - 0.55, 8.15 - 0.55, 0.50),
        (+2.1, 8.3 - 0.9, 8.15 - 0.75, 8.15 - 0.75, 0.50),
        (+2.1, 8.3 + 0.9, 8.15 + 0.75, 8.15 + 0.75, 0.50),
    ]
    dU12 = 0.0
    dU13 = 0.0
    dU23 = 0.0
    for A, c12, c13, c23, w in terms:
        w2 = float(w**2)
        r2 = (d12 - c12) ** 2 + (d13 - c13) ** 2 + (d23 - c23) ** 2
        term = A * np.exp(-0.5 * r2 / w2)
        fac = -1.0 / w2
        dU12 = dU12 + term * fac * (d12 - c12)
        dU13 = dU13 + term * fac * (d13 - c13)
        dU23 = dU23 + term * fac * (d23 - c23)
    return dU12, dU13, dU23


def deltaW3_raw(d12, d13, d23, p3=None):
    return toy_three_body(d12, d13, d23) + (LANDSCALE * landscape_gaussians(d12, d13, d23))


def Umb(d12, d13, d23, p2=None, p3=None):
    W2sum = W2(d12, p2) + W2(d13, p2) + W2(d23, p2)
    Vraw = deltaW3_raw(d12, d13, d23, p3)
    return W2sum + Vraw, Vraw, Vraw


def Umb_slice(d12, d13, p2=None, p3=None):
    d23 = d13
    return Umb(d12, d13, d23, p2, p3)


def W2sum_pairs(d12, d13, d23, p2=None):
    return W2(d12, p2) + W2(d13, p2) + W2(d23, p2)


def W2sum_slice(d12, d13, p2=None):
    d23 = d13
    return W2sum_pairs(d12, d13, d23, p2)


def triangle_mask(d12, d13, d23, eps=1e-12):
    d12 = np.asarray(d12, float)
    d13 = np.asarray(d13, float)
    d23 = np.asarray(d23, float)
    return (
        (d12 <= d13 + d23 + eps)
        & (d13 <= d12 + d23 + eps)
        & (d23 <= d12 + d13 + eps)
        & (d12 >= np.abs(d13 - d23) - eps)
        & (d13 >= np.abs(d12 - d23) - eps)
        & (d23 >= np.abs(d12 - d13) - eps)
    )


def slice_feasible_mask(d12, d13):
    d23 = d13
    return triangle_mask(d12, d13, d23)


def forces_Umb(d12, d13, d23):
    dU3_dd12, dU3_dd13, dU3_dd23 = dU3_ddists(d12, d13, d23)
    dL_dd12, dL_dd13, dL_dd23 = d_landscape_gaussians_ddists(d12, d13, d23)
    F_d12 = dlj_12_6_dr(d12) + dU3_dd12 + (LANDSCALE * dL_dd12)
    F_d13 = dlj_12_6_dr(d13) + dU3_dd13 + (LANDSCALE * dL_dd13)
    F_d23 = dlj_12_6_dr(d23) + dU3_dd23 + (LANDSCALE * dL_dd23)
    return F_d12, F_d13, F_d23


def forces_Umb_slice(d12, d13):
    d23 = d13
    F_d12, F_d13, F_d23 = forces_Umb(d12, d13, d23)
    Fx_slice = F_d13 + F_d23
    Fy_slice = F_d12
    return Fx_slice, Fy_slice

# =============================================================================
# Permutation-invariant feature map (same spirit as 3D sibling)
#   Physical slice order is (d13, d12), but features are built from the
#   virtual triplet (d13, d23=d13, d12).
# =============================================================================
def _features_from_P(P, symmetry_mode="sorted"):
    P = np.asarray(P, float)
    d13 = P[..., 0]
    d23 = P[..., 1]
    d12 = P[..., 2]
    D = np.stack([d12, d13, d23], axis=-1)

    mode = str(symmetry_mode).lower()
    if mode == "none":
        return np.stack([d13, d23, d12], axis=-1)

    if mode == "sorted":
        return np.sort(D, axis=-1)

    s1 = D[..., 0] + D[..., 1] + D[..., 2]
    s2 = D[..., 0] * D[..., 1] + D[..., 0] * D[..., 2] + D[..., 1] * D[..., 2]
    s3 = D[..., 0] * D[..., 1] * D[..., 2]
    return np.stack([s1, s2, s3], axis=-1)


def _slice_P_from_XY(XY):
    XY = np.asarray(XY, float)
    d13 = XY[..., 0]
    d12 = XY[..., 1]
    d23 = d13
    return np.stack([d13, d23, d12], axis=-1)


def _features_from_slice_XY(XY, symmetry_mode="sorted"):
    return _features_from_P(_slice_P_from_XY(XY), symmetry_mode=symmetry_mode)

# =============================================================================
# STYLE / FLAGS
# =============================================================================
LAB_X = r"$d_{13}=d_{23}\,(\sigma)$"
LAB_Y = r"$d_{12}\,(\sigma)$"
LAB_W3_TRUE = r"$W_3^{\mathrm{True}}\,(k_{\mathrm{B}}T)$"
LAB_W3_PRED = r"$W_3^{\mathrm{Pred}}\,(k_{\mathrm{B}}T)$"
LAB_DW3_TRUE = LAB_W3_TRUE
LAB_DW3_PRED = LAB_W3_PRED
LAB_C = LAB_W3_PRED
LAB_DW3 = LAB_DW3_PRED
LAB_ABSERR = r"$|W_3^{\mathrm{Pred}} - W_3^{\mathrm{True}}|\,(k_{\mathrm{B}}T)$"
ABSERR_CMAP = "magma"

USE_FIXED_CLIM = False
CLIM_MIN = -30.0
CLIM_MAX = 20.0

fig_scale = 2
FIGSIZE_3D = (6.0 * fig_scale, 4.5 * fig_scale)
FIGSIZE_RMSE = (4.5 * fig_scale, 4.2 * fig_scale)
FIGSIZE_SLICE2D = (4.8 * fig_scale, 3.9 * fig_scale)
FIGSIZE_PARITY = (4.5 * fig_scale, 4.2 * fig_scale)

AXIS_LABEL_FZ = 36
TICK_LABEL_FZ = 30
CBAR_FZ = AXIS_LABEL_FZ
CBAR_TICK_FZ = TICK_LABEL_FZ - 6
LEGEND_FZ = 30
SHOW_TITLES = False

# Same plotting controls as the latest toy/PIP 3D scripts.
AXIS_TICK_PAD_3D = 8
AXIS_TICK_PAD_3D_ALL = AXIS_TICK_PAD_3D * 0.5
AXIS_TICK_PAD_2D = 8
CBAR_TICK_PAD = 7
AXIS_TICK_LENGTH = 6.0
AXIS_TICK_WIDTH = 1.25
CBAR_TICK_LENGTH = 5.5
CBAR_TICK_WIDTH = 1.15

VIEW_ELEV = 28
VIEW_AZIM = -140
ORTHOGRAPHIC = True
Z_BOX_SCALE = 0.6

SWAP_XY = False
REVERSE_X = False
REVERSE_Y = False
REVERSE_Z = False
REVERSE_DIAG_X_2D = False
REVERSE_DIAG_Y_2D = False
REVERSE_DIAG_X_3D = False
REVERSE_DIAG_Y_3D = False

SHOW_CAND_OVERLAY = True
KNN_K = 100
CAND_SHOW_N = 1000
CAND_CMAP = "coolwarm"
CAND_MARKER_SIZE = 28.0
CAND_ALPHA = 0.95
CAND_EDGE_LW = 0.25
CAND_CBAR_PAD = 0.001
CAND_CBAR_YSHIFT = 0.015   # negative moves top colorbar closer to the 2D plot
CAND_CBAR_H = 0.03
CAND_CBAR_LABELPAD = 10     # keeps "Candidate score" away from the bar
CAND_CBAR_LABEL = r"Candidate score"
CAND_CBAR_NTICKS = 4          # fewer ticks so full-size labels fit
CAND_CBAR_TICK_FZ_TOP = CBAR_TICK_FZ
CAND_CBAR_LABEL_FZ_TOP = CBAR_FZ

SLICE_N12 = GRID_N
SLICE_N13 = GRID_N
SLICE_VIEW_ELEV = VIEW_ELEV
SLICE_VIEW_AZIM = VIEW_AZIM

KERNEL_SPEC = "m52+rq*se"
ARD = True
INIT_ITERS = 4000
AL_ITERS = 100
AL_BATCH = 5

VAL_FRAC = 0.15
VAL_MIN = 800
VAL_MAX = 4000

JITTER_DEF = 5e-4
LIK_NOISE0 = 5e-3
LIK_NOISE_RETRY = 5e-2
WHITE_NUGGET = 5e-4

PRUNE_NEAR_DUPLICATES = True
PRUNE_EPS = 1e-6

MIN_SEP_MULT = 0.7
ADAPT_MINSEP_DECAY = 0.92
POOL_MIN_FRACTION = 0.01

ACCEPT_TOL = 0.000

# Step-0 geometry counts
USE_EDGE_INIT_2D = True
# INIT_EDGE_COUNT = 1
# INIT_FACE_COUNT = 0

INIT_EDGE_COUNT       = int(os.environ.get("INIT_EDGE_COUNT", 1))
INIT_FACE_COUNT  = int(os.environ.get("INIT_FACE_COUNT", 0))

FIG_DIR = None

# Save the numerical data plotted in each figure without changing the analysis.
SAVE_FIGURE_DATA = True
FIG_DATA_SUBDIR = "figure_data"

MARKER_SCALE = 8.0
MS_TRAIN = 18 * MARKER_SCALE
MS_ANCHOR = 80 * MARKER_SCALE
LINE_MST_WIDTH = 3

TRAIN_FACE = "c"
TRAIN_EDGE = "black"
TRAIN_MARKER = "o"
MST_COLOR = "black"
MST_Q_COLOR = "tab:orange"

OVERLAY_ORDER_MODE = "mst_last"
Z_TRAIN, Z_KNN, Z_MST, Z_ANCHOR = 2, 3, 4, 5

FORCE_TEXT2D_ZLABEL = True
ZLABEL_2D_X = -0.11
ZLABEL_2D_Y = 0.5


USE_LATEX = False
if USE_LATEX:
    from matplotlib import rc, rcParams

    path_add = os.path.expanduser("~/.TinyTeX/bin/x86_64-linux")
    os.environ["PATH"] = path_add + ":" + os.environ.get("PATH", "")
    os.environ.pop("PERL5LIB", None)

    rc("text", usetex=True)
    rcParams["text.latex.preamble"] = (
        r"\usepackage{amsmath}\usepackage[T1]{fontenc}"
        r"\usepackage{helvet}"
        r"\usepackage{sansmath}\sansmath"
        r"\renewcommand{\familydefault}{\sfdefault}"
        r"\usepackage[strings]{underscore}"
    )
    rc("font", family="sans-serif")
else:
    mpl.rcParams["text.usetex"] = False

# With helvet+sansmath, plain \Delta can render as a black square.
# Use upright sans-serif Delta so it is nonitalic and does not become a box.
LAB_DW3_TRUE = r"$\mathsf{\Delta} W_3^{\mathrm{True}}\,(k_{\mathrm{B}}T)$"
LAB_DW3_PRED = r"$\mathsf{\Delta} W_3^{\mathrm{Pred}}\,(k_{\mathrm{B}}T)$"
LAB_DW3 = LAB_DW3_PRED
    
# =============================================================================
# Utilities
# =============================================================================
def _safe_output_name(name):
    safe_name = str(name)
    for a, b in (("$", ""), ("\\", ""), ("{", ""), ("}", ""), ("^", ""), (" ", "_"), ("/", "_"), (",", ""), ("(", ""), (")", "")):
        safe_name = safe_name.replace(a, b)
    return safe_name.replace("mathrm", "").replace("W_3", "W3")


def _figure_data_dir():
    global FIG_DIR
    if FIG_DIR is None:
        FIG_DIR = Path("./figs_UNSET")
        FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / FIG_DATA_SUBDIR
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_numeric_figure_data(name, values, header, suffix=""):
    if not SAVE_FIGURE_DATA:
        return
    path = _figure_data_dir() / f"{_safe_output_name(name)}{suffix}.csv"
    np.savetxt(
        path,
        np.asarray(values, float),
        delimiter=",",
        header=header,
        comments="",
        fmt="%.10e",
    )


def save_slice_figure_data(name, D12s, D13s, Z, mask):
    values = np.column_stack(
        [
            np.asarray(D13s, float).ravel(order="C"),
            np.asarray(D12s, float).ravel(order="C"),
            np.asarray(Z, float).ravel(order="C"),
            np.asarray(mask, bool).ravel(order="C").astype(int),
        ]
    )
    save_numeric_figure_data(name, values, "d13_sigma,d12_sigma,plotted_value,feasible")


def save_overlay_figure_data(name, info):
    if (not SAVE_FIGURE_DATA) or (not isinstance(info, dict)):
        return

    P = np.asarray(info["P"], float)
    train_all_idx = np.asarray(info.get("train_all_idx", []), dtype=int)
    train_q_idx = np.asarray(info.get("train_q_idx", []), dtype=int)
    knn_idx = np.asarray(info.get("knn_idx", []), dtype=int)
    anchor_idx = int(info["anchor_idx"])
    score_map = info.get("knn_score_map", None) or {}
    step0_edges = info.get("step0_overlay_edges", None) or []

    base = _safe_output_name(name)
    point_path = _figure_data_dir() / f"{base}_overlay_points.csv"
    with open(point_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["kind", "node_index", "d13_sigma", "d12_sigma", "candidate_score"])
        for kind, idxs in (("paid", train_all_idx), ("queried", train_q_idx), ("candidate", knn_idx)):
            for g in idxs:
                score = score_map.get(int(g), "") if kind == "candidate" else ""
                writer.writerow([kind, int(g), f"{P[g, 0]:.10e}", f"{P[g, 1]:.10e}", score])
        writer.writerow(["anchor", anchor_idx, f"{P[anchor_idx, 0]:.10e}", f"{P[anchor_idx, 1]:.10e}", ""])

    edge_rows = []
    for kind, edges in (
        ("mst_paid", mst_edges_over_nodes_2d(P, train_all_idx)),
        ("mst_queried", mst_edges_over_nodes_2d(P, train_q_idx)),
        ("step0_geometry", step0_edges),
    ):
        for g0, g1 in edges:
            edge_rows.append([kind, int(g0), int(g1), P[g0, 0], P[g0, 1], P[g1, 0], P[g1, 1]])

    edge_path = _figure_data_dir() / f"{base}_overlay_edges.csv"
    with open(edge_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["kind", "node_index_1", "node_index_2", "d13_1_sigma", "d12_1_sigma", "d13_2_sigma", "d12_2_sigma"]
        )
        writer.writerows(edge_rows)


def savefig(fig, name, extra_artists=None):
    global FIG_DIR
    if FIG_DIR is None:
        FIG_DIR = Path("./figs_UNSET")
        FIG_DIR.mkdir(exist_ok=True)
    safe_name = _safe_output_name(name)
    path = FIG_DIR / f"{safe_name}.pdf"
    prev_usetex = mpl.rcParams.get("text.usetex", False)
    try:
        fig.canvas.draw()
        fig.savefig(
            path,
            bbox_inches="tight",
            pad_inches=0.40,
            bbox_extra_artists=extra_artists,
            dpi=300,
        )
    except Exception as e:
        print(f"[WARN] LaTeX draw failed: {e}\n Falling back to mathtext for this figure only.")
        try:
            mpl.rcParams["text.usetex"] = False
            fig.canvas.draw()
        except Exception as e2:
            print(f"[WARN] mathtext draw also failed: {e2}")
        finally:
            fig.savefig(
                path,
                bbox_inches="tight",
                pad_inches=0.40,
                bbox_extra_artists=extra_artists,
                dpi=300,
            )
            mpl.rcParams["text.usetex"] = prev_usetex
    plt.show()


def standardize_fit(X):
    mu = X.mean(0)
    sd = X.std(0) + 1e-12
    return mu, sd


def standardize_apply(X, mu, sd):
    return (X - mu) / (sd + 1e-12)


def y_fit(y):
    m = float(np.mean(y))
    s = float(np.std(y) + 1e-12)
    return m, s


def y2z(y, m, s):
    return (np.asarray(y, float) - m) / s


def z2y(z, m, s):
    return np.asarray(z, float) * s + m


def unique_rows_idx(X):
    _, idx = np.unique(np.round(X, 14), axis=0, return_index=True)
    return np.sort(idx)


def rmse(y, yp):
    y = np.ravel(y)
    yp = np.ravel(yp)
    m = np.isfinite(y) & np.isfinite(yp)
    return float("nan") if (not np.any(m)) else float(np.sqrt(np.mean((y[m] - yp[m]) ** 2)))


def mst_total_length(P, idxs):
    idxs = np.array(sorted(set(map(int, idxs))), dtype=int)
    if idxs.size <= 1:
        return 0.0
    Q = P[idxs]
    D = np.linalg.norm(Q[:, None, :] - Q[None, :, :], axis=2)
    G = minimum_spanning_tree(coo_matrix(D))
    return float(G.sum())


def _sanitize_training_data(X, y):
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X = X[m]
    y = y[m]
    if X.shape[0] == 0:
        return X, y
    idx = unique_rows_idx(X)
    X = X[idx]
    y = y[idx]
    if (not PRUNE_NEAR_DUPLICATES) or X.shape[0] <= 2:
        return X, y
    tree = cKDTree(X)
    keep = np.ones(X.shape[0], bool)
    for i in range(X.shape[0]):
        if not keep[i]:
            continue
        nn = tree.query_ball_point(X[i], r=float(PRUNE_EPS))
        for j in nn:
            if j > i:
                keep[j] = False
    return X[keep], y[keep]


def _norm_from_clim():
    if USE_FIXED_CLIM:
        return mpl.colors.Normalize(vmin=float(CLIM_MIN), vmax=float(CLIM_MAX))
    return None


def _norm_from_masked(Z, mask, p_lo=5.0, p_hi=95.0):
    z = np.asarray(Z, float)
    m = np.asarray(mask, bool) & np.isfinite(z)
    if not np.any(m):
        return None
    v0 = float(np.nanpercentile(z[m], p_lo))
    v1 = float(np.nanpercentile(z[m], p_hi))
    if (not np.isfinite(v0)) or (not np.isfinite(v1)) or (v1 <= v0):
        v0 = float(np.nanmin(z[m]))
        v1 = float(np.nanmax(z[m]))
    if (not np.isfinite(v0)) or (not np.isfinite(v1)) or (v1 <= v0):
        return None
    return mpl.colors.Normalize(vmin=v0, vmax=v1)


def _abs_err(pred, truth):
    return np.abs(np.asarray(pred, float) - np.asarray(truth, float))

# =============================================================================
# KERNEL / GP
# =============================================================================
def kernel_from_spec(spec, ard, X=None, y=None):
    spec = spec.replace(" ", "").lower()

    def atom(tok):
        t = tok.lower()
        if t in ("m32", "matern32"):
            return gpflow.kernels.Matern32(lengthscales=[1, 1, 1] if ard else 1.0)
        if t in ("m52", "matern52"):
            return gpflow.kernels.Matern52(lengthscales=[1, 1, 1] if ard else 1.0)
        if t in ("se", "rbf"):
            return gpflow.kernels.SquaredExponential(lengthscales=[1, 1, 1] if ard else 1.0)
        if t in ("rq", "ratquad"):
            return gpflow.kernels.RationalQuadratic(
                lengthscales=[1, 1, 1] if ard else 1.0, alpha=1.0
            )
        if t in ("lin", "linear"):
            return gpflow.kernels.Linear()
        return gpflow.kernels.Matern52(lengthscales=[1, 1, 1] if ard else 1.0)

    if "+" in spec:
        terms = spec.split("+")
        Ks = []
        for term in terms:
            if "*" in term:
                factors = term.split("*")
                k = atom(factors[0])
                for f in factors[1:]:
                    k = k * atom(f)
                Ks.append(k)
            else:
                Ks.append(atom(term))
        K = Ks[0]
        for k2 in Ks[1:]:
            K = K + k2
    elif "*" in spec:
        toks = spec.split("*")
        K = atom(toks[0])
        for t2 in toks[1:]:
            K = K * atom(t2)
    else:
        K = atom(spec)

    if X is not None and y is not None and len(y) > 2:
        var0 = max(float(np.var(y)), 1e-6)

        def init_one(k):
            if hasattr(k, "variance"):
                try:
                    k.variance.assign(var0)
                except Exception:
                    pass
            if hasattr(k, "lengthscales"):
                ls0 = np.clip(np.std(X, axis=0), 0.15, 3.0)
                try:
                    k.lengthscales.assign(ls0 if ard else float(np.mean(ls0)))
                except Exception:
                    pass
            if hasattr(k, "alpha"):
                try:
                    k.alpha.assign(1.0)
                except Exception:
                    pass

        stack = [K]
        while stack:
            kk = stack.pop()
            if hasattr(kk, "kernels"):
                stack.extend(list(kk.kernels))
            else:
                init_one(kk)
    return K


def structured_kernel_1d3d(ard=True):
    k0 = gpflow.kernels.Matern52(active_dims=[0], lengthscales=1.0)
    k1 = gpflow.kernels.Matern52(active_dims=[1], lengthscales=1.0)
    k2 = gpflow.kernels.Matern52(active_dims=[2], lengthscales=1.0)
    if ard:
        k3s = gpflow.kernels.Matern52(active_dims=[0, 1, 2], lengthscales=[1.0, 1.0, 1.0])
        k3l = gpflow.kernels.RationalQuadratic(
            active_dims=[0, 1, 2],
            lengthscales=[1.0, 1.0, 1.0],
            variance=0.5,
            alpha=1.0,
        )
    else:
        k3s = gpflow.kernels.Matern52(active_dims=[0, 1, 2], lengthscales=1.0)
        k3l = gpflow.kernels.RationalQuadratic(
            active_dims=[0, 1, 2],
            lengthscales=1.0,
            variance=0.5,
            alpha=1.0,
        )
    return (k0 + k1 + k2) + (k3s + k3l)


def psd_rescue(model):
    comps = model.kernel.kernels if hasattr(model.kernel, "kernels") else [model.kernel]
    for k in comps:
        if hasattr(k, "lengthscales"):
            ls = np.asarray(k.lengthscales.numpy(), float)
            ls = np.clip(ls, 0.25, 8.0)
            k.lengthscales.assign(ls if ls.ndim else float(ls))
        if hasattr(k, "variance"):
            k.variance.assign(max(float(k.variance.numpy()), 1e-6))
        if hasattr(k, "alpha"):
            k.alpha.assign(np.clip(float(k.alpha.numpy()), 0.2, 5.0))
    model.likelihood.variance.assign(max(float(model.likelihood.variance.numpy()), 1e-2))


def _try_minimize(m, iters):
    gpflow.optimizers.Scipy().minimize(
        m.training_loss,
        m.trainable_variables,
        options={"maxiter": int(iters), "disp": False},
    )


def warm_start_params(m_new, m_prev):
    if m_prev is None:
        return
    try:
        pd_prev = gpf_util.parameter_dict(m_prev)
        pd_new = gpf_util.parameter_dict(m_new)
    except Exception:
        return
    for name, v_new in pd_new.items():
        v_old = pd_prev.get(name, None)
        if v_old is None:
            continue
        try:
            if v_new.shape == v_old.shape:
                v_new.assign(v_old)
        except Exception:
            pass


def build_gpr(X, z, kernel_mode, spec, ard, iters, m_prev=None):
    X = np.asarray(X, float)
    z = np.asarray(z, float).reshape(-1)
    if X.shape[0] < 2:
        raise RuntimeError("Not enough training points.")

    retry_jitters = [JITTER_DEF, 5e-4, 2e-3, 1e-2]
    retry_noises = [LIK_NOISE0, LIK_NOISE_RETRY, 5e-2, 1e-1]
    last_err = None

    for jj, nv in zip(retry_jitters, retry_noises):
        try:
            gpflow.config.set_default_jitter(float(jj))
            Xtf = tf.convert_to_tensor(X, tf.float64)
            ztf = tf.convert_to_tensor(z.reshape(-1, 1), tf.float64)

            if str(kernel_mode).lower() == "structured":
                base = structured_kernel_1d3d(ard=ard)
            else:
                base = kernel_from_spec(spec, ard, X, z)

            white = gpflow.kernels.White(variance=WHITE_NUGGET)
            gpflow.set_trainable(white.variance, False)
            K = base + white

            m = gpflow.models.GPR(
                (Xtf, ztf),
                kernel=K,
                mean_function=gpflow.mean_functions.Zero(),
            )
            warm_start_params(m, m_prev)
            m.likelihood.variance.assign(float(nv))
            psd_rescue(m)

            try:
                _try_minimize(m, iters)
            except Exception:
                psd_rescue(m)
                _try_minimize(m, max(200, iters // 2))

            psd_rescue(m)
            mu, var = m.predict_f(Xtf)
            mu = mu.numpy()
            var = var.numpy()
            if (not np.all(np.isfinite(mu))) or (not np.all(np.isfinite(var))) or np.any(var <= 0):
                raise FloatingPointError("predict_f produced non-finite/invalid variance.")
            return m
        except Exception as e:
            last_err = e

    raise RuntimeError(f"GP build failed after retries. Last error: {last_err}")


def predict_mu_std_z(m, Xnew):
    Xtf = tf.convert_to_tensor(np.asarray(Xnew, float), tf.float64)
    try:
        mu, var = m.predict_f(Xtf)
    except Exception:
        psd_rescue(m)
        mu, var = m.predict_f(Xtf)
    mu = mu.numpy().ravel()
    var = np.nan_to_num(var.numpy().ravel(), nan=1e-6, posinf=1e6, neginf=1e-12)
    std = np.sqrt(np.maximum(var, 1e-12))
    return mu, std

# =============================================================================
# Plotting helpers
# =============================================================================
def _compact_number_label(x, pos=None):
    """Compact tick labels: -10 instead of -10.0, but keep decimals if needed."""
    try:
        x = float(x)
    except Exception:
        return str(x)
    if abs(x) < 1e-12:
        x = 0.0
    if np.isclose(x, np.round(x), atol=1e-9):
        s = f"{int(np.round(x))}"
    else:
        s = f"{x:g}"
    if mpl.rcParams.get("axes.unicode_minus", True):
        s = s.replace("-", "−")
    return s

COMPACT_TICK_FORMATTER = FuncFormatter(_compact_number_label)


def _distance_ticks(lim):
    lo = int(np.ceil(float(lim[0])))
    hi = int(np.floor(float(lim[1])))
    return np.arange(lo, hi + 1, 1.0) if hi >= lo else None


def _apply_distance_ticks_2d(ax, xlim, ylim):
    xt = _distance_ticks(xlim)
    yt = _distance_ticks(ylim)
    if xt is not None: ax.set_xticks(xt)
    if yt is not None: ax.set_yticks(yt)
    ax.xaxis.set_major_formatter(COMPACT_TICK_FORMATTER)
    ax.yaxis.set_major_formatter(COMPACT_TICK_FORMATTER)


def _apply_distance_ticks_3d(ax, xlim, ylim, zlim=None):
    xt = _distance_ticks(xlim)
    yt = _distance_ticks(ylim)
    if xt is not None: ax.set_xticks(xt)
    if yt is not None: ax.set_yticks(yt)
    if zlim is not None:
        zt = _distance_ticks(zlim)
        if zt is not None: ax.set_zticks(zt)
    ax.xaxis.set_major_formatter(COMPACT_TICK_FORMATTER)
    ax.yaxis.set_major_formatter(COMPACT_TICK_FORMATTER)
    ax.zaxis.set_major_formatter(COMPACT_TICK_FORMATTER)


def _format_energy_zaxis(ax):
    try:
        ax.zaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3, integer=True))
    except TypeError:
        ax.zaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.zaxis.set_major_formatter(COMPACT_TICK_FORMATTER)


def _format_colorbar_ticks(cb):
    cb.ax.yaxis.set_major_formatter(COMPACT_TICK_FORMATTER)
    cb.ax.xaxis.set_major_formatter(COMPACT_TICK_FORMATTER)


def _make_3d_ticks_visible(ax):
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis._axinfo["tick"]["inward_factor"] = 0.0
            axis._axinfo["tick"]["outward_factor"] = 0.28
            axis._axinfo["tick"]["linewidth"] = {
                True: AXIS_TICK_WIDTH,
                False: AXIS_TICK_WIDTH,
            }
        except Exception:
            pass


def _candidate_score_tick_label(x, pos=None, decimals=2):
    """Compact but safe labels for the top candidate-score colorbar."""
    try:
        x = float(x)
    except Exception:
        return ""

    if not np.isfinite(x):
        return ""

    if abs(x) < 1e-12:
        x = 0.0

    s = f"{x:.{int(decimals)}f}"
    if int(decimals) <= 2:
        s = s.rstrip("0").rstrip(".")

    if mpl.rcParams.get("axes.unicode_minus", True):
        s = s.replace("-", "−")

    return s


def _add_top_colorbar(fig, ax, mappable, label):
    pos = ax.get_position()
    cax = fig.add_axes([
        pos.x0,
        pos.y1 + CAND_CBAR_PAD + CAND_CBAR_YSHIFT,
        pos.width,
        CAND_CBAR_H,
    ])

    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.xaxis.set_label_position("top")

    # Force multiple fixed ticks. This avoids auto-pruned single labels
    # and confusing labels such as 3.25 visually appearing like 325.
    vmin = getattr(mappable.norm, "vmin", None)
    vmax = getattr(mappable.norm, "vmax", None)
    if vmin is not None and vmax is not None:
        vmin = float(vmin)
        vmax = float(vmax)
        if np.isfinite(vmin) and np.isfinite(vmax):
            if np.isclose(vmin, vmax):
                eps = 0.01 * max(abs(vmin), 1.0)
                vmin -= eps
                vmax += eps

            ticks = np.linspace(vmin, vmax, int(CAND_CBAR_NTICKS))
            rng = abs(vmax - vmin)
            decimals = 4 if rng < 0.01 else (3 if rng < 0.1 else (2 if rng < 1.0 else 1))
            cb.set_ticks(ticks)
            cb.ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(ticks))
            cb.ax.xaxis.set_major_formatter(
                FuncFormatter(lambda x, pos: _candidate_score_tick_label(x, pos, decimals))
            )

    cb.set_label(
        label,
        fontsize=CAND_CBAR_LABEL_FZ_TOP,
        labelpad=CAND_CBAR_LABELPAD,
    )
    cb.ax.tick_params(
        axis="x",
        which="major",
        labelsize=CAND_CBAR_TICK_FZ_TOP,
        pad=CBAR_TICK_PAD,
        length=CBAR_TICK_LENGTH,
        width=CBAR_TICK_WIDTH,
    )

    cb.ax.xaxis.get_offset_text().set_visible(False)
    for lab in cb.ax.get_xticklabels():
        lab.set_clip_on(False)

    return cb


def _pin_3d_zlabel_2d(ax, label, fontsize, x=ZLABEL_2D_X, y=ZLABEL_2D_Y):
    try:
        ax.set_zlabel("")
    except Exception:
        pass
    try:
        art = ax.text2D(
            float(x),
            float(y),
            label,
            transform=ax.transAxes,
            fontsize=float(fontsize),
            rotation=90,
            va="center",
            ha="right",
            clip_on=False,
        )
        return art
    except Exception:
        return None


def plot_slice_2d_heat(
    D12s,
    D13s,
    Z,
    m,
    name,
    title,
    clabel,
    cmap="viridis",
    norm=None,
    overlay=None,
):
    Zm = np.where(m, Z, np.nan)
    x = D13s[0, :]
    y = D12s[:, 0]

    fig, ax = plt.subplots(figsize=FIGSIZE_SLICE2D)
    pc = ax.pcolormesh(x, y, Zm, shading="auto", cmap=cmap, norm=norm)

    ax.set_xlabel(LAB_X, fontsize=AXIS_LABEL_FZ, labelpad=10)
    ax.set_ylabel(LAB_Y, fontsize=AXIS_LABEL_FZ, labelpad=10)

    _apply_distance_ticks_2d(ax, (x.min(), x.max()), (y.min(), y.max()))

    ax.set_aspect("equal")
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FZ - 6, pad=AXIS_TICK_PAD_2D,
                   length=AXIS_TICK_LENGTH, width=AXIS_TICK_WIDTH)

    if REVERSE_DIAG_X_2D:
        ax.invert_xaxis()
    if REVERSE_DIAG_Y_2D:
        ax.invert_yaxis()

    if SHOW_TITLES:
        ax.set_title(title, fontsize=LEGEND_FZ + 2, pad=10)

    extra = overlay(ax) if callable(overlay) else None

    cb = fig.colorbar(pc, ax=ax, pad=0.02)
    cb.set_label(clabel, fontsize=AXIS_LABEL_FZ, labelpad=12)
    cb.ax.tick_params(labelsize=CBAR_TICK_FZ, pad=CBAR_TICK_PAD,
                      length=CBAR_TICK_LENGTH, width=CBAR_TICK_WIDTH)
    _format_colorbar_ticks(cb)

    if isinstance(extra, dict) and extra.get("knn_mappable", None) is not None:
        _add_top_colorbar(fig, ax, extra["knn_mappable"], extra.get("knn_label", CAND_CBAR_LABEL))

    save_slice_figure_data(name, D12s, D13s, Z, m)
    if callable(overlay) and hasattr(overlay, "figure_data"):
        save_overlay_figure_data(name, overlay.figure_data)
    savefig(fig, name)


def plot_slice_3d_surface(
    D12s,
    D13s,
    Z,
    m,
    name,
    title,
    clabel,
    cmap="viridis",
    norm=None,
):
    Zm = np.where(m, Z, np.nan)
    fig = plt.figure(figsize=FIGSIZE_3D)
    ax = fig.add_subplot(111, projection="3d")

    ax.tick_params(axis="z", pad=AXIS_TICK_PAD_3D_ALL)
    for lab in ax.zaxis.get_ticklabels():
        lab.set_horizontalalignment("right")
        lab.set_verticalalignment("center")
        lab.set_rotation(0)

    try:
        ax.computed_zorder = False
    except Exception:
        pass

    if ORTHOGRAPHIC:
        try:
            ax.set_proj_type("ortho")
        except Exception:
            pass

    surf = ax.plot_surface(D13s, D12s, Zm, cmap=cmap, norm=norm, linewidth=0, antialiased=True)

    ax.set_xlabel(LAB_X, fontsize=AXIS_LABEL_FZ, labelpad=26)
    ax.set_ylabel(LAB_Y, fontsize=AXIS_LABEL_FZ, labelpad=26)

    z_art = None
    if FORCE_TEXT2D_ZLABEL:
        z_art = _pin_3d_zlabel_2d(ax, clabel, AXIS_LABEL_FZ, x=ZLABEL_2D_X, y=ZLABEL_2D_Y)
    else:
        ax.set_zlabel(clabel, fontsize=AXIS_LABEL_FZ, labelpad=16)

    if SHOW_TITLES:
        ax.set_title(title, fontsize=LEGEND_FZ + 2, pad=10)

    ax.set_xlim(D13_MIN, D13_MAX)
    ax.set_ylim(D12_MIN, D12_MAX)
    _apply_distance_ticks_3d(ax, (D13_MIN, D13_MAX), (D12_MIN, D12_MAX), None)

    if REVERSE_DIAG_X_3D:
        ax.invert_xaxis()
    if REVERSE_DIAG_Y_3D:
        ax.invert_yaxis()
    if REVERSE_Z:
        ax.invert_zaxis()

    try:
        ax.set_box_aspect(
            (
                (D13_MAX - D13_MIN),
                (D12_MAX - D12_MIN),
                Z_BOX_SCALE * (D12_MAX - D12_MIN),
            )
        )
    except Exception:
        pass

    ax.view_init(elev=float(SLICE_VIEW_ELEV), azim=float(SLICE_VIEW_AZIM))
    ax.tick_params(axis="x", labelsize=TICK_LABEL_FZ - 6, pad=AXIS_TICK_PAD_3D_ALL,
                   length=AXIS_TICK_LENGTH, width=AXIS_TICK_WIDTH)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FZ - 6, pad=AXIS_TICK_PAD_3D_ALL,
                   length=AXIS_TICK_LENGTH, width=AXIS_TICK_WIDTH)
    ax.tick_params(axis="z", labelsize=TICK_LABEL_FZ - 6, pad=AXIS_TICK_PAD_3D_ALL,
                   length=AXIS_TICK_LENGTH, width=AXIS_TICK_WIDTH)
    _format_energy_zaxis(ax)
    _make_3d_ticks_visible(ax)

    try:
        xt = np.arange(np.floor(D13_MIN), np.ceil(D13_MAX) + 1e-9, 1.0)
        yt = np.arange(np.floor(D12_MIN), np.ceil(D12_MAX) + 1e-9, 1.0)
        ax.set_xticks(xt)
        ax.set_yticks(yt)

        # Hide both corner 6 labels and draw one shared centered 6.
        ax.set_xticklabels([""] + [f"{int(v)}" for v in xt[1:]])
        ax.set_yticklabels([""] + [f"{int(v)}" for v in yt[1:]])

        corner6_art = ax.text2D(
            0.558, 0.06, "6",
            transform=ax.transAxes,
            fontsize=TICK_LABEL_FZ - 6,
            ha="center",
            va="center",
            clip_on=False,
        )

        corner6_art = ax.text2D(
            0.558, 0.06, "6",
            transform=ax.transAxes,
            fontsize=TICK_LABEL_FZ - 6,
            ha="center",
            va="center",
            clip_on=False,
        )
    except Exception:
        pass

    cb = fig.colorbar(surf, ax=ax, shrink=0.78, pad=0.05)
    cb.set_label(clabel, fontsize=AXIS_LABEL_FZ, labelpad=12)
    cb.ax.tick_params(labelsize=TICK_LABEL_FZ - 6, pad=CBAR_TICK_PAD,
                      length=CBAR_TICK_LENGTH, width=CBAR_TICK_WIDTH)
    _format_colorbar_ticks(cb)

    extra_artists = [a for a in (z_art, locals().get("corner6_art", None)) if a is not None]
    save_slice_figure_data(name, D12s, D13s, Z, m)
    savefig(fig, name, extra_artists=extra_artists)


def gp_predict_on_slice(model, xmu, xsd, ymu, ysd, D12s, D13s, symmetry_mode):
    XY = np.c_[D13s.ravel(), D12s.ravel()]
    Pf = _features_from_slice_XY(XY, symmetry_mode=symmetry_mode)
    Xs = standardize_apply(Pf, xmu, xsd)
    mu_z, std_z = predict_mu_std_z(model, Xs)
    mu_y = z2y(mu_z, ymu, ysd)
    std_y = std_z * ysd
    return mu_y.reshape(D12s.shape), std_y.reshape(D12s.shape)

# =============================================================================
# 2D grid indexing / neighbors
# =============================================================================
def _idx_to_ij(idx, n2):
    return divmod(int(idx), int(n2))


def _ij_to_idx(i, j, n2):
    return int(i) * int(n2) + int(j)


def neighbors_8(g, n1, n2):
    i, j = divmod(int(g), int(n2))
    out = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            ii, jj = i + di, j + dj
            if 0 <= ii < n1 and 0 <= jj < n2:
                out.append(_ij_to_idx(ii, jj, n2))
    return out


def build_grid_edges(n1, n2, mode="8"):
    E = []
    idx = lambda i, j: i * n2 + j
    for i in range(n1):
        for j in range(n2):
            if i + 1 < n1:
                E.append((idx(i, j), idx(i + 1, j)))
            if j + 1 < n2:
                E.append((idx(i, j), idx(i, j + 1)))
            if mode == "8":
                if (i + 1 < n1) and (j + 1 < n2):
                    E.append((idx(i, j), idx(i + 1, j + 1)))
                if (i + 1 < n1) and (j - 1 >= 0):
                    E.append((idx(i, j), idx(i + 1, j - 1)))
    return np.asarray(E, int)

# =============================================================================
# Strict connectivity (path paying)
# =============================================================================
def manhattan_walk_add_2d(train_set, start_g, target_g, n2, allowed_mask):
    start_g = int(start_g)
    target_g = int(target_g)

    if (not allowed_mask[start_g]) or (not allowed_mask[target_g]):
        return []

    i0, j0 = _idx_to_ij(start_g, n2)
    i1, j1 = _idx_to_ij(target_g, n2)

    added = []
    train_set.add(start_g)
    added.append(start_g)

    if i0 != i1:
        step = 1 if i1 > i0 else -1
        while i0 != i1:
            i0 += step
            g = _ij_to_idx(i0, j0, n2)
            if allowed_mask[g]:
                train_set.add(g)
                added.append(g)

    if j0 != j1:
        step = 1 if j1 > j0 else -1
        while j0 != j1:
            j0 += step
            g = _ij_to_idx(i0, j0, n2)
            if allowed_mask[g]:
                train_set.add(g)
                added.append(g)

    return added


def shortest_path_add_2d(train_set, start_g, target_g, P, n1, n2, allowed_mask):
    start_g = int(start_g)
    target_g = int(target_g)

    if start_g == target_g:
        if allowed_mask[start_g]:
            train_set.add(start_g)
            return [start_g]
        return []

    if (not allowed_mask[start_g]) or (not allowed_mask[target_g]):
        return []

    train_set.add(start_g)
    added = [start_g]

    dist = {start_g: 0.0}
    prev = {start_g: None}
    pq = [(0.0, start_g)]

    while pq:
        dcur, g = heapq.heappop(pq)
        if g == target_g:
            break
        if dcur != dist.get(g, np.inf):
            continue

        for h in neighbors_8(g, n1, n2):
            h = int(h)
            if not allowed_mask[h]:
                continue
            step = float(np.linalg.norm(P[h] - P[g]))
            nd = dcur + step
            if nd < dist.get(h, np.inf):
                dist[h] = nd
                prev[h] = g
                heapq.heappush(pq, (nd, h))

    if target_g not in prev:
        raise RuntimeError("No allowed connected path between requested nodes")

    path = []
    g = target_g
    while g is not None:
        path.append(int(g))
        g = prev[g]
    path.reverse()

    for g in path[1:]:
        if allowed_mask[g]:
            train_set.add(g)
            added.append(g)

    return added

# =============================================================================
# LSQR helpers
# =============================================================================
def lsqr_U_on_known_nodes_2d(
    known_nodes,
    anchor_g,
    P,
    Fx,
    Fy,
    n1,
    n2,
    mode="8",
    atol=1e-10,
    btol=1e-10,
    iter_lim=8000,
    dw_clip=None,
):
    known_nodes = np.array(sorted(set(map(int, known_nodes))), dtype=int)
    nk = known_nodes.size
    if nk == 0:
        return np.array([], float), known_nodes

    loc = {int(g): i for i, g in enumerate(known_nodes)}
    if int(anchor_g) not in loc:
        return np.full(nk, np.nan, float), known_nodes

    edges = build_grid_edges(n1, n2, mode=mode)

    rows, cols, data, b = [], [], [], []
    n_skip_nonfinite = 0
    n_skip_clip = 0

    use_continuous = str(LSQR_INTEGRATION_MODE).lower() in (
        "continuous", "quad", "quadrature", "gauss", "gausslegendre"
    )
    q_order = int(LSQR_EDGE_QUAD_ORDER)

    Fx = np.asarray(Fx, float).ravel(order="C")
    Fy = np.asarray(Fy, float).ravel(order="C")

    for ii, jj in edges:
        ii = int(ii)
        jj = int(jj)
        if (ii not in loc) or (jj not in loc):
            continue

        dx = P[jj] - P[ii]

        if use_continuous:
            # Continuous edge work: evaluate the analytic slice gradient at
            # Gauss-Legendre points between the two paid nodes.
            dw = _edge_work_quadrature_2d(P[ii], P[jj], quad_order=q_order)
        else:
            # Old endpoint/trapezoid behavior, kept for reproducibility.
            Fx_i, Fy_i = Fx[ii], Fy[ii]
            Fx_j, Fy_j = Fx[jj], Fy[jj]
            if not (np.isfinite(Fx_i) and np.isfinite(Fy_i) and np.isfinite(Fx_j) and np.isfinite(Fy_j)):
                n_skip_nonfinite += 1
                continue

            Fx_mid = 0.5 * (Fx_i + Fx_j)
            Fy_mid = 0.5 * (Fy_i + Fy_j)
            dw = +(Fx_mid * dx[0] + Fy_mid * dx[1])

        if not np.isfinite(dw):
            n_skip_nonfinite += 1
            continue
        if (dw_clip is not None) and (abs(dw) > float(dw_clip)):
            n_skip_clip += 1
            continue

        r = len(b)
        b.append(dw)
        rows.extend([r, r])
        cols.extend([loc[ii], loc[jj]])
        data.extend([-1.0, +1.0])

        r2 = len(b)
        w = np.sqrt(1e-3)
        b.append(0.0)
        rows.extend([r2, r2])
        cols.extend([loc[ii], loc[jj]])
        data.extend([-w, +w])

    M = len(b)
    if LSQR_PRINT_EQUATION_STATS:
        print(
            f"[LSQR build 2D] mode={mode} nk={nk} M={M} "
            f"integration={'continuous' if use_continuous else 'endpoint'} "
            f"quad={q_order if use_continuous else '-'} "
            f"skip_nonfinite={n_skip_nonfinite} skip_clip={n_skip_clip}"
        )

    if M == 0:
        return np.full(nk, np.nan, float), known_nodes

    Asub = coo_matrix(
        (np.asarray(data, float), (np.asarray(rows, int), np.asarray(cols, int))),
        shape=(M, nk),
    ).tocsr()
    b = np.asarray(b, float)

    a_loc = loc[int(anchor_g)]
    keep = np.arange(nk, dtype=int) != a_loc

    U = np.zeros(nk, float)
    if np.any(keep):
        sol = lsqr(Asub[:, keep], b, atol=atol, btol=btol, iter_lim=iter_lim)[0]
        U[keep] = sol
    U[a_loc] = 0.0
    U -= U[a_loc]
    return U, known_nodes


def _edge_work_quadrature_2d(p0, p1, quad_order=PATHINT_QUAD_ORDER):
    """Integrate dU = grad(U) . dx from p0 to p1 on the 2D slice.

    Coordinates are P=(d13, d12).  forces_Umb_slice(d12,d13) returns
    (dU/dd13, dU/dd12), i.e. the gradient components in this P-space.
    """
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    dx = p1 - p0
    if not np.all(np.isfinite(dx)):
        return np.nan
    if float(np.linalg.norm(dx)) == 0.0:
        return 0.0

    qx, qw = np.polynomial.legendre.leggauss(int(quad_order))
    t = 0.5 * (qx + 1.0)
    w = 0.5 * qw
    pts = p0[None, :] + t[:, None] * dx[None, :]

    # pts[:,0]=d13, pts[:,1]=d12
    gx, gy = forces_Umb_slice(pts[:, 1], pts[:, 0])
    integrand = np.asarray(gx, float) * dx[0] + np.asarray(gy, float) * dx[1]
    if not np.all(np.isfinite(integrand)):
        return np.nan
    return float(np.sum(w * integrand))


def path_integrated_U_on_known_nodes_2d(
    known_nodes,
    anchor_g,
    P,
    n1,
    n2,
    mode="8",
    quad_order=PATHINT_QUAD_ORDER,
):
    """Assign nodal energies by on-the-fly path integration of exact forces.

    This is different from LSQR.  It does not read oracle energies and does not
    solve a global least-squares problem.  It uses the current paid graph, computes
    local line integrals of the exact force/gradient along neighboring paid-node
    edges, and propagates energies from the anchor along shortest graph paths.
    """
    known_nodes = np.array(sorted(set(map(int, known_nodes))), dtype=int)
    nk = known_nodes.size
    if nk == 0:
        return np.array([], float), known_nodes

    loc = {int(g): i for i, g in enumerate(known_nodes)}
    if int(anchor_g) not in loc:
        return np.full(nk, np.nan, float), known_nodes

    edges = build_grid_edges(n1, n2, mode=mode)
    adj = [[] for _ in range(nk)]
    n_edges = 0
    n_bad = 0

    for ii, jj in edges:
        ii = int(ii); jj = int(jj)
        if (ii not in loc) or (jj not in loc):
            continue
        li = loc[ii]; lj = loc[jj]
        length = float(np.linalg.norm(P[jj] - P[ii]))
        dU_ij = _edge_work_quadrature_2d(P[ii], P[jj], quad_order=quad_order)
        if not np.isfinite(dU_ij):
            n_bad += 1
            continue
        adj[li].append((lj, length, dU_ij))
        adj[lj].append((li, length, -dU_ij))
        n_edges += 1

    if LSQR_PRINT_EQUATION_STATS:
        print(f"[PATHINT build 2D] mode={mode} nk={nk} edges={n_edges} bad_edges={n_bad} quad={quad_order}")

    a_loc = loc[int(anchor_g)]
    dist = np.full(nk, np.inf, float)
    U = np.full(nk, np.nan, float)
    dist[a_loc] = 0.0
    U[a_loc] = 0.0
    pq = [(0.0, a_loc)]

    while pq:
        dcur, u = heapq.heappop(pq)
        if dcur != dist[u]:
            continue
        for v, length, dU_uv in adj[u]:
            nd = dcur + length
            if nd < dist[v]:
                dist[v] = nd
                U[v] = U[u] + dU_uv
                heapq.heappush(pq, (nd, v))

    U -= U[a_loc]
    return U, known_nodes

# =============================================================================
# Step-0 geometry builders
# =============================================================================
def _line_nodes_edges_between_2d(p, q, n1, n2, feas_mask_flat):
    i0, j0 = map(int, p)
    i1, j1 = map(int, q)
    di = 0 if i1 == i0 else (1 if i1 > i0 else -1)
    dj = 0 if j1 == j0 else (1 if j1 > j0 else -1)
    steps = max(abs(i1 - i0), abs(j1 - j0))

    nodes = []
    for t in range(steps + 1):
        ii = i0 + di * t
        jj = j0 + dj * t
        if 0 <= ii < n1 and 0 <= jj < n2:
            g = _ij_to_idx(ii, jj, n2)
            if feas_mask_flat[g]:
                nodes.append(int(g))

    seen = set()
    out_nodes = []
    for g in nodes:
        if g not in seen:
            seen.add(g)
            out_nodes.append(g)

    out_edges = []
    for a, b in zip(out_nodes[:-1], out_nodes[1:]):
        i, j = (a, b) if a < b else (b, a)
        if i != j:
            out_edges.append((int(i), int(j)))

    return np.array(out_nodes, dtype=int), out_edges


def _all_square_edges_ij(n1, n2):
    lo_i, hi_i = 0, n1 - 1
    lo_j, hi_j = 0, n2 - 1
    return [
        ((hi_i, lo_j), (hi_i, hi_j)),
        ((lo_i, hi_j), (hi_i, hi_j)),
        ((lo_i, lo_j), (lo_i, hi_j)),
        ((lo_i, lo_j), (hi_i, lo_j)),
    ]


def _all_square_diags_ij(n1, n2):
    lo_i, hi_i = 0, n1 - 1
    lo_j, hi_j = 0, n2 - 1
    return [
        ((lo_i, lo_j), (hi_i, hi_j)),
        ((lo_i, hi_j), (hi_i, lo_j)),
    ]


def _rank_lines_by_anchor_endpoint_2d(lines, anchor_ij):
    a = tuple(map(int, anchor_ij))

    def key_fn(pair):
        p, q = pair
        p = tuple(p)
        q = tuple(q)
        hit = (p == a) or (q == a)
        lo, hi = (p, q) if p < q else (q, p)
        return (0 if hit else 1, lo, hi)

    return sorted(lines, key=key_fn)


def build_step0_geometry_2d(anchor_g, n1, n2, feas_mask_flat, edge_count, diag_count):
    edge_count = int(np.clip(edge_count, 0, 4))
    diag_count = int(np.clip(diag_count, 0, 2))
    ai, aj = _idx_to_ij(anchor_g, n2)
    anchor_ij = (ai, aj)

    nodes_all = set([int(anchor_g)])
    extra_edges = set()

    all_edges = _rank_lines_by_anchor_endpoint_2d(_all_square_edges_ij(n1, n2), anchor_ij)
    for p, q in all_edges[:edge_count]:
        n, e = _line_nodes_edges_between_2d(p, q, n1, n2, feas_mask_flat)
        nodes_all.update(map(int, n.tolist()))
        for i, j in e:
            extra_edges.add((i, j) if i < j else (j, i))

    all_diags = _rank_lines_by_anchor_endpoint_2d(_all_square_diags_ij(n1, n2), anchor_ij)
    for p, q in all_diags[:diag_count]:
        n, e = _line_nodes_edges_between_2d(p, q, n1, n2, feas_mask_flat)
        nodes_all.update(map(int, n.tolist()))
        for i, j in e:
            extra_edges.add((i, j) if i < j else (j, i))

    return np.array(sorted(nodes_all), dtype=int), list(extra_edges)


def _components_in_step0_2d(nodes, n1, n2, feas_flat):
    nodes = np.array(sorted(set(map(int, nodes))), dtype=int)
    node_set = set(nodes.tolist())
    seen = set()
    comps = []
    for g in nodes:
        g = int(g)
        if g in seen:
            continue
        if not feas_flat[g]:
            continue
        stack = [g]
        seen.add(g)
        comp = []
        while stack:
            u = int(stack.pop())
            comp.append(u)
            for v in neighbors_8(u, n1, n2):
                v = int(v)
                if v in seen:
                    continue
                if (v in node_set) and feas_flat[v]:
                    seen.add(v)
                    stack.append(v)
        comps.append(comp)
    return comps


def enforce_step0_connected_2d(step0_nodes, anchor_g, P, n1, n2, feas_flat):
    anchor_g = int(anchor_g)
    step0_nodes = np.array(sorted(set(map(int, step0_nodes))), dtype=int)
    if step0_nodes.size == 0:
        return step0_nodes

    nodes_set = set(step0_nodes.tolist())
    nodes_set.add(anchor_g)

    comps = _components_in_step0_2d(nodes_set, n1, n2, feas_flat)
    comp_anchor = None
    for comp in comps:
        if anchor_g in comp:
            comp_anchor = comp
            break
    if comp_anchor is None:
        comp_anchor = [anchor_g]
        comps = [comp_anchor] + comps

    if len(comps) <= 1:
        print("[STEP0 connect 2D] components=1 (already connected)")
        return np.array(sorted(nodes_set), dtype=int)

    anchor_pt = P[anchor_g]
    train_set = set(nodes_set)

    sizes = [len(c) for c in comps]
    print(
        f"[STEP0 connect 2D] components={len(comps)} "
        f"sizes={sorted(sizes, reverse=True)[:8]}{'...' if len(comps)>8 else ''}"
    )

    for comp in comps:
        if anchor_g in comp:
            continue
        comp = np.array(comp, dtype=int)
        pts = P[comp]
        j = int(np.argmin(np.linalg.norm(pts - anchor_pt[None, :], axis=1)))
        g_rep = int(comp[j])
        added = shortest_path_add_2d(train_set, g_rep, anchor_g, P, n1, n2, feas_flat)
        if len(added) == 0:
            _ = manhattan_walk_add_2d(train_set, g_rep, anchor_g, n2, feas_flat)

    out = np.array(sorted(train_set), dtype=int)
    comps2 = _components_in_step0_2d(out, n1, n2, feas_flat)
    print(f"[STEP0 connect 2D] after-bridge components={len(comps2)} nodes={out.size}")
    return out

# =============================================================================
# Overlay helpers
# =============================================================================
def mst_edges_over_nodes_2d(P, nodes):
    nodes = np.array(sorted(set(map(int, nodes))), dtype=int)
    if nodes.size < 2:
        return []
    Q = P[nodes]
    D = np.linalg.norm(Q[:, None, :] - Q[None, :, :], axis=2)
    G = minimum_spanning_tree(coo_matrix(D)).tocoo()
    out = []
    for i, j in zip(G.row, G.col):
        out.append((int(nodes[int(i)]), int(nodes[int(j)])))
    return out


def overlay_train_val_mst_factory_2d(
    P,
    train_all_idx,
    train_q_idx,
    anchor_idx,
    knn_idx=None,
    knn_score_map=None,
    step0_overlay_edges=None,
):
    train_all_idx = np.array(sorted(set(map(int, train_all_idx))), dtype=int)
    train_q_idx = np.array(sorted(set(map(int, train_q_idx))), dtype=int)
    knn_idx = np.array([], dtype=int) if knn_idx is None else np.array(sorted(set(map(int, knn_idx))), dtype=int)

    if step0_overlay_edges:
        step0_overlay_edges = list({
            (int(a), int(b)) if int(a) < int(b) else (int(b), int(a))
            for (a, b) in step0_overlay_edges
            if int(a) != int(b)
        })
    else:
        step0_overlay_edges = []

    def _plot_mst(ax, idxs, color, lw, alpha, zorder=9999):
        idxs = np.array(sorted(set(map(int, idxs))), dtype=int)
        if idxs.size < 2:
            return
        mstE = mst_edges_over_nodes_2d(P, idxs)
        for gi, gj in mstE:
            ax.plot(
                [P[gi, 0], P[gj, 0]],
                [P[gi, 1], P[gj, 1]],
                color=color,
                lw=lw,
                alpha=alpha,
                solid_capstyle="round",
                clip_on=False,
                zorder=zorder,
            )

    def overlay(ax):
        def draw_train():
            tr = P[train_all_idx]
            ax.scatter(
                tr[:, 0],
                tr[:, 1],
                s=MS_TRAIN,
                marker=TRAIN_MARKER,
                facecolors=TRAIN_FACE,
                edgecolors=TRAIN_EDGE,
                linewidths=0.8,
                alpha=1.0,
                zorder=Z_TRAIN,
            )

        def draw_anchor():
            a = P[int(anchor_idx)]
            ax.scatter(
                [a[0]],
                [a[1]],
                s=MS_ANCHOR,
                marker="*",
                c="gold",
                edgecolors="k",
                linewidths=1.0,
                alpha=1.0,
                zorder=Z_ANCHOR,
            )

        knn_mappable = None

        def draw_knn():
            nonlocal knn_mappable
            knn_mappable = None
            if SHOW_CAND_OVERLAY and knn_idx.size > 0 and isinstance(knn_score_map, dict):
                cs = np.asarray([knn_score_map.get(int(g), np.nan) for g in knn_idx], float)
                msk = np.isfinite(cs)
                if np.any(msk):
                    pts = P[knn_idx[msk]]
                    cs2 = cs[msk]
                    v0 = float(np.percentile(cs2, 5))
                    v1 = float(np.percentile(cs2, 95))
                    if (not np.isfinite(v0)) or (not np.isfinite(v1)) or (v1 <= v0):
                        v0 = float(np.nanmin(cs2))
                        v1 = float(np.nanmax(cs2))
                    cnorm = mpl.colors.Normalize(vmin=v0, vmax=v1)
                    knn_mappable = ax.scatter(
                        pts[:, 0],
                        pts[:, 1],
                        s=CAND_MARKER_SIZE,
                        c=cs2,
                        cmap=CAND_CMAP,
                        norm=cnorm,
                        edgecolors="k",
                        linewidths=CAND_EDGE_LW,
                        alpha=CAND_ALPHA,
                        zorder=Z_KNN,
                    )

        def draw_mst_all():
            _plot_mst(ax, train_all_idx, MST_COLOR, LINE_MST_WIDTH, 0.95, zorder=12000)

        def draw_mst_q():
            _plot_mst(ax, train_q_idx, MST_Q_COLOR, LINE_MST_WIDTH * 1.15, 0.98, zorder=13000)

        def draw_step0_geometry_edges():
            if not step0_overlay_edges:
                return
            for gi, gj in step0_overlay_edges:
                ax.plot(
                    [P[gi, 0], P[gj, 0]],
                    [P[gi, 1], P[gj, 1]],
                    color="k",
                    lw=LINE_MST_WIDTH * 1.35,
                    alpha=1.0,
                    solid_capstyle="round",
                    clip_on=False,
                    zorder=20000,
                )

        if OVERLAY_ORDER_MODE == "mst_first":
            order = ["mst_all", "mst_q", "train", "anchor", "knn", "step0geom"]
        else:
            order = ["train", "anchor", "knn", "mst_all", "mst_q", "step0geom"]

        dispatch = {
            "mst_all": draw_mst_all,
            "mst_q": draw_mst_q,
            "train": draw_train,
            "anchor": draw_anchor,
            "knn": draw_knn,
            "step0geom": draw_step0_geometry_edges,
        }
        for key in order:
            fn = dispatch.get(key, None)
            if fn is not None:
                fn()

        if knn_mappable is not None:
            return {"knn_mappable": knn_mappable, "knn_label": CAND_CBAR_LABEL}
        return None

    overlay.figure_data = {
        "P": P,
        "train_all_idx": train_all_idx,
        "train_q_idx": train_q_idx,
        "anchor_idx": int(anchor_idx),
        "knn_idx": knn_idx,
        "knn_score_map": knn_score_map,
        "step0_overlay_edges": step0_overlay_edges,
    }
    return overlay

# =============================================================================
# Metrics plots
# =============================================================================
def plot_rmse_history_each_step(rmse_tr_hist, rmse_val_hist, step_name, target_label, train_set_label="paid"):
    fig, ax = plt.subplots(figsize=FIGSIZE_RMSE)
    its = np.arange(0, len(rmse_tr_hist))
    ax.plot(its, rmse_tr_hist, "-", lw=5, label=f"Train")
    ax.plot(its, rmse_val_hist, "-", lw=5, label=f"Val")
    ax.set_xlabel("AL step", fontsize=AXIS_LABEL_FZ)
    ax.set_ylabel("RMSE", fontsize=AXIS_LABEL_FZ)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis="x", labelsize=TICK_LABEL_FZ, pad=6)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FZ, pad=6)
    ax.legend(
        frameon=False,
        fontsize=LEGEND_FZ,
        loc="upper right",
        markerscale=2.0,
        handlelength=0.5,
        handletextpad=0.6,
        labelspacing=0.5,
        borderpad=0.2,
        borderaxespad=0.2,
    )
    fig_name = f"{target_label}_AL_step{step_name}_rmse_trace"
    save_numeric_figure_data(
        fig_name,
        np.column_stack([its, np.asarray(rmse_tr_hist, float), np.asarray(rmse_val_hist, float)]),
        "al_step,rmse_train,rmse_val",
    )
    savefig(fig, fig_name)


def plot_mst_history_each_step(mst_hist, step_name):
    fig, ax = plt.subplots(figsize=FIGSIZE_RMSE)
    steps = np.arange(len(mst_hist))
    ax.plot(steps, mst_hist, "-o", lw=5, ms=16, mfc="none")
    ax.set_xlabel("AL step", fontsize=AXIS_LABEL_FZ)
    ax.set_ylabel("MST length", fontsize=AXIS_LABEL_FZ, labelpad=18)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis="x", labelsize=TICK_LABEL_FZ, pad=6)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FZ, pad=6)
    fig_name = f"W3_AL_step{step_name}_mst_trace"
    save_numeric_figure_data(fig_name, np.column_stack([steps, np.asarray(mst_hist, float)]), "al_step,mst_length")
    savefig(fig, fig_name)


def plot_parity_each_step(model, X_tr, y_tr, X_val, y_val, ymu, ysd, step_name, target_label, train_set_label="paid"):
    mu_tr_z, _ = predict_mu_std_z(model, X_tr)
    mu_val_z, _ = predict_mu_std_z(model, X_val)
    yhat_tr = z2y(mu_tr_z, ymu, ysd)
    yhat_val = z2y(mu_val_z, ymu, ysd)

    fig, ax = plt.subplots(figsize=FIGSIZE_PARITY)
    ax.scatter(y_tr, yhat_tr, s=160, alpha=1, label=f"Train")
    ax.scatter(y_val, yhat_val, s=160, alpha=0.35, edgecolors="none", label="Val")

    allv = np.concatenate(
        [
            y_tr[np.isfinite(y_tr)],
            y_val[np.isfinite(y_val)],
            yhat_tr[np.isfinite(yhat_tr)],
            yhat_val[np.isfinite(yhat_val)],
        ]
    )
    if allv.size > 0:
        vmin = float(np.nanpercentile(allv, 1))
        vmax = float(np.nanpercentile(allv, 99))
    else:
        vmin, vmax = -1.0, 1.0

    ax.plot([vmin, vmax], [vmin, vmax], "k-", lw=2.0, alpha=0.7)
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_xlabel(r"$W_3^{\mathrm{True}}\,(k_{\mathrm{B}}T)$", fontsize=AXIS_LABEL_FZ)
    ax.set_ylabel(r"$W_3^{\mathrm{Pred}}\,(k_{\mathrm{B}}T)$", fontsize=AXIS_LABEL_FZ)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FZ, pad=6)
    ax.legend(
        frameon=False,
        fontsize=LEGEND_FZ,
        loc="upper left",
        markerscale=1.5,
        handlelength=0.2,
        handletextpad=0.6,
        labelspacing=0.5,
        borderpad=0.2,
        borderaxespad=0.2,
    )
    fig_name = f"{target_label}_AL_step{step_name}_parity"
    parity_data = np.row_stack(
        [
            np.column_stack([np.zeros(len(y_tr)), y_tr, yhat_tr]),
            np.column_stack([np.ones(len(y_val)), y_val, yhat_val]),
        ]
    )
    save_numeric_figure_data(fig_name, parity_data, "set_code_0_train_1_val,true_value,predicted_value")
    savefig(fig, fig_name)


def plot_lsqr_paid_vs_fullgrid_parity(U_paid, paid_nodes, U_full_map, step_name):
    paid_nodes = np.array(paid_nodes, dtype=int)
    u_paid = np.asarray(U_paid, float).reshape(-1)
    u_full = np.asarray(U_full_map[paid_nodes], float).reshape(-1)
    m = np.isfinite(u_paid) & np.isfinite(u_full)
    if not np.any(m):
        return
    u_paid = u_paid[m]
    u_full = u_full[m]
    r = rmse(u_full, u_paid)

    fig, ax = plt.subplots(figsize=FIGSIZE_PARITY)
    ax.scatter(u_full, u_paid, s=22, alpha=0.55, edgecolors="none")
    vmin = float(np.nanpercentile(np.r_[u_full, u_paid], 1))
    vmax = float(np.nanpercentile(np.r_[u_full, u_paid], 99))
    ax.plot([vmin, vmax], [vmin, vmax], "k-", lw=2.0, alpha=0.7)
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_xlabel(r"Full-grid LSQR $W_3\,(k_{\mathrm{B}}T)$", fontsize=AXIS_LABEL_FZ)
    ax.set_ylabel(r"On-the-fly LSQR $W_3\,(k_{\mathrm{B}}T)$", fontsize=AXIS_LABEL_FZ)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FZ, pad=6)
    ax.text(
        0.02,
        0.98,
        f"RMSE = {r:.4f}\nN = {u_full.size}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=LEGEND_FZ,
    )
    fig_name = f"LSQR_AL_step{step_name}_paid_vs_fullgrid_parity"
    save_numeric_figure_data(fig_name, np.column_stack([u_full, u_paid]), "full_grid_lsqr,on_the_fly_lsqr")
    savefig(fig, fig_name)

# =============================================================================
# Small training helpers
# =============================================================================
def make_train_xy_from_nodes(train_nodes, y_phys, Xs_feas, feas_pos):
    train_nodes = np.asarray(train_nodes, dtype=int)
    pos = feas_pos[train_nodes]
    m = pos >= 0
    if not np.any(m):
        return np.zeros((0, 3)), np.zeros((0,)), np.zeros((0,)), 0.0, 1.0

    X_tr = Xs_feas[pos[m]]
    y_tr = np.asarray(y_phys, float)[m]

    X_tr, y_tr = _sanitize_training_data(X_tr, y_tr)
    if X_tr.shape[0] == 0:
        return X_tr, y_tr, y_tr, 0.0, 1.0

    ymu, ysd = y_fit(y_tr)
    z_tr = y2z(y_tr, ymu, ysd)
    return X_tr, y_tr, z_tr, ymu, ysd


def eval_model_rmse(model, X, y_true_phys, ymu_model, ysd_model):
    mu_z, _ = predict_mu_std_z(model, X)
    yhat = z2y(mu_z, ymu_model, ysd_model)
    return rmse(y_true_phys, yhat)

# =============================================================================
# Per-step full plotting
# =============================================================================
def plot_all_figs_for_step(
    step_name,
    model,
    Xg,
    Yg,
    P,
    Xs_feas,
    xmu,
    xsd,
    ymu,
    ysd,
    normU,
    U_oracle_grid,
    train_all_set,
    train_q_set,
    anchor_g,
    W2anch_all,
    score_map_for_this_step=None,
    cand_overlay_idx_for_this_step=None,
    X_tr=None,
    y_tr=None,
    X_val=None,
    y_val=None,
    rmse_tr_hist=None,
    rmse_val_hist=None,
    mst_hist=None,
    step0_overlay_edges=None,
    target_mode="umb",
    target_label="Umb",
    symmetry_mode="sorted",
    train_set_label="paid",
    U_full_map=None,
    paid_nodes_step=None,
    U_paid_step=None,
):
    if cand_overlay_idx_for_this_step is None:
        cand_overlay_idx_for_this_step = np.array([], dtype=int)

    overlay = overlay_train_val_mst_factory_2d(
        P,
        train_all_idx=np.array(sorted(train_all_set), int),
        train_q_idx=np.array(sorted(train_q_set), int),
        anchor_idx=anchor_g,
        knn_idx=cand_overlay_idx_for_this_step,
        knn_score_map=score_map_for_this_step,
        step0_overlay_edges=step0_overlay_edges,
    )

    if (U_full_map is not None) and (paid_nodes_step is not None) and (U_paid_step is not None):
        plot_lsqr_paid_vs_fullgrid_parity(U_paid_step, paid_nodes_step, U_full_map, step_name)

    mu_target, std_target = gp_predict_on_slice(
        model, xmu, xsd, ymu, ysd, Yg, Xg, symmetry_mode
    )

    if str(target_mode).lower() == "residual":
        mu = mu_target + W2anch_all.reshape(Xg.shape, order="C")
    else:
        mu = mu_target

    Z_mu = mu
    Z_sig = std_target
    Z_oracle = U_oracle_grid
    Z_err = _abs_err(Z_mu, Z_oracle)
    Z_dW3 = Z_mu - W2anch_all.reshape(Xg.shape, order="C")

    mask2 = np.isfinite(Z_oracle)

    normErr = None  # automatic colorbar range for absolute error
    normDW3 = _norm_from_masked(Z_dW3, mask2)
    normSig = _norm_from_masked(Z_sig, mask2)

    plot_slice_2d_heat(
        Yg,
        Xg,
        Z_mu,
        mask2,
        f"W3_AL_step{step_name}_mu_slice2D",
        title=rf"Student $\mu$ diag slice, step {step_name}, {GRAFT}",
        clabel=LAB_C,
        cmap="viridis",
        norm=normU,
        overlay=overlay,
    )
    plot_slice_3d_surface(
        Yg,
        Xg,
        Z_mu,
        mask2,
        f"W3_AL_step{step_name}_mu_slice3D",
        title=rf"Student $\mu$ diag surface, step {step_name}, {GRAFT}",
        clabel=LAB_C,
        cmap="viridis",
        norm=normU,
    )

    plot_slice_2d_heat(
        Yg,
        Xg,
        Z_sig,
        mask2,
        f"W3_AL_step{step_name}_sigma_slice2D",
        title=rf"Student $\sigma$ diag slice, step {step_name}, {GRAFT}",
        clabel=r"$\sigma^{\mathrm{Pred}}$",
        cmap="plasma",
        norm=normSig,
        overlay=overlay,
    )
    plot_slice_3d_surface(
        Yg,
        Xg,
        Z_sig,
        mask2,
        f"W3_AL_step{step_name}_sigma_slice3D",
        title=rf"Student $\sigma$ diag surface, step {step_name}, {GRAFT}",
        clabel=r"$\sigma^{\mathrm{Pred}}$",
        cmap="plasma",
        norm=normSig,
    )

    plot_slice_2d_heat(
        Yg,
        Xg,
        Z_err,
        mask2,
        f"W3_AL_step{step_name}_abserr_slice2D",
        title=rf"Absolute error diag slice, step {step_name}, {GRAFT}",
        clabel=LAB_ABSERR,
        cmap=ABSERR_CMAP,
        norm=normErr,
    )
    plot_slice_3d_surface(
        Yg,
        Xg,
        Z_err,
        mask2,
        f"W3_AL_step{step_name}_abserr_slice3D",
        title=rf"Absolute error diag surface, step {step_name}, {GRAFT}",
        clabel=LAB_ABSERR,
        cmap=ABSERR_CMAP,
        norm=normErr,
    )

    plot_slice_2d_heat(
        Yg,
        Xg,
        Z_dW3,
        mask2,
        f"DeltaW3_AL_step{step_name}_extracted_DeltaW3Pred_diag_slice2D",
        title=rf"Student extracted $\mathsf{{\Delta}} W_3^{{\mathrm{{Pred}}}}$ diag slice, step {step_name}, {GRAFT}",
        clabel=LAB_DW3,
        cmap="viridis",
        norm=normDW3,
    )
    plot_slice_3d_surface(
        Yg,
        Xg,
        Z_dW3,
        mask2,
        f"DeltaW3_AL_step{step_name}_extracted_DeltaW3Pred_diag_slice3D",
        title=rf"Student extracted $\mathsf{{\Delta}} W_3^{{\mathrm{{Pred}}}}$ diag surface, step {step_name}, {GRAFT}",
        clabel=LAB_DW3,
        cmap="viridis",
        norm=normDW3,
    )

    if (X_tr is not None) and (y_tr is not None) and (X_val is not None) and (y_val is not None):
        plot_parity_each_step(
            model,
            X_tr,
            y_tr,
            X_val,
            y_val,
            ymu,
            ysd,
            step_name,
            target_label,
            train_set_label=train_set_label,
        )

    if (rmse_tr_hist is not None) and (rmse_val_hist is not None):
        plot_rmse_history_each_step(
            rmse_tr_hist,
            rmse_val_hist,
            step_name,
            target_label,
            train_set_label=train_set_label,
        )

    if mst_hist is not None:
        plot_mst_history_each_step(mst_hist, step_name)

# =============================================================================
# MAIN
# =============================================================================
def main():
    global FIG_DIR

    args = parse_cli()
    START_MODE = args.start_mode
    TARGET_MODE = args.target_mode
    SYMM_MODE = args.symmetry_mode
    KMODE = args.kernel_mode
    LABEL_SOURCE = args.label_source

    USE_LSQR = LABEL_SOURCE in ("lsqr", "pathint")
    TRAIN_SET_LABEL = "queried" if LABEL_SOURCE == "oracle_queried" else "paid"
    TARGET_LABEL = r"$W_3$" if TARGET_MODE == "umb" else "Residual"

    print(
        f"[CONFIG] start_mode={START_MODE}  target_mode={TARGET_MODE}  "
        f"sym={SYMM_MODE}  kernel={KMODE}  label_source={LABEL_SOURCE}"
    )

    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    np.set_printoptions(precision=4, suppress=True)
    tf.keras.utils.set_random_seed(SEED)
    gpflow.config.set_default_float(np.float64)
    gpflow.config.set_default_jitter(JITTER_DEF)

    FIG_DIR = Path(
        f"./figs_{RUN_TAG}_stage2_AL_toy2diag_oracle_2D_forceint_"
        f"edges{INIT_EDGE_COUNT}_face{INIT_FACE_COUNT}_"
        f"{START_MODE}_{TARGET_MODE}_sym{SYMM_MODE}_k{KMODE}_lab{LABEL_SOURCE}"
    )
    FIG_DIR.mkdir(exist_ok=True)
    print(f"[OUT] FIG_DIR = {FIG_DIR}")
    if SAVE_FIGURE_DATA:
        print(f"[OUT] FIG_DATA_DIR = {FIG_DIR / FIG_DATA_SUBDIR}")

    rng = np.random.default_rng(SEED)
    p2 = None
    p3 = None

    # -------------------------------------------------------------------------
    # 2D diagonal slice grid
    # -------------------------------------------------------------------------
    d13s = np.linspace(D13_MIN, D13_MAX, SLICE_N13)
    d12s = np.linspace(D12_MIN, D12_MAX, SLICE_N12)
    Yg, Xg = np.meshgrid(d12s, d13s, indexing="ij")  # Y=d12, X=d13
    n1, n2 = Xg.shape

    feas = slice_feasible_mask(Yg, Xg)

    Umb_raw, W3sw_raw, _ = Umb_slice(Yg, Xg, p2, p3)
    Umb_grid = np.where(feas, Umb_raw, np.nan)
    W3sw_grid = np.where(feas, W3sw_raw, np.nan)

    feas_flat = np.isfinite(Umb_grid.ravel(order="C"))
    feasible_g = np.where(feas_flat)[0]
    if feasible_g.size == 0:
        raise RuntimeError("No feasible points on this diagonal slice.")

    P = np.c_[Xg.ravel(order="C"), Yg.ravel(order="C")]  # (d13, d12)

    anchor_target = np.array([D13_MAX, D12_MAX], float)
    kdt_feas = cKDTree(P[feasible_g])
    anchor_g = int(feasible_g[int(kdt_feas.query(anchor_target, k=1)[1])])

    if USE_EDGE_INIT_2D:
        step0_nodes, step0_extra_edges = build_step0_geometry_2d(
            anchor_g=anchor_g,
            n1=n1,
            n2=n2,
            feas_mask_flat=feas_flat,
            edge_count=INIT_EDGE_COUNT,
            diag_count=INIT_FACE_COUNT,
        )
        step0_nodes_connected = enforce_step0_connected_2d(
            step0_nodes, anchor_g, P, n1, n2, feas_flat
        )
    else:
        step0_nodes_connected = np.array([int(anchor_g)], dtype=int)
        step0_extra_edges = None

    print(
        f"[STEP0 geometry 2D] requested edges/diag = {INIT_EDGE_COUNT}/{INIT_FACE_COUNT}"
    )
    print(
        f"[STEP0 geometry 2D] nodes(after connect)={len(step0_nodes_connected)} "
        f"extra_edges={0 if step0_extra_edges is None else len(step0_extra_edges)} "
        f"anchor_g={anchor_g}"
    )

    step0_overlay_edges = step0_extra_edges if (step0_extra_edges is not None and len(step0_extra_edges) > 0) else None

    U_raw_flat = Umb_grid.ravel(order="C")
    U_anchor_raw = float(U_raw_flat[anchor_g])
    U_ref = U_raw_flat - U_anchor_raw

    d13_a, d12_a = P[anchor_g]
    W2_anchor = float(W2sum_slice(d12_a, d13_a, p2))

    W2sum_flat = W2sum_slice(P[:, 1], P[:, 0], p2)
    W2anch_flat = W2sum_flat - W2_anchor
    RES_ref = U_ref - W2anch_flat

    plot_g = feasible_g

    normU = _norm_from_clim()
    if normU is None:
        normU = _norm_from_masked(U_ref.reshape(Xg.shape, order="C"), feas)

    # -------------------------------------------------------------------------
    # Oracle plots
    # -------------------------------------------------------------------------
    plot_slice_2d_heat(
        Yg,
        Xg,
        U_ref.reshape(Xg.shape, order="C"),
        feas,
        "Umb_01_oracle_Umb_slice2D",
        title=rf"Toy oracle $W_3^{{\mathrm{{True}}}}$ diag slice, {GRAFT}",
        clabel=LAB_W3_TRUE,
        cmap="viridis",
        norm=normU,
    )
    plot_slice_3d_surface(
        Yg,
        Xg,
        U_ref.reshape(Xg.shape, order="C"),
        feas,
        "Umb_01_oracle_Umb_slice3D",
        title=rf"Toy oracle $W_3^{{\mathrm{{True}}}}$ diag surface, {GRAFT}",
        clabel=LAB_W3_TRUE,
        cmap="viridis",
        norm=normU,
    )

    normDW3_oracle = _norm_from_masked(W3sw_grid, feas)
    plot_slice_2d_heat(
        Yg,
        Xg,
        W3sw_grid,
        feas,
        "Umb_02_oracle_deltaW3_slice2D",
        title=rf"Toy oracle $\mathsf{{\Delta}} W_3^{{\mathrm{{True}}}}$ diag slice, {GRAFT}",
        clabel=LAB_DW3_TRUE,
        cmap="viridis",
        norm=normDW3_oracle,
    )
    plot_slice_3d_surface(
        Yg,
        Xg,
        W3sw_grid,
        feas,
        "Umb_02_oracle_deltaW3_slice3D",
        title=rf"Toy oracle $\mathsf{{\Delta}} W_3^{{\mathrm{{True}}}}$ diag surface, {GRAFT}",
        clabel=LAB_DW3_TRUE,
        cmap="viridis",
        norm=normDW3_oracle,
    )

    # -------------------------------------------------------------------------
    # LSQR teacher force machinery
    # -------------------------------------------------------------------------
    U_full_map = None
    Fx_known = np.full(P.shape[0], np.nan, float)
    Fy_known = np.full(P.shape[0], np.nan, float)

    Fx_slice, Fy_slice = forces_Umb_slice(Yg, Xg)
    Fx_slice = np.where(feas, Fx_slice, np.nan)
    Fy_slice = np.where(feas, Fy_slice, np.nan)
    Fx_flat = Fx_slice.ravel(order="C")
    Fy_flat = Fy_slice.ravel(order="C")

    def teacher_query_force_at_nodes(nodes):
        nodes = np.array(sorted(set(map(int, nodes))), dtype=int)
        if nodes.size == 0:
            return
        nodes = nodes[feas_flat[nodes]]
        if nodes.size == 0:
            return
        Fx_known[nodes] = Fx_flat[nodes]
        Fy_known[nodes] = Fy_flat[nodes]

    if USE_LSQR and DO_FULLGRID_LSQR_DIAGNOSTIC:
        teacher_query_force_at_nodes(feasible_g)
        U_full, nodes_full = lsqr_U_on_known_nodes_2d(
            known_nodes=feasible_g,
            anchor_g=anchor_g,
            P=P,
            Fx=Fx_known,
            Fy=Fy_known,
            n1=n1,
            n2=n2,
            mode="8",
            iter_lim=12000,
            dw_clip=None,
        )
        U_full_map = np.full(P.shape[0], np.nan, float)
        U_full_map[nodes_full] = U_full
        keep = np.isfinite(U_full_map[feasible_g]) & np.isfinite(U_ref[feasible_g])
        print(
            f"[LSQR full-grid diagnostic 2D] RMSE vs oracle (feasible) = "
            f"{rmse(U_ref[feasible_g][keep], U_full_map[feasible_g][keep]):.6f}"
        )
        if PLOT_FULLGRID_LSQR_DIAGNOSTIC:
            plot_slice_2d_heat(
                Yg,
                Xg,
                U_full_map.reshape(Xg.shape, order="C"),
                feas,
                "Umb_03_diag_LSQR_fullgrid_slice2D",
                title=rf"LSQR PMF from toy oracle forces (diag diagnostic), {GRAFT}",
                clabel=LAB_C,
                cmap="viridis",
                norm=normU,
            )
        Fx_known[:] = np.nan
        Fy_known[:] = np.nan

    # -------------------------------------------------------------------------
    # GP coordinates on feasible set
    # -------------------------------------------------------------------------
    P_feas = P[feasible_g]
    P_feat_feas = _features_from_slice_XY(P_feas, symmetry_mode=SYMM_MODE)
    xmu, xsd = standardize_fit(P_feat_feas)
    Xs_feas = standardize_apply(P_feat_feas, xmu, xsd)

    W2sum_feas = W2sum_slice(P_feas[:, 1], P_feas[:, 0], p2)
    W2anch_feas = W2sum_feas - W2_anchor

    feas_pos = -np.ones(P.shape[0], dtype=int)
    feas_pos[feasible_g] = np.arange(feasible_g.size, dtype=int)

    # -------------------------------------------------------------------------
    # Validation set
    # -------------------------------------------------------------------------
    exclude0 = np.union1d(step0_nodes_connected, np.array([anchor_g], dtype=int))
    val_pool = np.setdiff1d(feasible_g, exclude0)
    n_val = int(np.clip(VAL_FRAC * feasible_g.size, VAL_MIN, VAL_MAX))
    if val_pool.size == 0:
        val_pool = feasible_g[feasible_g != anchor_g]
    val_g = rng.choice(val_pool, size=min(n_val, val_pool.size), replace=False)

    is_val = np.zeros(P.shape[0], dtype=bool)
    is_val[val_g] = True
    is_val[int(anchor_g)] = False

    query_mask = feas_flat & (~is_val)
    query_mask[int(anchor_g)] = True
    path_mask = query_mask.copy()

    np.savetxt(
        FIG_DIR / "validation_nodes.csv",
        val_g, fmt="%d", header="node_index", comments=""
    )

    def check_validation(nodes, name):
        overlap = set(map(int, nodes)) & set(val_g.tolist())
        if overlap:
            raise RuntimeError(
                f"VALIDATION LEAKAGE in {name}: "
                f"{len(overlap)} overlapping nodes"
            )


    y_val_true = U_ref[val_g] if TARGET_MODE == "umb" else RES_ref[val_g]
    X_val = Xs_feas[feas_pos[val_g]]

    def _oracle_target_on_nodes(nodes):
        nodes = np.asarray(nodes, dtype=int)
        if TARGET_MODE == "umb":
            return np.asarray(U_ref[nodes], float)
        return np.asarray(RES_ref[nodes], float)

    def _build_training_nodes_and_labels(label_source, train_all, train_q):
        ls = str(label_source).lower()

        if ls == "pathint":
            U_paid, paid_nodes = path_integrated_U_on_known_nodes_2d(
                known_nodes=np.array(sorted(set(map(int, train_all))), dtype=int),
                anchor_g=anchor_g,
                P=P,
                n1=n1,
                n2=n2,
                mode="8",
                quad_order=PATHINT_QUAD_ORDER,
            )
            ok = np.isfinite(U_paid)
            paid_nodes = paid_nodes[ok]
            U_paid = U_paid[ok]
            if paid_nodes.size == 0:
                return np.array([], int), np.array([], float), None, None
            if paid_nodes.size > 0:
                u_true = np.asarray(U_ref[paid_nodes], float)
                err = U_paid - u_true
                bias = float(np.mean(err))
                print(
                    "[PATHINT vs ORACLE paid] "
                    f"n={paid_nodes.size} "
                    f"RMSE={np.sqrt(np.mean(err**2)):.6e} "
                    f"MAE={np.mean(np.abs(err)):.6e} "
                    f"bias={bias:.6e} "
                    f"RMSE_bias_removed={np.sqrt(np.mean((err-bias)**2)):.6e} "
                    f"maxabs={np.max(np.abs(err)):.6e}"
                )
            y = U_paid.copy()
            if TARGET_MODE == "residual":
                y = y - W2anch_flat[paid_nodes]
            return paid_nodes, y, paid_nodes, U_paid

        if ls == "lsqr":
            U_paid, paid_nodes = lsqr_U_on_known_nodes_2d(
                known_nodes=np.array(sorted(set(map(int, train_all))), dtype=int),
                anchor_g=anchor_g,
                P=P,
                Fx=Fx_known,
                Fy=Fy_known,
                n1=n1,
                n2=n2,
                mode="8",
                iter_lim=8000,
                dw_clip=DW_CLIP,
            )
            ok = np.isfinite(U_paid)
            paid_nodes = paid_nodes[ok]
            U_paid = U_paid[ok]
            if paid_nodes.size == 0:
                return np.array([], int), np.array([], float), None, None
            y = U_paid.copy()
            if TARGET_MODE == "residual":
                y = y - W2anch_flat[paid_nodes]
            return paid_nodes, y, paid_nodes, U_paid

        if ls == "oracle_paid":
            nodes = np.array(sorted(set(map(int, train_all))), dtype=int)
            nodes = nodes[feas_flat[nodes]]
            return nodes, _oracle_target_on_nodes(nodes), None, None

        if ls == "oracle_queried":
            nodes = np.array(sorted(set(map(int, train_q))), dtype=int)
            nodes = nodes[feas_flat[nodes]]
            return nodes, _oracle_target_on_nodes(nodes), None, None

        raise ValueError(f"Unknown label_source={label_source}")

    # -------------------------------------------------------------------------
    # Step-0 paid / queried sets
    # -------------------------------------------------------------------------
    train_q = set(map(int, step0_nodes_connected.tolist()))
    train_all = set(train_q)

    if (not USE_EDGE_INIT_2D) and (len(train_q) <= 1):
        remain = np.setdiff1d(feasible_g, np.array([int(anchor_g)] + list(val_g), dtype=int))
        remain = remain[query_mask[remain]]
        fp = [int(anchor_g)]
        if remain.size > 0:
            dmin = np.linalg.norm(P[remain] - P[anchor_g], axis=1)
            while len(fp) < AL_SEEDS and remain.size > 0:
                j = int(remain[np.argmax(dmin)])
                fp.append(j)
                mask = remain != j
                if not np.any(mask):
                    break
                dmin = np.minimum(dmin[mask], np.linalg.norm(P[remain[mask]] - P[j], axis=1))
                remain = remain[mask]
        train_q.update(fp)
        train_all = set(train_q)
        tr_set = set(sorted(train_all))
        seed_nodes = sorted(tr_set)
        mst_edges = mst_edges_over_nodes_2d(P, seed_nodes)
        for u, v in mst_edges:
            _ = shortest_path_add_2d(tr_set, int(u), int(v), P, n1, n2, path_mask)
        train_all = set(tr_set)

    check_validation(train_all, "initial paid nodes")
    check_validation(train_q, "initial queried nodes")

    if USE_LSQR:
        teacher_query_force_at_nodes(train_all)

    train_nodes, y_phys_raw, paid_nodes_lsqr, U_paid_lsqr = _build_training_nodes_and_labels(
        LABEL_SOURCE, train_all, train_q
    )
    if train_nodes.size == 0:
        raise RuntimeError("Initial labeling produced no training labels.")

    check_validation(train_nodes, "GP training labels")
    X_tr, y_tr, z_tr, ymu, ysd = make_train_xy_from_nodes(train_nodes, y_phys_raw, Xs_feas, feas_pos)
    if X_tr.shape[0] < 2:
        raise RuntimeError("Not enough training points after sanitization.")

    prev_m = build_gpr(X_tr, z_tr, KMODE, KERNEL_SPEC, ARD, INIT_ITERS, m_prev=None)
    prev_ymu, prev_ysd = ymu, ysd

    rtr0 = eval_model_rmse(prev_m, X_tr, y_tr, prev_ymu, prev_ysd)
    rv0 = eval_model_rmse(prev_m, X_val, y_val_true, prev_ymu, prev_ysd)

    rmse_tr_hist = [rtr0]
    rmse_val_hist = [rv0]
    mst_hist = [mst_total_length(P, np.array(sorted(train_all), int))]

    print(
        f"[AL init] target={TARGET_MODE}  paid={len(train_all)} queried={len(train_q)}  "
        f"RMSE(train-{TRAIN_SET_LABEL}, {TARGET_LABEL})={rtr0:.4f}  "
        f"RMSE(val-oracle, {TARGET_LABEL})={rv0:.4f}"
    )

    pending_plot = {
        "step_name": "00",
        "model": prev_m,
        "ymu": prev_ymu,
        "ysd": prev_ysd,
        "train_all": set(train_all),
        "train_q": set(train_q),
        "X_tr": X_tr,
        "y_tr": y_tr.copy(),
        "X_val": X_val,
        "y_val": y_val_true,
        "paid_nodes": (paid_nodes_lsqr.copy() if paid_nodes_lsqr is not None else None),
        "U_paid": (U_paid_lsqr.copy() if U_paid_lsqr is not None else None),
    }

    # -------------------------------------------------------------------------
    # min-sep in standardized feature space
    # -------------------------------------------------------------------------
    subN = min(6000, Xs_feas.shape[0])
    if subN >= 2:
        sub_idx = (
            rng.choice(np.arange(Xs_feas.shape[0]), size=subN, replace=False)
            if subN < Xs_feas.shape[0]
            else np.arange(subN)
        )
        tree_sub = cKDTree(Xs_feas[sub_idx])
        dnn, _ = tree_sub.query(Xs_feas[sub_idx], k=2)
        base_h = float(np.median(dnn[:, 1])) if dnn.ndim == 2 and dnn.shape[1] >= 2 else 0.15
    else:
        base_h = 0.15
    base_h = max(base_h, 0.10)
    min_sep = MIN_SEP_MULT * base_h

    # -------------------------------------------------------------------------
    # Active learning loop
    # -------------------------------------------------------------------------
    for it in range(AL_ITERS):
        mask_unl = np.ones(P.shape[0], bool)
        if len(train_all) > 0:
            mask_unl[list(train_all)] = False
        if val_g.size > 0:
            mask_unl[val_g] = False

        cand_g_all = feasible_g[mask_unl[feasible_g]]
        if cand_g_all.size == 0:
            print(f"[AL {it+1:02d}] stop: no candidates left.")
            break

        _, std_old_z = predict_mu_std_z(prev_m, Xs_feas)

        tr_nodes_all = np.array(sorted(train_all), dtype=int)
        tr_pos_all = feas_pos[tr_nodes_all]
        tr_pos_all = tr_pos_all[tr_pos_all >= 0]
        if tr_pos_all.size == 0:
            tr_pos_all = np.array([feas_pos[anchor_g]], int)

        kd = cKDTree(Xs_feas[tr_pos_all])
        cand_pos_all = feas_pos[cand_g_all]
        d_to_train_all, _ = kd.query(Xs_feas[cand_pos_all], k=1)

        gate = d_to_train_all >= min_sep
        cand_g = cand_g_all[gate]
        cand_pos = cand_pos_all[gate]
        d_to_train = d_to_train_all[gate]

        if cand_g.size < max(1, int(POOL_MIN_FRACTION * feasible_g.size)):
            old = min_sep
            min_sep *= ADAPT_MINSEP_DECAY
            print(f"[AL {it+1:02d}] relax min_sep: {old:.4g} -> {min_sep:.4g} (pool {cand_g.size})")
            gate = d_to_train_all >= min_sep
            cand_g = cand_g_all[gate]
            cand_pos = cand_pos_all[gate]
            d_to_train = d_to_train_all[gate]

        if cand_g.size == 0:
            cand_g = cand_g_all
            cand_pos = cand_pos_all
            d_to_train = d_to_train_all

        std_c = std_old_z[cand_pos]
        noise_var = float(prev_m.likelihood.variance.numpy())

        info = 0.5 * np.log1p((std_c**2) / (noise_var + 1e-12))
        spacing = np.minimum(d_to_train, 4.0 * base_h) ** 0.5
        score = info * spacing
        score = np.nan_to_num(score, nan=-np.inf, posinf=np.finfo(float).max, neginf=-np.inf)
        print(f"Score stats: min={score.min():.3f}, max={score.max():.3f}, std={score.std():.3f}")

        order = np.argsort(-score)
        if order.size == 0 or (not np.isfinite(score).any()):
            order = np.argsort(-std_c)

        take_g = []
        taken = []
        r_min = base_h
        for idx in order:
            g = int(cand_g[int(idx)])
            p = Xs_feas[int(cand_pos[int(idx)])]
            if not taken:
                take_g.append(g)
                taken.append(p)
            else:
                dmin_now = np.min(np.linalg.norm(p - np.vstack(taken), axis=1))
                if dmin_now >= r_min:
                    take_g.append(g)
                    taken.append(p)
            if len(take_g) >= AL_BATCH:
                break
        if len(take_g) == 0:
            take_g = [int(x) for x in cand_g[order[:AL_BATCH]]]
        take_g = np.array(take_g, dtype=int)

        score_map = {int(cand_g[i]): float(score[i]) for i in range(cand_g.size)}

        cand_overlay = set()
        if SHOW_CAND_OVERLAY and cand_g.size > 0:
            tree_c = cKDTree(Xs_feas[cand_pos])
            k_use = min(int(KNN_K), cand_g.size)
            for g in take_g:
                _, nn_loc = tree_c.query(Xs_feas[feas_pos[g]], k=k_use)
                nn_loc = np.atleast_1d(nn_loc)
                for kk in nn_loc:
                    cand_overlay.add(int(cand_g[int(kk)]))

        cand_overlay_idx = np.array(sorted(cand_overlay), dtype=int)
        if cand_overlay_idx.size > CAND_SHOW_N:
            cs = np.asarray([score_map.get(int(g), -np.inf) for g in cand_overlay_idx], float)
            keep = np.argsort(-cs)[:CAND_SHOW_N]
            cand_overlay_idx = cand_overlay_idx[keep]

        plot_all_figs_for_step(
            pending_plot["step_name"],
            pending_plot["model"],
            Xg,
            Yg,
            P,
            Xs_feas,
            xmu,
            xsd,
            pending_plot["ymu"],
            pending_plot["ysd"],
            normU,
            U_ref.reshape(Xg.shape, order="C"),
            pending_plot["train_all"],
            pending_plot["train_q"],
            anchor_g,
            W2anch_flat,
            score_map_for_this_step=score_map,
            cand_overlay_idx_for_this_step=cand_overlay_idx,
            X_tr=pending_plot["X_tr"],
            y_tr=pending_plot["y_tr"],
            X_val=pending_plot["X_val"],
            y_val=pending_plot["y_val"],
            rmse_tr_hist=rmse_tr_hist,
            rmse_val_hist=rmse_val_hist,
            mst_hist=mst_hist,
            step0_overlay_edges=step0_overlay_edges,
            target_mode=TARGET_MODE,
            target_label=TARGET_LABEL,
            symmetry_mode=SYMM_MODE,
            train_set_label=TRAIN_SET_LABEL,
            U_full_map=U_full_map,
            paid_nodes_step=pending_plot.get("paid_nodes", None),
            U_paid_step=pending_plot.get("U_paid", None),
        )

        for g in take_g.tolist():
            train_q.add(int(g))

        train_set_all = set(train_all)
        pending = [int(g) for g in take_g.tolist()]
        newly_paid = []

        while pending:
            cur_nodes = np.array(sorted(train_set_all), dtype=int)
            if cur_nodes.size == 0:
                cur_nodes = np.array([int(anchor_g)], dtype=int)
                train_set_all.add(int(anchor_g))

            kd_cur = cKDTree(P[cur_nodes])
            dists, locs = kd_cur.query(P[pending], k=1)
            best_pos = int(np.argmin(np.atleast_1d(dists)))
            g_new = int(pending[best_pos])
            g_tgt = int(cur_nodes[int(np.atleast_1d(locs)[best_pos])])

            added_path = shortest_path_add_2d(train_set_all, g_new, g_tgt, P, n1, n2, path_mask)
            newly_paid.extend(added_path)
            pending.pop(best_pos)

        newly_paid = [int(g) for g in newly_paid if path_mask[int(g)]]
        if USE_LSQR:
            teacher_query_force_at_nodes(newly_paid)

        train_all = set(train_set_all)

        check_validation(train_all, f"step {it+1} paid")
        check_validation(train_q, f"step {it+1} queried")

        print(f"[VALIDATION CHECK] step={it+1} overlap=0")

        train_nodes, y_phys_raw, paid_nodes_lsqr, U_paid_lsqr = _build_training_nodes_and_labels(
            LABEL_SOURCE, train_all, train_q
        )
        if train_nodes.size == 0:
            print(f"[AL {it+1:02d}] WARNING: no training labels produced; skipping update.")
            continue

        check_validation(train_nodes, "GP training labels")
        X_tr, y_tr, z_tr, ymu, ysd = make_train_xy_from_nodes(train_nodes, y_phys_raw, Xs_feas, feas_pos)
        if X_tr.shape[0] < 2:
            print(f"[AL {it+1:02d}] WARNING: too few points after sanitization; skipping update.")
            continue

        rtr_prev = eval_model_rmse(prev_m, X_tr, y_tr, prev_ymu, prev_ysd)
        rv_prev = eval_model_rmse(prev_m, X_val, y_val_true, prev_ymu, prev_ysd)

        mprev_for_build = prev_m if (START_MODE == "warm") else None

        try:
            m_new = build_gpr(X_tr, z_tr, KMODE, KERNEL_SPEC, ARD, 800, m_prev=mprev_for_build)
        except Exception as e:
            print(f"[AL {it+1:02d}] WARN: GP build failed ({e}); retrying with safer settings.")
            gpflow.config.set_default_jitter(1e-2)
            m_new = build_gpr(X_tr, z_tr, KMODE, KERNEL_SPEC, ARD, 400, m_prev=mprev_for_build)

        rtr_new = eval_model_rmse(m_new, X_tr, y_tr, ymu, ysd)
        rv_new = eval_model_rmse(m_new, X_val, y_val_true, ymu, ysd)

        mu_new_tr_z, std_new_tr_z = predict_mu_std_z(m_new, X_tr)
        ok_new = (
            np.all(np.isfinite(mu_new_tr_z))
            and np.all(np.isfinite(std_new_tr_z))
            and np.all(std_new_tr_z > 0)
            and np.isfinite(rtr_new)
        )

        if ok_new and (rtr_new <= rtr_prev * (1.0 + ACCEPT_TOL)):
            prev_m = m_new
            prev_ymu, prev_ysd = ymu, ysd
            rtr_acc, rv_acc = rtr_new, rv_new
            acc_flag = "ACCEPT"
        else:
            rtr_acc, rv_acc = rtr_prev, rv_prev
            acc_flag = "REJECT"

        rmse_tr_hist.append(rtr_acc)
        rmse_val_hist.append(rv_acc)
        mst_hist.append(mst_total_length(P, np.array(sorted(train_all), int)))

        step_next = f"{it+1:02d}"
        print(
            f"[AL {step_next}] {acc_flag}  target={TARGET_MODE}  paid={len(train_all)} "
            f"queried={len(train_q)}  RMSE(train-{TRAIN_SET_LABEL}, {TARGET_LABEL})={rtr_acc:.4f}  "
            f"RMSE(val-oracle, {TARGET_LABEL})={rv_acc:.4f}"
        )

        pending_plot = {
            "step_name": step_next,
            "model": prev_m,
            "ymu": prev_ymu,
            "ysd": prev_ysd,
            "train_all": set(train_all),
            "train_q": set(train_q),
            "X_tr": X_tr,
            "y_tr": y_tr.copy(),
            "X_val": X_val,
            "y_val": y_val_true,
            "paid_nodes": (paid_nodes_lsqr.copy() if paid_nodes_lsqr is not None else None),
            "U_paid": (U_paid_lsqr.copy() if U_paid_lsqr is not None else None),
        }

    # -------------------------------------------------------------------------
    # Final plot
    # -------------------------------------------------------------------------
    plot_all_figs_for_step(
        pending_plot["step_name"],
        pending_plot["model"],
        Xg,
        Yg,
        P,
        Xs_feas,
        xmu,
        xsd,
        pending_plot["ymu"],
        pending_plot["ysd"],
        normU,
        U_ref.reshape(Xg.shape, order="C"),
        pending_plot["train_all"],
        pending_plot["train_q"],
        anchor_g,
        W2anch_flat,
        score_map_for_this_step=None,
        cand_overlay_idx_for_this_step=np.array([], dtype=int),
        X_tr=pending_plot["X_tr"],
        y_tr=pending_plot["y_tr"],
        X_val=pending_plot["X_val"],
        y_val=pending_plot["y_val"],
        rmse_tr_hist=rmse_tr_hist,
        rmse_val_hist=rmse_val_hist,
        mst_hist=mst_hist,
        step0_overlay_edges=step0_overlay_edges,
        target_mode=TARGET_MODE,
        target_label=TARGET_LABEL,
        symmetry_mode=SYMM_MODE,
        train_set_label=TRAIN_SET_LABEL,
        U_full_map=U_full_map,
        paid_nodes_step=pending_plot.get("paid_nodes", None),
        U_paid_step=pending_plot.get("U_paid", None),
    )

    # -------------------------------------------------------------------------
    # Summary traces
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=FIGSIZE_RMSE)
    its = np.arange(1, len(rmse_tr_hist) + 1)
    ax.plot(its, rmse_tr_hist, "-", lw=5, ms=80, label=f"Train")
    ax.plot(its, rmse_val_hist, "-", lw=5, ms=80, label=f"Val")
    ax.set_xlabel("Iteration", fontsize=AXIS_LABEL_FZ)
    ax.set_ylabel(f"RMSE ({TARGET_LABEL})", fontsize=AXIS_LABEL_FZ)
    ax.set_xlim(0.5, len(its) + 0.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis="x", labelsize=TICK_LABEL_FZ, pad=6)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FZ, pad=6)
    ax.legend(
        frameon=False,
        fontsize=LEGEND_FZ,
        loc="upper right",
        markerscale=2.0,
        handlelength=2.0,
        handletextpad=0.6,
        labelspacing=0.5,
        borderpad=0.2,
        borderaxespad=0.2,
    )
    fig_name = "W3_90_rmse_vs_iter" if TARGET_MODE == "umb" else f"{TARGET_LABEL}_90_rmse_vs_iter"
    save_numeric_figure_data(
        fig_name,
        np.column_stack([its, np.asarray(rmse_tr_hist, float), np.asarray(rmse_val_hist, float)]),
        "iteration,rmse_train,rmse_val",
    )
    savefig(fig, fig_name)

    fig, ax = plt.subplots(figsize=FIGSIZE_RMSE)
    steps = np.arange(len(mst_hist))
    ax.plot(steps, mst_hist, "-o", lw=5, ms=16, mfc="none")
    ax.set_xlabel("AL step", fontsize=AXIS_LABEL_FZ)
    ax.set_ylabel("MST length", fontsize=AXIS_LABEL_FZ)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis="x", labelsize=TICK_LABEL_FZ, pad=6)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FZ, pad=6)
    save_numeric_figure_data("W3_91_mst_vs_iter", np.column_stack([steps, np.asarray(mst_hist, float)]), "al_step,mst_length")
    savefig(fig, "W3_91_mst_vs_iter")

    steps_out = np.arange(len(rmse_tr_hist), dtype=int)
    metrics_out = np.column_stack(
        [
            steps_out,
            np.asarray(rmse_tr_hist, float),
            np.asarray(rmse_val_hist, float),
            np.asarray(mst_hist, float),
        ]
    )
    np.savetxt(
        FIG_DIR / "al_metrics.csv",
        metrics_out,
        header="step rmse_train rmse_val mst_length",
        fmt=["%d", "%.10f", "%.10f", "%.10f"],
    )

    plt.close("all")


if __name__ == "__main__":
    main()

