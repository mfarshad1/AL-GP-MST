#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# Stage-2 GP Umb 3D  (PIP / MD-poly version, plotting aligned to toy script)
#
# Results:
#   - Oracle 3D scatter + diag slice (2D/3D) + face slice (2D/3D)
#   - Oracle ΔW3 3D scatter
#   - Optional full-grid LSQR diagnostic scatter
#   - Optional per-step LSQR paid-vs-full-grid parity
#   - Per-step: student μ/σ 3D scatter (with overlays)
#   - Per-step: μ/σ diag slice (2D/3D), μ face slice (2D/3D)
#   - Per-step: absolute error |pred - oracle| scatter + diag slice (2D/3D) + face slice (2D/3D)
#   - Per-step: extracted ΔW3 on diag and face (2D/3D)
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
import heapq
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import tensorflow as tf
import gpflow
from gpflow import utilities as gpf_util

from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.linalg import lsqr
from matplotlib.ticker import MaxNLocator

# =============================================================================
# CLI options
# =============================================================================
DEFAULT_START_MODE    = "cold"         # warm/cold
DEFAULT_TARGET_MODE   = "umb"          # umb/residual
DEFAULT_SYMM_MODE     = "sorted"         # sorted/invariants/none
DEFAULT_KERNEL_MODE   = "plain"        # structured/plain
DEFAULT_LABEL_SOURCE  = "oracle_paid"  # lsqr|oracle_paid|oracle_queried

RUN_TAG = "UmbDirect"

def parse_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--start_mode", choices=["warm", "cold"], default=DEFAULT_START_MODE,
                   help="warm=copies previous GP params; cold=rebuild from scratch each step")
    p.add_argument("--target_mode", choices=["umb", "residual"], default=DEFAULT_TARGET_MODE,
                   help="umb=GP trains on anchored Umb; residual=GP trains on Umb-(W2sum-W2_anchor)")
    p.add_argument("--symmetry_mode", choices=["sorted", "invariants", "none"], default=DEFAULT_SYMM_MODE,
                   help="Permutation-invariant inputs (PIP spirit): sorted or symmetric invariants")
    p.add_argument("--kernel_mode", choices=["structured", "plain"], default=DEFAULT_KERNEL_MODE,
                   help="structured=(1D+1D+1D)+3D, plain=use KERNEL_SPEC")
    p.add_argument("--label_source", choices=["lsqr", "oracle_paid", "oracle_queried"],
                   default=DEFAULT_LABEL_SOURCE,
                   help=("Label source for GP training: "
                         "lsqr=LSQR from forces on paid graph; "
                         "oracle_paid=direct oracle energies on ALL paid nodes; "
                         "oracle_queried=direct oracle energies on QUERIED nodes only "
                         "(still pays path nodes, but excludes them from training)."))
    return p.parse_args()

# =============================================================================
# MD ORACLE PARAMETERS
# =============================================================================
GRAFT = "0.30"  # "0.15" or "0.30"

D12_MIN, D12_MAX = 6.04, 11.0
D13_MIN, D13_MAX = 6.04, 11.0
D23_MIN, D23_MAX = 6.04, 11.0

N3 = 50
PLOT_MAX = 100000000
SEED = 12345

DO_FULLGRID_LSQR_DIAGNOSTIC = True
PLOT_FULLGRID_LSQR_DIAGNOSTIC = False

LSQR_PRINT_EQUATION_STATS = True
DW_CLIP = 40.0

PARAMS = {
    "0.15": {
        "dimer": {
            "k":  1.525067170652243,
            "x0": 8.363446322328578,
            "C": [-1.851203176386485, 2.658258259143189e-01, -2.640932576942780e-02,
                  1.349865646591411e-03, -3.729456173796494e-05, 4.265841953601893e-07],
            "rin": 10.0, "rout": 12.0,
        },
        "trimer": {
            "k":  5.039639114081719e-02,
            "x0": 4.999985984055976e+01,
            "C": [ 5.606910952220617e+02, -1.714059495393181e+03, -1.036597547189558e+00,
                  -2.465375325002020e+03,  9.698122318213179e+02,  9.011895158565868e+02,
                  -7.262392333263364e+01,  7.398735000575322e+02, -5.287031738300126e+02,
                  -5.878008118858911e+02,  8.285603106331479e+00,  1.427961679585090e+02,
                  -6.798268695275270e+01, -2.255193567829562e+02,  9.115792931264546e+01],
            "Ri": 6.0, "Ro": 10.0,
        },
    },
    "0.30": {
        "dimer": {
            "k":  1.426598250344009,
            "x0": 8.189162153344407,
            "C": [ 2.914781927874392e+00, -1.009595164815141e+00,  1.870210969460044e-01,
                 -2.030955490318639e-02,  1.239669836304152e-03, -4.171931659803831e-05,
                  6.155231317677564e-07],
            "rin": 10.0, "rout": 12.0,
        },
        "trimer": {
            "k":  2.446835361961843e-01,
            "x0": 2.118153650543119e+01,
            "C": [ 6.674892424965560e+00, -7.684095412814079e+00,  2.393000139152252e+00,
                  1.001282066575281e+00,  2.701725383910903e+00, -7.018160213524727e-01,
                 -3.928487047117967e-01, -7.150042936617682e-02,  9.622682681857697e-02,
                  5.045340594110370e-02, -1.027943320237391e-03, -6.432621099307617e-03,
                  2.111046653110089e-02,  6.310019710739115e-03, -1.157370931056368e-03],
            "Ri": 6.0, "Ro": 10.0,
        },
    },
}

# =============================================================================
# ORACLE FUNCTIONS
# =============================================================================
def switch_poly(d, rin=10.0, rout=12.0):
    d = np.asarray(d, float)
    x = (d - rin) / (rout - rin)
    return np.where(x < 0.0, 1.0,
           np.where(x >= 1.0, 0.0, 1.0 + 2.0*(2.0*x - 3.0)*(x**2)))

def switch_cos2(d, Ri, Ro):
    d = np.asarray(d, float)
    t = (d - Ri) / (Ro - Ri)
    return np.where(
        t < 0.0, 1.0,
        np.where(t >= 1.0, 0.0, np.cos(0.5*np.pi*t)**2)
    )

def y_dimer_basis(d, k, x0):
    return np.exp(-k*(np.asarray(d, float) - x0))

def y_trimer_basis(d, k, x0):
    d = np.maximum(np.asarray(d, float), 1e-12)
    return np.exp(-k*(d - x0))/d

def W2(d, p2):
    d = np.asarray(d, float)
    C = np.asarray(p2["C"], float)
    y = y_dimer_basis(d, p2["k"], p2["x0"])
    acc = y.copy()
    poly = np.zeros_like(d, float)
    for ci in C:
        poly += ci * acc
        acc *= y
    return switch_poly(d, p2["rin"], p2["rout"]) * poly

def deltaW3_raw(d12, d13, d23, p3):
    C = np.asarray(p3["C"], float)
    if C.size != 15:
        raise ValueError("deltaW3 expects 15 coefficients.")
    y0 = y_trimer_basis(d12, p3["k"], p3["x0"])
    y1 = y_trimer_basis(d13, p3["k"], p3["x0"])
    y2 = y_trimer_basis(d23, p3["k"], p3["x0"])

    p0  = y0 + y1 + y2
    p1  = y0*y0 + y1*y1 + y2*y2
    p2m = y0*y1 + y0*y2 + y1*y2
    p3m = y0*y1*y2
    p4  = y0**3 + y1**3 + y2**3
    p5  = (y0*y2*y2 + y0*y1*y1 + y1*y2*y2 + y0*y0*y1 + y1*y1*y2 + y0*y0*y2)
    p6  = y0**4 + y1**4 + y2**4
    p7  = y0*y0*y1*y2 + y0*y1*y2*y2 + y0*y1*y1*y2
    p8  = (y0**3*y2 + y0*y2**3 + y0**3*y1 + y0*y1**3 + y1**3*y2 + y1*y2**3)
    p9  = y0*y0*y2*y2 + y1*y1*y2*y2 + y0*y0*y1*y1
    p10 = y0*y0*y1*y2*y2 + y0*y1*y1*y2*y2 + y0*y0*y1*y1*y2
    p11 = (y1**4*y2 + y0**4*y2 + y0*y2**4 + y0**4*y1 + y1*y2**4 + y0*y1**4)
    p12 = y0**5 + y1**5 + y2**5
    p13 = y0*y1**3*y2 + y0*y1*y2**3 + y0**3*y1*y2
    p14 = (y1**3*y2**2 + y1**2*y2**3 + y0**2*y1**3 +
           y0**2*y2**3 + y0**3*y1**2 + y0**3*y2**2)

    return (C[0]*p0 + C[1]*p1 + C[2]*p2m + C[3]*p3m + C[4]*p4 +
            C[5]*p5 + C[6]*p6 + C[7]*p7 + C[8]*p8 + C[9]*p9 +
            C[10]*p10 + C[11]*p11 + C[12]*p12 + C[13]*p13 + C[14]*p14)

def sf_sum(d12, d13, d23, p3):
    s12 = switch_cos2(d12, p3["Ri"], p3["Ro"])
    s13 = switch_cos2(d13, p3["Ri"], p3["Ro"])
    s23 = switch_cos2(d23, p3["Ri"], p3["Ro"])
    return s12*s13 + s12*s23 + s13*s23

def Umb(d12, d13, d23, p2, p3):
    W2sum = W2(d12, p2) + W2(d13, p2) + W2(d23, p2)
    Vraw  = deltaW3_raw(d12, d13, d23, p3)
    sf    = sf_sum(d12, d13, d23, p3)
    return W2sum + sf*Vraw, sf*Vraw, Vraw

def triangle_mask(d12, d13, d23, eps=1e-12):
    d12 = np.asarray(d12, float)
    d13 = np.asarray(d13, float)
    d23 = np.asarray(d23, float)
    return (
        (d12 <= d13 + d23 + eps) &
        (d13 <= d12 + d23 + eps) &
        (d23 <= d12 + d13 + eps) &
        (d12 >= np.abs(d13 - d23) - eps) &
        (d13 >= np.abs(d12 - d23) - eps) &
        (d23 >= np.abs(d12 - d13) - eps)
    )

def W2sum_pairs(d12, d13, d23, p2):
    return W2(d12, p2) + W2(d13, p2) + W2(d23, p2)

# =============================================================================
# Permutation-invariant feature map (PIP spirit)
#   Input P is always in "physical grid order" (d13, d23, d12).
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
    s2 = D[..., 0]*D[..., 1] + D[..., 0]*D[..., 2] + D[..., 1]*D[..., 2]
    s3 = D[..., 0]*D[..., 1]*D[..., 2]
    return np.stack([s1, s2, s3], axis=-1)

# =============================================================================
# STYLE / FLAGS
#   These are kept matched to the toy plotting script as closely as possible.
# =============================================================================
LAB_X = r"$d_{13}\,(\sigma)$"
LAB_Y = r"$d_{23}\,(\sigma)$"
LAB_Z = r"$d_{12}\,(\sigma)$"
LAB_C   = r"$U_{\mathrm{mb}}\,(k_{\mathrm{B}}T)$"
LAB_DW3 = r"$\Delta W_3\,(k_{\mathrm{B}}T)$"
LAB_ABSERR = r"$|U_{\mathrm{pred}}-U_{\mathrm{oracle}}|\,(k_{\mathrm{B}}T)$"
ABSERR_CMAP = "magma"

XLIM = (D13_MIN, D13_MAX)
YLIM = (D23_MIN, D23_MAX)
ZLIM = (D12_MIN, D12_MAX)

USE_FIXED_CLIM = False
CLIM_MIN = -30.0
CLIM_MAX = 20.0

fig_scale = 2
FIGSIZE_3D      = (6.0 * fig_scale, 4.5 * fig_scale)
FIGSIZE_RMSE    = (4.5 * fig_scale, 3.6 * fig_scale)
FIGSIZE_SLICE2D = (4.8 * fig_scale, 3.9 * fig_scale)
FIGSIZE_PARITY  = (4.5 * fig_scale, 4.2 * fig_scale)

AXIS_LABEL_FZ = 36
TICK_LABEL_FZ = 30
CBAR_FZ = 36
CBAR_TICK_FZ = 30
LEGEND_FZ = 30
SHOW_TITLES = False

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
REVERSE_FACE_X_2D = False
REVERSE_FACE_Y_2D = False
REVERSE_FACE_X_3D = False
REVERSE_FACE_Y_3D = False

SHOW_CAND_OVERLAY = True
KNN_K = 100
CAND_SHOW_N = 1000
CAND_CMAP = "coolwarm"
CAND_MARKER_SIZE = 28.0
CAND_ALPHA = 0.95
CAND_EDGE_LW = 0.25
CAND_CBAR_PAD = 0.012
CAND_CBAR_H = 0.03
CAND_CBAR_LABEL = r"Candidate score"

SLICE_N12 = 90
SLICE_N13 = 90
SLICE_VIEW_ELEV = VIEW_ELEV
SLICE_VIEW_AZIM = VIEW_AZIM
FACE_N13 = SLICE_N13
FACE_N23 = SLICE_N13

# GP
KERNEL_SPEC = "m52+rq*se"
ARD = True
INIT_ITERS = 10000
AL_ITERS = 50
AL_BATCH = 10

VAL_FRAC = 0.15
VAL_MIN = 2000
VAL_MAX = 12000

JITTER_DEF      = 5e-4
LIK_NOISE0      = 5e-3
LIK_NOISE_RETRY = 5e-2
WHITE_NUGGET    = 5e-4

PRUNE_NEAR_DUPLICATES = True
PRUNE_EPS = 1e-6

MIN_SEP_MULT = 0.7
ADAPT_MINSEP_DECAY = 0.92
POOL_MIN_FRACTION = 0.01

ACCEPT_TOL = 0.000

STEP0_LSQR_GRAPH_MODE = "grid26"
MAIN_LSQR_GRAPH_MODE  = "grid26"
LSQR_GRAPH_MODE = MAIN_LSQR_GRAPH_MODE

INIT_LSQR_KNN_K = 10
INIT_LSQR_KNN_MAX_DIST = None
LSQR_KNN_K = INIT_LSQR_KNN_K
LSQR_KNN_MAX_DIST = INIT_LSQR_KNN_MAX_DIST

INIT_EDGE_COUNT       = 12
INIT_FACE_DIAG_COUNT  = 12
INIT_SPACE_DIAG_COUNT = 4

# INIT_EDGE_COUNT       = int(os.environ.get("INIT_EDGE_COUNT", 1))
# INIT_FACE_DIAG_COUNT  = int(os.environ.get("INIT_FACE_DIAG_COUNT", 0))
# INIT_SPACE_DIAG_COUNT = int(os.environ.get("INIT_SPACE_DIAG_COUNT", 0))

FIG_DIR = None

MARKER_SCALE = 8.0
MS_TRAIN  = 18 * MARKER_SCALE
MS_VAL    = 18 * MARKER_SCALE
MS_ANCHOR = 80 * MARKER_SCALE
LINE_MST_WIDTH = 3

TRAIN_FACE = "c"
TRAIN_EDGE = "black"
VAL_EDGE = "red"
VAL_FACE = "none"
TRAIN_MARKER = "o"
VAL_MARKER = "^"
MST_COLOR = "black"
MST_Q_COLOR = "tab:orange"

OVERLAY_ORDER_MODE = "mst_last"
Z_VAL, Z_TRAIN, Z_KNN, Z_MST, Z_ANCHOR = 1, 2, 3, 4, 5

FORCE_TEXT2D_ZLABEL = True
ZLABEL_2D_X = -0.12
ZLABEL_2D_Y = 0.52

USE_LATEX = True
if USE_LATEX:
    from matplotlib import rc, rcParams
    path_add = os.path.expanduser("~/.TinyTeX/bin/x86_64-linux")
    os.environ["PATH"] = path_add + ":" + os.environ.get("PATH", "")
    os.environ.pop("PERL5LIB", None)
    rc("text", usetex=True)
    rcParams["text.latex.preamble"] = (
        r"\usepackage{amsmath}\usepackage[T1]{fontenc}"
        r"\usepackage{lmodern}\usepackage[strings]{underscore}"
    )
    rc("font", family="serif")
else:
    mpl.rcParams["text.usetex"] = False

# =============================================================================
# Utilities
# =============================================================================
def savefig(fig, name, extra_artists=None):
    global FIG_DIR
    if FIG_DIR is None:
        FIG_DIR = Path("./figs_UNSET")
        FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / f"{name}.pdf"
    prev_usetex = mpl.rcParams.get("text.usetex", False)
    try:
        fig.canvas.draw()
        fig.savefig(path, bbox_inches="tight", pad_inches=0.25,
                    bbox_extra_artists=extra_artists)
    except Exception as e:
        print(f"[WARN] LaTeX draw failed: {e}\n Falling back to mathtext for this figure only.")
        try:
            mpl.rcParams["text.usetex"] = False
            fig.canvas.draw()
        except Exception as e2:
            print(f"[WARN] mathtext draw also failed: {e2}")
        finally:
            fig.savefig(path, bbox_inches="tight", pad_inches=0.25,
                        bbox_extra_artists=extra_artists)
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

def y2z(y, m, s): return (np.asarray(y, float) - m) / s
def z2y(z, m, s): return np.asarray(z, float) * s + m

def unique_rows_idx(X):
    _, idx = np.unique(np.round(X, 14), axis=0, return_index=True)
    return np.sort(idx)

def rmse(y, yp):
    y = np.ravel(y); yp = np.ravel(yp)
    m = np.isfinite(y) & np.isfinite(yp)
    return float("nan") if (not np.any(m)) else float(np.sqrt(np.mean((y[m]-yp[m])**2)))

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
    X = X[m]; y = y[m]
    if X.shape[0] == 0:
        return X, y
    idx = unique_rows_idx(X)
    X = X[idx]; y = y[idx]
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
        v0 = float(np.nanmin(z[m])); v1 = float(np.nanmax(z[m]))
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
        if t in ("m32","matern32"):
            return gpflow.kernels.Matern32(lengthscales=[1,1,1] if ard else 1.0)
        if t in ("m52","matern52"):
            return gpflow.kernels.Matern52(lengthscales=[1,1,1] if ard else 1.0)
        if t in ("se","rbf"):
            return gpflow.kernels.SquaredExponential(lengthscales=[1,1,1] if ard else 1.0)
        if t in ("rq","ratquad"):
            return gpflow.kernels.RationalQuadratic(lengthscales=[1,1,1] if ard else 1.0, alpha=1.0)
        if t in ("lin","linear"):
            return gpflow.kernels.Linear()
        return gpflow.kernels.Matern52(lengthscales=[1,1,1] if ard else 1.0)

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
                try: k.variance.assign(var0)
                except Exception: pass
            if hasattr(k, "lengthscales"):
                ls0 = np.clip(np.std(X, axis=0), 0.15, 3.0)
                try: k.lengthscales.assign(ls0 if ard else float(np.mean(ls0)))
                except Exception: pass
            if hasattr(k, "alpha"):
                try: k.alpha.assign(1.0)
                except Exception: pass

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
        k3s = gpflow.kernels.Matern52(active_dims=[0,1,2], lengthscales=[1.0,1.0,1.0])
        k3l = gpflow.kernels.RationalQuadratic(active_dims=[0,1,2],
                                               lengthscales=[1.0,1.0,1.0],
                                               variance=0.5, alpha=1.0)
    else:
        k3s = gpflow.kernels.Matern52(active_dims=[0,1,2], lengthscales=1.0)
        k3l = gpflow.kernels.RationalQuadratic(active_dims=[0,1,2],
                                               lengthscales=1.0,
                                               variance=0.5, alpha=1.0)
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
        m.training_loss, m.trainable_variables,
        options={"maxiter": int(iters), "disp": False}
    )

def warm_start_params(m_new, m_prev):
    if m_prev is None:
        return
    try:
        pd_prev = gpf_util.parameter_dict(m_prev)
        pd_new  = gpf_util.parameter_dict(m_new)
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
    retry_noises  = [LIK_NOISE0, LIK_NOISE_RETRY, 5e-2, 1e-1]
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

            m = gpflow.models.GPR((Xtf, ztf), kernel=K,
                                  mean_function=gpflow.mean_functions.Zero())
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
            mu = mu.numpy(); var = var.numpy()
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
def _apply_axis_mapping(Pxyz, swap_xy=None):
    use_swap = SWAP_XY if swap_xy is None else bool(swap_xy)
    return Pxyz[:, [1, 0, 2]] if use_swap else Pxyz

def _mapped_labels(swap_xy=None):
    use_swap = SWAP_XY if swap_xy is None else bool(swap_xy)
    return (LAB_Y, LAB_X, LAB_Z) if use_swap else (LAB_X, LAB_Y, LAB_Z)

def _mapped_limits(swap_xy=None):
    use_swap = SWAP_XY if swap_xy is None else bool(swap_xy)
    return (YLIM, XLIM, ZLIM) if use_swap else (XLIM, YLIM, ZLIM)

def _apply_axis_inversions(ax, swap_xy=None):
    use_swap = SWAP_XY if swap_xy is None else bool(swap_xy)
    if REVERSE_X:
        ax.invert_yaxis() if use_swap else ax.invert_xaxis()
    if REVERSE_Y:
        ax.invert_xaxis() if use_swap else ax.invert_yaxis()
    if REVERSE_Z:
        ax.invert_zaxis()

def _add_top_colorbar(fig, ax, mappable, label):
    pos = ax.get_position()
    cax = fig.add_axes([pos.x0, pos.y1 + CAND_CBAR_PAD, pos.width, CAND_CBAR_H])
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.xaxis.set_label_position("top")
    cb.set_label(label, fontsize=CBAR_FZ, labelpad=10)
    cb.ax.tick_params(labelsize=CBAR_TICK_FZ, pad=6)
    return cb

def _pin_3d_zlabel_2d(ax, label, fontsize, x=ZLABEL_2D_X, y=ZLABEL_2D_Y):
    try:
        ax.set_zlabel("")
    except Exception:
        pass
    try:
        art = ax.text2D(
            float(x), float(y), label,
            transform=ax.transAxes,
            fontsize=float(fontsize),
            rotation=90,
            va="center", ha="right",
            clip_on=False
        )
        return art
    except Exception:
        return None

def plot_scatter3d_color(Pxyz, C, clabel, name, title,
                         overlay=None, cmap="viridis", norm=None, cbar_ticks=None,
                         scatter_swap_xy=False):
    fig = plt.figure(figsize=FIGSIZE_3D)
    ax = fig.add_subplot(111, projection="3d")
    try: ax.computed_zorder = False
    except Exception: pass
    if ORTHOGRAPHIC:
        try: ax.set_proj_type("ortho")
        except Exception: pass

    Q = _apply_axis_mapping(Pxyz, swap_xy=scatter_swap_xy)
    lx, ly, lz = _mapped_labels(swap_xy=scatter_swap_xy)
    xlim, ylim, zlim = _mapped_limits(swap_xy=scatter_swap_xy)

    sc = ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2],
                    c=C, s=8, cmap=cmap, norm=norm,
                    linewidths=0, alpha=0.85,
                    depthshade=False, zorder=0)

    ax.set_xlabel(lx, fontsize=AXIS_LABEL_FZ, labelpad=20)
    ax.set_ylabel(ly, fontsize=AXIS_LABEL_FZ, labelpad=20)

    z_art = None
    if FORCE_TEXT2D_ZLABEL:
        z_art = _pin_3d_zlabel_2d(ax, lz, AXIS_LABEL_FZ, x=-0.05, y=ZLABEL_2D_Y)
    else:
        ax.set_zlabel(lz, fontsize=AXIS_LABEL_FZ, labelpad=10)

    if SHOW_TITLES:
        ax.set_title(title, fontsize=LEGEND_FZ + 2, pad=10)

    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
    try:
        ax.set_box_aspect(((xlim[1]-xlim[0]), (ylim[1]-ylim[0]), Z_BOX_SCALE*(zlim[1]-zlim[0])))
    except Exception:
        pass

    ax.view_init(elev=float(VIEW_ELEV), azim=float(VIEW_AZIM))
    _apply_axis_inversions(ax, swap_xy=scatter_swap_xy)
    ax.tick_params(axis="x", labelsize=TICK_LABEL_FZ, pad=3)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FZ, pad=3)
    ax.tick_params(axis="z", labelsize=TICK_LABEL_FZ, pad=3)

    cb = fig.colorbar(sc, ax=ax, shrink=0.78, pad=0.05)
    cb.set_label(clabel, fontsize=CBAR_FZ, labelpad=10)
    cb.ax.tick_params(labelsize=CBAR_TICK_FZ, pad=6)
    if cbar_ticks is not None:
        cb.set_ticks(cbar_ticks)

    extra = overlay(ax) if callable(overlay) else None
    if isinstance(extra, dict) and extra.get("knn_mappable", None) is not None:
        _add_top_colorbar(fig, ax, extra["knn_mappable"], extra.get("knn_label", CAND_CBAR_LABEL))

    extra_artists = [z_art] if z_art is not None else None
    savefig(fig, name, extra_artists=extra_artists)

# =============================================================================
# Slice grids + plotting
# =============================================================================
def build_diag_slice_grid():
    d12s = np.linspace(D12_MIN, D12_MAX, SLICE_N12)
    d13s = np.linspace(D13_MIN, D13_MAX, SLICE_N13)
    D12s, D13s = np.meshgrid(d12s, d13s, indexing="ij")
    D23s = D13s.copy()
    m = triangle_mask(D12s, D13s, D23s)
    return d12s, d13s, D12s, D13s, D23s, m

def build_face_slice_grid():
    d13f = np.linspace(D13_MIN, D13_MAX, FACE_N13)
    d23f = np.linspace(D23_MIN, D23_MAX, FACE_N23)
    D23f, D13f = np.meshgrid(d23f, d13f, indexing="ij")
    D12f = np.full_like(D13f, float(D12_MIN))
    m = triangle_mask(D12f, D13f, D23f)
    return d13f, d23f, D13f, D23f, D12f, m

def plot_slice_2d_heat(D12s, D13s, Z, m, name, title, clabel, cmap="viridis", norm=None):
    Zm = np.where(m, Z, np.nan)
    x = D13s[0, :]
    y = D12s[:, 0]
    fig, ax = plt.subplots(figsize=FIGSIZE_SLICE2D)
    pc = ax.pcolormesh(x, y, Zm, shading="auto", cmap=cmap, norm=norm)
    ax.set_xlabel(r"$d_{13}=d_{23}\,(\sigma)$", fontsize=AXIS_LABEL_FZ)
    ax.set_ylabel(r"$d_{12}\,(\sigma)$", fontsize=AXIS_LABEL_FZ)

    ax.set_xticks(np.arange(np.floor(x.min()), np.ceil(x.max()) + 1e-9, 1.0))
    ax.set_yticks(np.arange(np.floor(y.min()), np.ceil(y.max()) + 1e-9, 1.0))

    ax.tick_params(axis="both", labelsize=TICK_LABEL_FZ, pad=6)
    if REVERSE_DIAG_X_2D: ax.invert_xaxis()
    if REVERSE_DIAG_Y_2D: ax.invert_yaxis()
    if SHOW_TITLES: ax.set_title(title, fontsize=LEGEND_FZ + 2, pad=10)
    cb = fig.colorbar(pc, ax=ax, pad=0.02)
    cb.set_label(clabel, fontsize=CBAR_FZ, labelpad=10)
    cb.ax.tick_params(labelsize=CBAR_TICK_FZ, pad=6)
    savefig(fig, name)

def plot_slice_3d_surface(D12s, D13s, Z, m, name, title, clabel, cmap="viridis", norm=None):
    Zm = np.where(m, Z, np.nan)
    fig = plt.figure(figsize=FIGSIZE_3D)
    ax = fig.add_subplot(111, projection="3d")
    ax.tick_params(axis="z", pad=50)
    for lab in ax.zaxis.get_ticklabels():
        lab.set_horizontalalignment("right")
        lab.set_verticalalignment("center")
        lab.set_rotation(0)
    try: ax.computed_zorder = False
    except Exception: pass
    if ORTHOGRAPHIC:
        try: ax.set_proj_type("ortho")
        except Exception: pass
    surf = ax.plot_surface(D13s, D12s, Zm, cmap=cmap, norm=norm, linewidth=0, antialiased=True)
    ax.set_xlabel(r"$d_{13}=d_{23}\,(\sigma)$", fontsize=AXIS_LABEL_FZ, labelpad=20)
    ax.set_ylabel(r"$d_{12}\,(\sigma)$", fontsize=AXIS_LABEL_FZ, labelpad=20)

    z_art = None
    if FORCE_TEXT2D_ZLABEL:
        z_art = _pin_3d_zlabel_2d(ax, clabel, AXIS_LABEL_FZ)
    else:
        ax.set_zlabel(clabel, fontsize=AXIS_LABEL_FZ, labelpad=16)

    if SHOW_TITLES: ax.set_title(title, fontsize=LEGEND_FZ + 2, pad=10)
    ax.set_xlim(D13_MIN, D13_MAX); ax.set_ylim(D12_MIN, D12_MAX)
    if REVERSE_DIAG_X_3D: ax.invert_xaxis()
    if REVERSE_DIAG_Y_3D: ax.invert_yaxis()
    try:
        ax.set_box_aspect(((D13_MAX - D13_MIN), (D12_MAX - D12_MIN), Z_BOX_SCALE * (D12_MAX - D12_MIN)))
    except Exception:
        pass
    ax.view_init(elev=float(SLICE_VIEW_ELEV), azim=float(SLICE_VIEW_AZIM))
    ax.tick_params(axis="x", labelsize=TICK_LABEL_FZ, pad=3)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FZ, pad=3)
    ax.tick_params(axis="z", labelsize=TICK_LABEL_FZ, pad=3)
    cb = fig.colorbar(surf, ax=ax, shrink=0.78, pad=0.05)
    cb.set_label(clabel, fontsize=CBAR_FZ, labelpad=10)
    cb.ax.tick_params(labelsize=CBAR_TICK_FZ, pad=6)
    extra_artists = [z_art] if z_art is not None else None
    savefig(fig, name, extra_artists=extra_artists)

def plot_face_2d_heat(d13f, d23f, Z, m, name, title, clabel, cmap="viridis", norm=None):
    Zm = np.where(m, Z, np.nan)
    fig, ax = plt.subplots(figsize=FIGSIZE_SLICE2D)
    pc = ax.pcolormesh(d13f, d23f, Zm, shading="auto", cmap=cmap, norm=norm)
    ax.set_xlabel(r"$d_{13}\,(\sigma)$", fontsize=AXIS_LABEL_FZ)
    ax.set_ylabel(r"$d_{23}\,(\sigma)$", fontsize=AXIS_LABEL_FZ)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FZ, pad=6)
    if REVERSE_FACE_X_2D: ax.invert_xaxis()
    if REVERSE_FACE_Y_2D: ax.invert_yaxis()
    if SHOW_TITLES: ax.set_title(title, fontsize=LEGEND_FZ + 2, pad=10)
    cb = fig.colorbar(pc, ax=ax, pad=0.02)
    cb.set_label(clabel, fontsize=CBAR_FZ, labelpad=10)
    cb.ax.tick_params(labelsize=CBAR_TICK_FZ, pad=6)
    ax.set_xticks(np.arange(np.floor(d13f.min()), np.ceil(d13f.max()) + 1e-9, 1.0))
    ax.set_yticks(np.arange(np.floor(d23f.min()), np.ceil(d23f.max()) + 1e-9, 1.0))
    savefig(fig, name)

def plot_face_3d_surface(D13f, D23f, Z, m, name, title, clabel, cmap="viridis", norm=None):
    Zm = np.where(m, Z, np.nan)
    fig = plt.figure(figsize=FIGSIZE_3D)
    ax = fig.add_subplot(111, projection="3d")
    ax.tick_params(axis="z", pad=50)
    for lab in ax.zaxis.get_ticklabels():
        lab.set_horizontalalignment("right")
        lab.set_verticalalignment("center")
        lab.set_rotation(0)
    try: ax.computed_zorder = False
    except Exception: pass
    if ORTHOGRAPHIC:
        try: ax.set_proj_type("ortho")
        except Exception: pass
    surf = ax.plot_surface(D13f, D23f, Zm, cmap=cmap, norm=norm, linewidth=0, antialiased=True)
    ax.set_xlabel(r"$d_{13}\,(\sigma)$", fontsize=AXIS_LABEL_FZ, labelpad=20)
    ax.set_ylabel(r"$d_{23}\,(\sigma)$", fontsize=AXIS_LABEL_FZ, labelpad=20)

    z_art = None
    if FORCE_TEXT2D_ZLABEL:
        z_art = _pin_3d_zlabel_2d(ax, clabel, AXIS_LABEL_FZ)
    else:
        ax.set_zlabel(clabel, fontsize=AXIS_LABEL_FZ, labelpad=16)

    if SHOW_TITLES: ax.set_title(title, fontsize=LEGEND_FZ + 2, pad=10)
    ax.set_xlim(D13_MIN, D13_MAX); ax.set_ylim(D23_MIN, D23_MAX)
    if REVERSE_FACE_X_3D: ax.invert_xaxis()
    if REVERSE_FACE_Y_3D: ax.invert_yaxis()
    try:
        ax.set_box_aspect(((D13_MAX - D13_MIN), (D23_MAX - D23_MIN), Z_BOX_SCALE * (D23_MAX - D23_MIN)))
    except Exception:
        pass
    ax.view_init(elev=float(SLICE_VIEW_ELEV), azim=float(SLICE_VIEW_AZIM))
    ax.tick_params(axis="x", labelsize=TICK_LABEL_FZ, pad=3)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FZ, pad=3)
    ax.tick_params(axis="z", labelsize=TICK_LABEL_FZ, pad=3)
    cb = fig.colorbar(surf, ax=ax, shrink=0.78, pad=0.05)
    cb.set_label(clabel, fontsize=CBAR_FZ, labelpad=10)
    cb.ax.tick_params(labelsize=CBAR_TICK_FZ, pad=6)
    extra_artists = [z_art] if z_art is not None else None
    savefig(fig, name, extra_artists=extra_artists)

# =============================================================================
# GP predict on slice/face (uses symmetry_mode feature map)
# =============================================================================
def gp_predict_on_slice(model, xmu, xsd, ymu, ysd, D12s, D13s, symmetry_mode):
    P = np.c_[D13s.ravel(), D13s.ravel(), D12s.ravel()]
    Pf = _features_from_P(P, symmetry_mode=symmetry_mode)
    Xs = standardize_apply(Pf, xmu, xsd)
    mu_z, std_z = predict_mu_std_z(model, Xs)
    mu_y = z2y(mu_z, ymu, ysd)
    std_y = std_z * ysd
    return mu_y.reshape(D12s.shape), std_y.reshape(D12s.shape)

def gp_predict_on_face(model, xmu, xsd, ymu, ysd, D13f, D23f, D12f, symmetry_mode):
    P = np.c_[D13f.ravel(), D23f.ravel(), D12f.ravel()]
    Pf = _features_from_P(P, symmetry_mode=symmetry_mode)
    Xs = standardize_apply(Pf, xmu, xsd)
    mu_z, std_z = predict_mu_std_z(model, Xs)
    mu_y = z2y(mu_z, ymu, ysd)
    std_y = std_z * ysd
    return mu_y.reshape(D13f.shape), std_y.reshape(D13f.shape)

def oracle_on_slice(D12s, D13s, p2, p3):
    D23s = D13s
    U, W3sw, W3raw = Umb(D12s, D13s, D23s, p2, p3)
    return U, W3sw, W3raw

def oracle_on_face(D13f, D23f, D12f, p2, p3):
    U, W3sw, W3raw = Umb(D12f, D13f, D23f, p2, p3)
    return U, W3sw, W3raw

# =============================================================================
# Grid indexing / neighbors
# =============================================================================
def ijk_to_g(i12, i13, i23, n12, n13, n23):
    return (int(i12) * n13 + int(i13)) * n23 + int(i23)

def g_to_ijk(g, n12, n13, n23):
    g = int(g)
    i12, rem = divmod(g, n13 * n23)
    i13, i23 = divmod(rem, n23)
    return int(i12), int(i13), int(i23)

def neighbors_26(g, n12, n13, n23):
    i12, i13, i23 = g_to_ijk(g, n12, n13, n23)
    out = []
    for di12 in (-1, 0, 1):
        for di13 in (-1, 0, 1):
            for di23 in (-1, 0, 1):
                if di12 == 0 and di13 == 0 and di23 == 0:
                    continue
                j12 = i12 + di12
                j13 = i13 + di13
                j23 = i23 + di23
                if 0 <= j12 < n12 and 0 <= j13 < n13 and 0 <= j23 < n23:
                    out.append(ijk_to_g(j12, j13, j23, n12, n13, n23))
    return out

# =============================================================================
# Strict connectivity (path paying)
# =============================================================================
def manhattan_walk_add_3d(train_set, start_g, target_g, n12, n13, n23, allowed_mask_flat):
    start_g = int(start_g); target_g = int(target_g)
    if (not allowed_mask_flat[start_g]) or (not allowed_mask_flat[target_g]):
        return []
    i12, i13, i23 = g_to_ijk(start_g, n12, n13, n23)
    j12, j13, j23 = g_to_ijk(target_g, n12, n13, n23)

    added = []
    train_set.add(start_g); added.append(start_g)

    if i12 != j12:
        step = 1 if j12 > i12 else -1
        while i12 != j12:
            i12 += step
            g = ijk_to_g(i12, i13, i23, n12, n13, n23)
            if allowed_mask_flat[g]:
                train_set.add(g); added.append(g)

    if i13 != j13:
        step = 1 if j13 > i13 else -1
        while i13 != j13:
            i13 += step
            g = ijk_to_g(i12, i13, i23, n12, n13, n23)
            if allowed_mask_flat[g]:
                train_set.add(g); added.append(g)

    if i23 != j23:
        step = 1 if j23 > i23 else -1
        while i23 != j23:
            i23 += step
            g = ijk_to_g(i12, i13, i23, n12, n13, n23)
            if allowed_mask_flat[g]:
                train_set.add(g); added.append(g)
    return added

def shortest_path_add_3d(train_set, start_g, target_g, P_full, n12, n13, n23, allowed_mask_flat):
    start_g = int(start_g); target_g = int(target_g)
    if start_g == target_g:
        if allowed_mask_flat[start_g]:
            train_set.add(start_g); return [start_g]
        return []
    if (not allowed_mask_flat[start_g]) or (not allowed_mask_flat[target_g]):
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
        for h in neighbors_26(g, n12, n13, n23):
            h = int(h)
            if not allowed_mask_flat[h]:
                continue
            step = float(np.linalg.norm(P_full[h] - P_full[g]))
            nd = dcur + step
            if nd < dist.get(h, np.inf):
                dist[h] = nd
                prev[h] = g
                heapq.heappush(pq, (nd, h))

    if target_g not in prev:
        return manhattan_walk_add_3d(train_set, start_g, target_g, n12, n13, n23, allowed_mask_flat)

    path = []
    g = target_g
    while g is not None:
        path.append(int(g))
        g = prev[g]
    path.reverse()

    for g in path[1:]:
        if allowed_mask_flat[int(g)]:
            train_set.add(int(g)); added.append(int(g))
    return added

# =============================================================================
# LSQR force integration helpers
# =============================================================================
def build_edges_from_nodes(nodes_set, n12, n13, n23, feas_mask_flat):
    nodes = np.array(sorted(set(map(int, nodes_set))), dtype=int)
    node_set = set(nodes.tolist())
    edges = set()
    for g in nodes:
        for h in neighbors_26(g, n12, n13, n23):
            if (h in node_set) and feas_mask_flat[h]:
                i, j = (g, h) if g < h else (h, g)
                edges.add((i, j))
    return list(edges)

def build_edges_axis1_from_nodes(nodes_set, n12, n13, n23, feas_mask_flat):
    nodes = np.array(sorted(set(map(int, nodes_set))), dtype=int)
    node_set = set(nodes.tolist())
    edges = set()
    for g in nodes:
        g = int(g)
        if not feas_mask_flat[g]:
            continue
        i12, i13, i23 = g_to_ijk(g, n12, n13, n23)
        for di12, di13, di23 in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            j12, j13, j23 = i12 + di12, i13 + di13, i23 + di23
            if 0 <= j12 < n12 and 0 <= j13 < n13 and 0 <= j23 < n23:
                h = ijk_to_g(j12, j13, j23, n12, n13, n23)
                if (h in node_set) and feas_mask_flat[h]:
                    a, b = (g, h) if g < h else (h, g)
                    edges.add((a, b))
    return list(edges)

def build_edges_knn_from_nodes(nodes_set, P_full, k=10, max_dist=None):
    nodes = np.array(sorted(set(map(int, nodes_set))), dtype=int)
    if nodes.size < 2:
        return []
    pts = P_full[nodes]
    tree = cKDTree(pts)
    k_use = int(min(max(1, k), nodes.size - 1))
    dists, idxs = tree.query(pts, k=k_use + 1)
    dists = np.atleast_2d(dists)
    idxs  = np.atleast_2d(idxs)

    edges = set()
    for a in range(nodes.size):
        ga = int(nodes[a])
        for t in range(1, k_use + 1):
            b = int(idxs[a, t])
            if b < 0 or b >= nodes.size:
                continue
            if (max_dist is not None) and (float(dists[a, t]) > float(max_dist)):
                continue
            gb = int(nodes[b])
            i, j = (ga, gb) if ga < gb else (gb, ga)
            if i != j:
                edges.add((i, j))
    return list(edges)

def lsqr_U_on_known_nodes_3d(known_nodes, anchor_g, P_full, Fx, Fy, Fz,
                             n12, n13, n23, feas_mask_flat,
                             atol=1e-10, btol=1e-10, iter_lim=8000,
                             graph_mode=None, knn_k=None, knn_max_dist=None,
                             extra_edges=None):
    known_nodes = np.array(sorted(set(map(int, known_nodes))), dtype=int)
    nk = known_nodes.size
    if nk == 0:
        return np.array([], float), known_nodes

    loc = {int(g): i for i, g in enumerate(known_nodes)}
    if int(anchor_g) not in loc:
        return np.full(nk, np.nan, float), known_nodes

    mode = (LSQR_GRAPH_MODE if graph_mode is None else str(graph_mode)).lower()

    if mode == "knn":
        k_use = int(knn_k if knn_k is not None else LSQR_KNN_K)
        r_use = knn_max_dist if knn_max_dist is not None else LSQR_KNN_MAX_DIST
        edges = build_edges_knn_from_nodes(known_nodes, P_full, k=k_use, max_dist=r_use)
    elif mode == "axis1":
        edges = build_edges_axis1_from_nodes(known_nodes, n12, n13, n23, feas_mask_flat)
    else:
        edges = build_edges_from_nodes(known_nodes, n12, n13, n23, feas_mask_flat)

    if extra_edges:
        es = set((int(a), int(b)) if int(a) < int(b) else (int(b), int(a)) for (a, b) in edges)
        for (a, b) in extra_edges:
            a = int(a); b = int(b)
            if a == b:
                continue
            if (a not in loc) or (b not in loc):
                continue
            if (not feas_mask_flat[a]) or (not feas_mask_flat[b]):
                continue
            i, j = (a, b) if a < b else (b, a)
            es.add((i, j))
        edges = list(es)

    rows, cols, data, b = [], [], [], []
    n_skip_nonfinite = 0
    n_skip_clip = 0

    for (ii, jj) in edges:
        ii = int(ii); jj = int(jj)
        Fx_i, Fy_i, Fz_i = Fx[ii], Fy[ii], Fz[ii]
        Fx_j, Fy_j, Fz_j = Fx[jj], Fy[jj], Fz[jj]
        if not (np.isfinite(Fx_i) and np.isfinite(Fy_i) and np.isfinite(Fz_i) and
                np.isfinite(Fx_j) and np.isfinite(Fy_j) and np.isfinite(Fz_j)):
            n_skip_nonfinite += 1
            continue

        Fx_mid = 0.5 * (Fx_i + Fx_j)
        Fy_mid = 0.5 * (Fy_i + Fy_j)
        Fz_mid = 0.5 * (Fz_i + Fz_j)

        dx = P_full[jj] - P_full[ii]
        dw = +(Fx_mid * dx[0] + Fy_mid * dx[1] + Fz_mid * dx[2])

        if (not np.isfinite(dw)) or (abs(dw) > float(DW_CLIP)):
            n_skip_clip += 1
            continue

        r = len(b)
        b.append(dw)
        rows.extend([r, r]); cols.extend([loc[ii], loc[jj]]); data.extend([-1.0, +1.0])

        r2 = len(b)
        w = np.sqrt(1e-3)
        b.append(0.0)
        rows.extend([r2, r2]); cols.extend([loc[ii], loc[jj]]); data.extend([-w, +w])

    M = len(b)
    if LSQR_PRINT_EQUATION_STATS:
        print(f"[LSQR build] mode={mode} nk={nk} edges_in={len(edges)} "
              f"M={M} skip_nonfinite={n_skip_nonfinite} skip_clip={n_skip_clip}")

    if M == 0:
        return np.full(nk, np.nan, float), known_nodes

    Asub = coo_matrix((np.asarray(data, float),
                       (np.asarray(rows, int), np.asarray(cols, int))),
                      shape=(M, nk)).tocsr()
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

# =============================================================================
# Step-0 geometry builders (unchanged behavior)
# =============================================================================
def cube_edge_nodes(feasible_g, n12, n13, n23):
    out = []
    for g in feasible_g:
        i12, i13, i23 = g_to_ijk(int(g), n12, n13, n23)
        b12 = (i12 == 0) or (i12 == n12 - 1)
        b13 = (i13 == 0) or (i13 == n13 - 1)
        b23 = (i23 == 0) or (i23 == n23 - 1)
        if (b12 + b13 + b23) >= 2:
            out.append(int(g))
    return np.array(sorted(set(out)), dtype=int)

def _line_nodes_edges_between(i12, i13, i23, j12, j13, j23, n12, n13, n23, feas_mask_flat):
    i12 = int(i12); i13 = int(i13); i23 = int(i23)
    j12 = int(j12); j13 = int(j13); j23 = int(j23)
    s12 = 0 if j12 == i12 else (1 if j12 > i12 else -1)
    s13 = 0 if j13 == i13 else (1 if j13 > i13 else -1)
    s23 = 0 if j23 == i23 else (1 if j23 > i23 else -1)
    steps = max(abs(j12 - i12), abs(j13 - i13), abs(j23 - i23))
    nodes = []
    for t in range(steps + 1):
        a12 = i12 + s12 * t
        a13 = i13 + s13 * t
        a23 = i23 + s23 * t
        if 0 <= a12 < n12 and 0 <= a13 < n13 and 0 <= a23 < n23:
            g = ijk_to_g(a12, a13, a23, n12, n13, n23)
            if feas_mask_flat[int(g)]:
                nodes.append(int(g))
    seen = set()
    out_nodes = []
    for g in nodes:
        if g not in seen:
            seen.add(g); out_nodes.append(g)
    out_nodes = np.array(out_nodes, dtype=int)
    out_edges = []
    for a, b in zip(out_nodes[:-1], out_nodes[1:]):
        i, j = (a, b) if a < b else (b, a)
        if i != j:
            out_edges.append((int(i), int(j)))
    return out_nodes, out_edges

def _all_cube_edges_ijk(n12, n13, n23):
    lo12, hi12 = 0, n12-1
    lo13, hi13 = 0, n13-1
    lo23, hi23 = 0, n23-1
    edges = []
    for a13 in (lo13, hi13):
        for a23 in (lo23, hi23):
            edges.append(((lo12, a13, a23), (hi12, a13, a23)))
    for a12 in (lo12, hi12):
        for a23 in (lo23, hi23):
            edges.append(((a12, lo13, a23), (a12, hi13, a23)))
    for a12 in (lo12, hi12):
        for a13 in (lo13, hi13):
            edges.append(((a12, a13, lo23), (a12, a13, hi23)))
    seen = set(); uniq = []
    for p, q in edges:
        key = tuple(p) + tuple(q) if tuple(p) < tuple(q) else tuple(q) + tuple(p)
        if key not in seen:
            seen.add(key); uniq.append((p, q))
    return uniq

def _select_connected_cube_edges_ijk(n12, n13, n23, anchor_ijk, edge_count):
    edge_count = int(max(0, min(12, edge_count)))
    if edge_count == 0:
        return []

    lo12, hi12 = 0, n12 - 1
    lo13, hi13 = 0, n13 - 1
    lo23, hi23 = 0, n23 - 1

    a12, a13, a23 = map(int, anchor_ijk)

    def _snap(i, lo, hi):
        if i <= lo:
            return lo
        if i >= hi:
            return hi
        return hi

    start = (_snap(a12, lo12, hi12), _snap(a13, lo13, hi13), _snap(a23, lo23, hi23))

    edges = _all_cube_edges_ijk(n12, n13, n23)
    if edge_count >= len(edges):
        return edges[:]

    adj = {}
    for (p, q) in edges:
        p = tuple(map(int, p)); q = tuple(map(int, q))
        adj.setdefault(p, []).append(q)
        adj.setdefault(q, []).append(p)

    from collections import deque
    visited = {start}
    depth = {start: 0}
    q = deque([start])

    selected = []
    selected_set = set()

    def _edge_key(p, q):
        p = tuple(p); q = tuple(q)
        return (p, q) if p < q else (q, p)

    while q and len(selected) < min(edge_count, 7):
        u = q.popleft()
        nbrs = sorted(adj.get(u, []))
        for v in nbrs:
            ekey = _edge_key(u, v)
            if ekey in selected_set:
                continue
            selected.append((u, v))
            selected_set.add(ekey)
            if v not in visited:
                visited.add(v)
                depth[v] = depth[u] + 1
                q.append(v)
            if len(selected) >= min(edge_count, 7):
                break

    if len(selected) >= edge_count:
        return selected[:edge_count]

    remaining = []
    for (p, q2) in edges:
        ekey = _edge_key(p, q2)
        if ekey in selected_set:
            continue
        dp = depth.get(tuple(p), 10**9)
        dq = depth.get(tuple(q2), 10**9)
        remaining.append((min(dp, dq), ekey, (tuple(p), tuple(q2))))
    remaining.sort(key=lambda t: (t[0], t[1]))

    for _, _, (p, q2) in remaining:
        selected.append((p, q2))
        if len(selected) >= edge_count:
            break

    return selected[:edge_count]

def _all_face_diagonals_ijk(n12, n13, n23):
    lo12, hi12 = 0, n12-1
    lo13, hi13 = 0, n13-1
    lo23, hi23 = 0, n23-1
    diags = []
    for a12 in (lo12, hi12):
        diags.append(((a12, lo13, lo23), (a12, hi13, hi23)))
        diags.append(((a12, lo13, hi23), (a12, hi13, lo23)))
    for a13 in (lo13, hi13):
        diags.append(((lo12, a13, lo23), (hi12, a13, hi23)))
        diags.append(((lo12, a13, hi23), (hi12, a13, lo23)))
    for a23 in (lo23, hi23):
        diags.append(((lo12, lo13, a23), (hi12, hi13, a23)))
        diags.append(((lo12, hi13, a23), (hi12, lo13, a23)))
    seen = set(); uniq = []
    for p, q in diags:
        key = tuple(p) + tuple(q) if tuple(p) < tuple(q) else tuple(q) + tuple(p)
        if key not in seen:
            seen.add(key); uniq.append((p, q))
    return uniq

def _all_space_diagonals_ijk(n12, n13, n23):
    lo12, hi12 = 0, n12-1
    lo13, hi13 = 0, n13-1
    lo23, hi23 = 0, n23-1
    return [
        ((lo12, lo13, lo23), (hi12, hi13, hi23)),
        ((lo12, lo13, hi23), (hi12, hi13, lo23)),
        ((lo12, hi13, lo23), (hi12, lo13, hi23)),
        ((lo12, hi13, hi23), (hi12, lo13, lo23)),
    ]

def _rank_lines_by_anchor_endpoint(lines, anchor_ijk):
    a = tuple(map(int, anchor_ijk))
    def key_fn(pair):
        p, q = pair
        p = tuple(p); q = tuple(q)
        hit = (p == a) or (q == a)
        lo, hi = (p, q) if p < q else (q, p)
        return (0 if hit else 1, lo, hi)
    return sorted(lines, key=key_fn)

def _build_step0_geometry_from_counts(anchor_g, n12, n13, n23, feas_mask_flat,
                                      edge_count, face_diag_count, space_diag_count):
    edge_count = int(np.clip(edge_count, 0, 12))
    face_diag_count = int(np.clip(face_diag_count, 0, 12))
    space_diag_count = int(np.clip(space_diag_count, 0, 4))

    a12, a13, a23 = g_to_ijk(int(anchor_g), n12, n13, n23)
    anchor_ijk = (a12, a13, a23)

    nodes_all = set([int(anchor_g)])
    extra_edges = set()

    all_edges = _select_connected_cube_edges_ijk(n12, n13, n23, anchor_ijk, edge_count)
    for (p, q) in all_edges:
        n, e = _line_nodes_edges_between(*p, *q, n12, n13, n23, feas_mask_flat)
        nodes_all.update(map(int, n.tolist()))
        for (i, j) in e:
            extra_edges.add((i, j) if i < j else (j, i))

    all_face = _rank_lines_by_anchor_endpoint(_all_face_diagonals_ijk(n12, n13, n23), anchor_ijk)
    for (p, q) in all_face[:face_diag_count]:
        n, e = _line_nodes_edges_between(*p, *q, n12, n13, n23, feas_mask_flat)
        nodes_all.update(map(int, n.tolist()))
        for (i, j) in e:
            extra_edges.add((i, j) if i < j else (j, i))

    all_space = _all_space_diagonals_ijk(n12, n13, n23)
    a = tuple(anchor_ijk)
    anchor_space = [pq for pq in all_space if (tuple(pq[0]) == a or tuple(pq[1]) == a)]
    others_space = [pq for pq in all_space if pq not in anchor_space]

    chosen_space = []
    if (space_diag_count >= 1) and (len(anchor_space) == 1):
        chosen_space.append(anchor_space[0])
        chosen_space.extend(sorted(others_space)[:max(0, space_diag_count - 1)])
    else:
        chosen_space = sorted(all_space)[:space_diag_count]

    for (p, q) in chosen_space:
        n, e = _line_nodes_edges_between(*p, *q, n12, n13, n23, feas_mask_flat)
        nodes_all.update(map(int, n.tolist()))
        for (i, j) in e:
            extra_edges.add((i, j) if i < j else (j, i))

    return np.array(sorted(nodes_all), dtype=int), list(extra_edges)

# =============================================================================
# STEP-0 CONNECTIVITY FIX
# =============================================================================
def _components_in_step0(nodes, n12, n13, n23, feas_flat):
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
            for v in neighbors_26(u, n12, n13, n23):
                v = int(v)
                if v in seen:
                    continue
                if (v in node_set) and feas_flat[v]:
                    seen.add(v)
                    stack.append(v)
        comps.append(comp)
    return comps

def enforce_step0_connected(step0_nodes, anchor_g, P_full, n12, n13, n23, feas_flat):
    anchor_g = int(anchor_g)
    step0_nodes = np.array(sorted(set(map(int, step0_nodes))), dtype=int)
    if step0_nodes.size == 0:
        return step0_nodes

    nodes_set = set(step0_nodes.tolist())
    nodes_set.add(anchor_g)

    comps = _components_in_step0(nodes_set, n12, n13, n23, feas_flat)
    comp_anchor = None
    for comp in comps:
        if anchor_g in comp:
            comp_anchor = comp
            break
    if comp_anchor is None:
        comp_anchor = [anchor_g]
        comps = [comp_anchor] + comps

    if len(comps) <= 1:
        print(f"[STEP0 connect] components=1 (already connected)")
        return np.array(sorted(nodes_set), dtype=int)

    anchor_pt = P_full[anchor_g]
    train_set = set(nodes_set)

    sizes = [len(c) for c in comps]
    print(f"[STEP0 connect] components={len(comps)} sizes={sorted(sizes, reverse=True)[:8]}{'...' if len(comps)>8 else ''}")

    for comp in comps:
        if anchor_g in comp:
            continue
        comp = np.array(comp, dtype=int)
        pts = P_full[comp]
        j = int(np.argmin(np.linalg.norm(pts - anchor_pt[None, :], axis=1)))
        g_rep = int(comp[j])
        added = shortest_path_add_3d(train_set, g_rep, anchor_g, P_full, n12, n13, n23, feas_flat)
        if len(added) == 0:
            _ = manhattan_walk_add_3d(train_set, g_rep, anchor_g, n12, n13, n23, feas_flat)

    out = np.array(sorted(train_set), dtype=int)
    comps2 = _components_in_step0(out, n12, n13, n23, feas_flat)
    print(f"[STEP0 connect] after-bridge components={len(comps2)}  nodes={out.size}")
    return out

# =============================================================================
# Overlay MST helpers
# =============================================================================
def mst_edges_over_allowed_edges(P, nodes, allowed_edges):
    nodes = np.array(sorted(set(map(int, nodes))), dtype=int)
    if nodes.size < 2:
        return []
    pos = {int(g): i for i, g in enumerate(nodes)}
    rows, cols, data = [], [], []
    for (gi, gj) in allowed_edges:
        gi = int(gi); gj = int(gj)
        ai = pos.get(gi, None); aj = pos.get(gj, None)
        if ai is None or aj is None:
            continue
        w = float(np.linalg.norm(P[gi] - P[gj]))
        rows += [ai, aj]; cols += [aj, ai]; data += [w, w]
    if len(data) == 0:
        return []
    m = nodes.size
    G = coo_matrix((np.asarray(data, float),
                    (np.asarray(rows, int), np.asarray(cols, int))),
                   shape=(m, m)).tocsr()
    T = minimum_spanning_tree(G).tocoo()
    out = []
    for a, b in zip(T.row, T.col):
        out.append((int(nodes[int(a)]), int(nodes[int(b)])))
    return out

def overlay_train_val_mst_factory(Pall, train_all_idx, train_q_idx, val_idx, anchor_idx,
                                  knn_idx=None, knn_score_map=None,
                                  mst_allowed_edges_all=None, mst_allowed_edges_q=None,
                                  scatter_swap_xy=False,
                                  step0_overlay_edges=None):

    train_all_idx = np.array(sorted(set(map(int, train_all_idx))), dtype=int)
    train_q_idx   = np.array(sorted(set(map(int, train_q_idx))), dtype=int)
    val_idx       = np.array(sorted(set(map(int, val_idx))), dtype=int)
    knn_idx = np.array([], dtype=int) if knn_idx is None else np.array(sorted(set(map(int, knn_idx))), dtype=int)

    if step0_overlay_edges:
        step0_overlay_edges = list({
            (int(a), int(b)) if int(a) < int(b) else (int(b), int(a))
            for (a, b) in step0_overlay_edges
            if int(a) != int(b)
        })
    else:
        step0_overlay_edges = []

    def _plot_mst_3d(ax, idxs, allowed_edges, color, lw, alpha, zorder=9999):
        idxs = np.array(sorted(set(map(int, idxs))), dtype=int)
        if idxs.size < 2:
            return
        if allowed_edges is not None and len(allowed_edges) > 0:
            mstE = mst_edges_over_allowed_edges(Pall, idxs, allowed_edges)
            for (gi, gj) in mstE:
                pts = _apply_axis_mapping(Pall[[gi, gj]], swap_xy=scatter_swap_xy)
                ax.plot([pts[0, 0], pts[1, 0]],
                        [pts[0, 1], pts[1, 1]],
                        [pts[0, 2], pts[1, 2]],
                        color=color, lw=lw, alpha=alpha,
                        solid_capstyle="round", clip_on=False, zorder=zorder)
        else:
            pts = _apply_axis_mapping(Pall[idxs], swap_xy=scatter_swap_xy)
            D = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
            G = minimum_spanning_tree(coo_matrix(D)).tocoo()
            for i, j in zip(G.row, G.col):
                ax.plot([pts[i, 0], pts[j, 0]],
                        [pts[i, 1], pts[j, 1]],
                        [pts[i, 2], pts[j, 2]],
                        color=color, lw=lw, alpha=alpha,
                        solid_capstyle="round", clip_on=False, zorder=zorder)

    def overlay(ax):
        Pm = _apply_axis_mapping(Pall, swap_xy=scatter_swap_xy)

        def draw_train():
            tr = Pm[train_all_idx]
            ax.scatter(tr[:, 0], tr[:, 1], tr[:, 2],
                       s=MS_TRAIN, marker=TRAIN_MARKER,
                       facecolors=TRAIN_FACE, edgecolors=TRAIN_EDGE,
                       linewidths=0.8, alpha=1.0,
                       depthshade=False, zorder=Z_TRAIN)

        def draw_val():
            if val_idx.size > 0:
                vv = val_idx[:min(15000, val_idx.size)]
                v = Pm[vv]
                ax.scatter(v[:, 0], v[:, 1], v[:, 2],
                           s=MS_VAL, marker=VAL_MARKER,
                           facecolors=VAL_FACE, edgecolors=VAL_EDGE,
                           linewidths=0.9, alpha=0.12,
                           depthshade=False, zorder=Z_VAL)

        def draw_anchor():
            a = Pm[int(anchor_idx)]
            ax.scatter([a[0]], [a[1]], [a[2]],
                       s=MS_ANCHOR, marker="*", c="gold",
                       edgecolors="k", linewidths=1.0, alpha=1.0,
                       depthshade=False, zorder=Z_ANCHOR)

        knn_mappable = None
        def draw_knn():
            nonlocal knn_mappable
            knn_mappable = None
            if SHOW_CAND_OVERLAY and knn_idx.size > 0 and isinstance(knn_score_map, dict):
                cs = np.asarray([knn_score_map.get(int(g), np.nan) for g in knn_idx], float)
                msk = np.isfinite(cs)
                if np.any(msk):
                    pts = Pm[knn_idx[msk]]
                    cs2 = cs[msk]
                    v0 = float(np.percentile(cs2, 5))
                    v1 = float(np.percentile(cs2, 95))
                    if (not np.isfinite(v0)) or (not np.isfinite(v1)) or (v1 <= v0):
                        v0 = float(np.nanmin(cs2)); v1 = float(np.nanmax(cs2))
                    cnorm = mpl.colors.Normalize(vmin=v0, vmax=v1)
                    knn_mappable = ax.scatter(
                        pts[:, 0], pts[:, 1], pts[:, 2],
                        s=CAND_MARKER_SIZE, c=cs2,
                        cmap=CAND_CMAP, norm=cnorm,
                        edgecolors="k", linewidths=CAND_EDGE_LW,
                        alpha=CAND_ALPHA, depthshade=False, zorder=Z_KNN
                    )

        def draw_mst_all():
            _plot_mst_3d(ax, train_all_idx, mst_allowed_edges_all, MST_COLOR, LINE_MST_WIDTH, 0.95, zorder=12000)

        def draw_mst_q():
            _plot_mst_3d(ax, train_q_idx, mst_allowed_edges_q, MST_Q_COLOR, LINE_MST_WIDTH * 1.15, 0.98, zorder=13000)

        def draw_step0_geometry_edges():
            if not step0_overlay_edges:
                return
            for (gi, gj) in step0_overlay_edges:
                pts = _apply_axis_mapping(Pall[[gi, gj]], swap_xy=scatter_swap_xy)
                ax.plot([pts[0, 0], pts[1, 0]],
                        [pts[0, 1], pts[1, 1]],
                        [pts[0, 2], pts[1, 2]],
                        color="k", lw=LINE_MST_WIDTH * 1.35, alpha=1.0,
                        solid_capstyle="round", clip_on=False, zorder=20000)

        if OVERLAY_ORDER_MODE == "mst_first":
            order = ["mst_all", "mst_q", "train", "val", "anchor", "knn", "step0geom"]
        else:
            order = ["train", "val", "anchor", "knn", "mst_all", "mst_q", "step0geom"]

        dispatch = {
            "mst_all": draw_mst_all, "mst_q": draw_mst_q,
            "train": draw_train, "val": draw_val,
            "anchor": draw_anchor, "knn": draw_knn,
            "step0geom": draw_step0_geometry_edges,
        }
        for key in order:
            fn = dispatch.get(key, None)
            if fn is not None:
                fn()

        if knn_mappable is not None:
            return {"knn_mappable": knn_mappable, "knn_label": CAND_CBAR_LABEL}
        return None

    return overlay

# =============================================================================
# Metrics plots
# =============================================================================
def plot_rmse_history_each_step(rmse_tr_hist, rmse_val_hist, step_name, target_label, train_set_label="paid"):
    fig, ax = plt.subplots(figsize=FIGSIZE_RMSE)
    its = np.arange(0, len(rmse_tr_hist))
    ax.plot(its, rmse_tr_hist, "-", lw=5, label=f"Train ({train_set_label},{target_label})")
    ax.plot(its, rmse_val_hist, "-", lw=5, label=f"Val (oracle,{target_label})")
    ax.set_xlabel("AL step", fontsize=AXIS_LABEL_FZ)
    ax.set_ylabel("RMSE", fontsize=AXIS_LABEL_FZ)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis="x", labelsize=TICK_LABEL_FZ, pad=6)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FZ, pad=6)
    ax.legend(frameon=False, fontsize=LEGEND_FZ, loc="upper right",
          markerscale=2.0, handlelength=0.5, handletextpad=0.6,
          labelspacing=0.5, borderpad=0.2, borderaxespad=0.2)
    savefig(fig, f"{target_label}_AL_step{step_name}_rmse_trace")

def plot_mst_history_each_step(mst_hist, step_name):
    fig, ax = plt.subplots(figsize=FIGSIZE_RMSE)
    steps = np.arange(len(mst_hist))
    ax.plot(steps, mst_hist, "-o", lw=5, ms=16, mfc="none")
    ax.set_xlabel("AL step", fontsize=AXIS_LABEL_FZ)
    ax.set_ylabel("MST length", fontsize=AXIS_LABEL_FZ)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis="x", labelsize=TICK_LABEL_FZ, pad=6)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FZ, pad=6)
    savefig(fig, f"Umb_AL_step{step_name}_mst_trace")

def plot_parity_each_step(model, X_tr, y_tr, X_val, y_val, ymu, ysd, step_name, target_label, train_set_label="paid"):
    mu_tr_z, _ = predict_mu_std_z(model, X_tr)
    mu_val_z, _ = predict_mu_std_z(model, X_val)
    yhat_tr = z2y(mu_tr_z, ymu, ysd)
    yhat_val = z2y(mu_val_z, ymu, ysd)

    fig, ax = plt.subplots(figsize=FIGSIZE_PARITY)
    ax.scatter(y_tr, yhat_tr, s=160, alpha=1, label=f"Train ({train_set_label})")
    ax.scatter(y_val, yhat_val, s=160, alpha=0.35, edgecolors="none", label="Val (oracle)")

    allv = np.concatenate([y_tr[np.isfinite(y_tr)], y_val[np.isfinite(y_val)],
                           yhat_tr[np.isfinite(yhat_tr)], yhat_val[np.isfinite(yhat_val)]])
    if allv.size > 0:
        vmin = float(np.nanpercentile(allv, 1))
        vmax = float(np.nanpercentile(allv, 99))
    else:
        vmin, vmax = -1.0, 1.0

    ax.plot([vmin, vmax], [vmin, vmax], "k-", lw=2.0, alpha=0.7)
    ax.set_xlim(vmin, vmax); ax.set_ylim(vmin, vmax)
    ax.set_xlabel(f"True ({target_label}, anchored)", fontsize=AXIS_LABEL_FZ)
    ax.set_ylabel(f"Predicted ({target_label}, anchored)", fontsize=AXIS_LABEL_FZ)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FZ, pad=6)
    ax.legend(frameon=False, fontsize=LEGEND_FZ, loc="upper left",
          markerscale=1.5, handlelength=0.2, handletextpad=0.6,
          labelspacing=0.5, borderpad=0.2, borderaxespad=0.2)
    savefig(fig, f"{target_label}_AL_step{step_name}_parity")

def plot_lsqr_paid_vs_fullgrid_parity(U_paid, paid_nodes, U_full_map, step_name):
    paid_nodes = np.array(paid_nodes, dtype=int)
    u_paid = np.asarray(U_paid, float).reshape(-1)
    u_full = np.asarray(U_full_map[paid_nodes], float).reshape(-1)
    m = np.isfinite(u_paid) & np.isfinite(u_full)
    if not np.any(m):
        return
    u_paid = u_paid[m]; u_full = u_full[m]
    r = rmse(u_full, u_paid)
    fig, ax = plt.subplots(figsize=FIGSIZE_PARITY)
    ax.scatter(u_full, u_paid, s=22, alpha=0.55, edgecolors="none")
    vmin = float(np.nanpercentile(np.r_[u_full, u_paid], 1))
    vmax = float(np.nanpercentile(np.r_[u_full, u_paid], 99))
    ax.plot([vmin, vmax], [vmin, vmax], "k-", lw=2.0, alpha=0.7)
    ax.set_xlim(vmin, vmax); ax.set_ylim(vmin, vmax)
    ax.set_xlabel("Full-grid LSQR (anchored Umb)", fontsize=AXIS_LABEL_FZ)
    ax.set_ylabel("On-the-fly LSQR (anchored Umb)", fontsize=AXIS_LABEL_FZ)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FZ, pad=6)
    ax.text(0.02, 0.98, f"RMSE = {r:.4f}\nN = {u_full.size}",
            transform=ax.transAxes, va="top", ha="left", fontsize=LEGEND_FZ)
    savefig(fig, f"LSQR_AL_step{step_name}_paid_vs_fullgrid_parity")

# =============================================================================
# Force computation on masked feasible region (for LSQR teacher forces)
# =============================================================================
def fill_infeasible_by_nearest(Ugrid_nan, feas_mask, d12, d13, d23):
    Ugrid_nan = np.asarray(Ugrid_nan, float)
    feas_mask = np.asarray(feas_mask, bool)

    if not np.any(feas_mask):
        return np.zeros_like(Ugrid_nan)

    D12c, D13c, D23c = np.meshgrid(d12, d13, d23, indexing="ij")
    coords = np.c_[D12c[feas_mask].ravel(), D13c[feas_mask].ravel(), D23c[feas_mask].ravel()]
    vals   = Ugrid_nan[feas_mask].ravel()

    tree = cKDTree(coords)

    nan_mask = ~feas_mask
    if not np.any(nan_mask):
        return Ugrid_nan.copy()

    qcoords = np.c_[D12c[nan_mask].ravel(), D13c[nan_mask].ravel(), D23c[nan_mask].ravel()]
    _, nn = tree.query(qcoords, k=1)
    filled = Ugrid_nan.copy()
    filled[nan_mask] = vals[nn]
    filled = np.nan_to_num(filled, nan=0.0)
    return filled

def compute_teacher_forces(Ugrid_nan, feas, d12, d13, d23):
    U_filled = fill_infeasible_by_nearest(Ugrid_nan, feas, d12, d13, d23)

    dd12 = float(d12[1] - d12[0]) if len(d12) > 1 else 1.0
    dd13 = float(d13[1] - d13[0]) if len(d13) > 1 else 1.0
    dd23 = float(d23[1] - d23[0]) if len(d23) > 1 else 1.0

    dU_dd12, dU_dd13, dU_dd23 = np.gradient(U_filled, dd12, dd13, dd23, edge_order=2)

    dU_dd12 = np.where(feas, dU_dd12, np.nan)
    dU_dd13 = np.where(feas, dU_dd13, np.nan)
    dU_dd23 = np.where(feas, dU_dd23, np.nan)
    return dU_dd12, dU_dd13, dU_dd23

# =============================================================================
# Per-step full plotting
# =============================================================================
def plot_all_figs_for_step(step_name, model, plot_g, P_full, P_feas, feas_pos, Xs_feas,
                           xmu, xsd, ymu, ysd, normU,
                           U_oracle_flat, U_oracle_diag, U_oracle_face,
                           train_all_set, train_q_set, val_g, anchor_g,
                           d12s, d13s, D12s, D13s, D23s, mask2,
                           d13f, d23f, D13f, D23f, D12f, maskf,
                           p2, W2_anchor,
                           score_map_for_this_step=None, cand_overlay_idx_for_this_step=None,
                           X_tr=None, y_tr=None, X_val=None, y_val=None,
                           rmse_tr_hist=None, rmse_val_hist=None, mst_hist=None,
                           n12=None, n13=None, n23=None, feas_flat=None,
                           step0_extra_edges=None,
                           step0_overlay_edges=None,
                           target_mode="umb",
                           W2anch_feas=None,
                           target_label="Umb",
                           symmetry_mode="sorted",
                           train_set_label="paid",
                           U_full_map=None, paid_nodes_step=None, U_paid_step=None):

    if cand_overlay_idx_for_this_step is None:
        cand_overlay_idx_for_this_step = np.array([], dtype=int)

    mst_edges_all = None
    mst_edges_q = None
    if (n12 is not None) and (n13 is not None) and (n23 is not None) and (feas_flat is not None):
        mode_use = STEP0_LSQR_GRAPH_MODE if str(step_name) == "00" else LSQR_GRAPH_MODE
        mode = str(mode_use).lower()

        if mode == "axis1":
            mst_edges_all = build_edges_axis1_from_nodes(train_all_set, n12, n13, n23, feas_flat)
            mst_edges_q   = build_edges_axis1_from_nodes(train_q_set,   n12, n13, n23, feas_flat)
            if step0_extra_edges:
                mst_edges_all = list(set(tuple(sorted(e)) for e in (mst_edges_all + step0_extra_edges)))
                mst_edges_q   = list(set(tuple(sorted(e)) for e in (mst_edges_q   + step0_extra_edges)))
        elif mode == "knn":
            mst_edges_all = build_edges_knn_from_nodes(train_all_set, P_full, k=LSQR_KNN_K, max_dist=LSQR_KNN_MAX_DIST)
            mst_edges_q   = build_edges_knn_from_nodes(train_q_set,   P_full, k=LSQR_KNN_K, max_dist=LSQR_KNN_MAX_DIST)
        else:
            mst_edges_all = build_edges_from_nodes(train_all_set, n12, n13, n23, feas_flat)
            mst_edges_q   = build_edges_from_nodes(train_q_set,   n12, n13, n23, feas_flat)

    overlay = overlay_train_val_mst_factory(
        P_full,
        train_all_idx=np.array(sorted(train_all_set), int),
        train_q_idx=np.array(sorted(train_q_set), int),
        val_idx=val_g,
        anchor_idx=anchor_g,
        knn_idx=cand_overlay_idx_for_this_step,
        knn_score_map=score_map_for_this_step,
        mst_allowed_edges_all=mst_edges_all,
        mst_allowed_edges_q=mst_edges_q,
        scatter_swap_xy=False,
        step0_overlay_edges=step0_overlay_edges
    )

    if (U_full_map is not None) and (paid_nodes_step is not None) and (U_paid_step is not None):
        plot_lsqr_paid_vs_fullgrid_parity(U_paid_step, paid_nodes_step, U_full_map, step_name)

    mu_z, std_z = predict_mu_std_z(model, Xs_feas)
    mu_target = z2y(mu_z, ymu, ysd)
    std_target = std_z * ysd

    if str(target_mode).lower() == "residual":
        if W2anch_feas is None:
            raise RuntimeError("W2anch_feas must be provided when target_mode='residual'.")
        mu = mu_target + W2anch_feas
    else:
        mu = mu_target

    plot_pos = feas_pos[plot_g]

    plot_scatter3d_color(
        P_feas[plot_pos], mu[plot_pos], LAB_C,
        f"Umb_AL_step{step_name}_mu_scatter",
        title=rf"Student $\mu$ (AL step {step_name}), {GRAFT}",
        cmap="viridis", norm=normU, overlay=overlay,
        scatter_swap_xy=False
    )

    mu2_t, sig2 = gp_predict_on_slice(model, xmu, xsd, ymu, ysd, D12s, D13s, symmetry_mode)
    mu_face_t, _ = gp_predict_on_face(model, xmu, xsd, ymu, ysd, D13f, D23f, D12f, symmetry_mode)

    if str(target_mode).lower() == "residual":
        W2sum_diag = W2sum_pairs(D12s, D13s, D23s, p2)
        mu2 = mu2_t + (W2sum_diag - W2_anchor)
        W2sum_face = W2sum_pairs(D12f, D13f, D23f, p2)
        mu_face = mu_face_t + (W2sum_face - W2_anchor)
    else:
        mu2 = mu2_t
        mu_face = mu_face_t

    err_scatter = _abs_err(mu[plot_pos], U_oracle_flat[plot_g])
    err_diag = _abs_err(mu2, U_oracle_diag)
    err_face = _abs_err(mu_face, U_oracle_face)

    err_vals = []

    e0 = err_scatter[np.isfinite(err_scatter)]
    if e0.size > 0:
        err_vals.append(e0)

    e2 = err_diag[mask2 & np.isfinite(err_diag)]
    if e2.size > 0:
        err_vals.append(e2)

    ef = err_face[maskf & np.isfinite(err_face)]
    if ef.size > 0:
        err_vals.append(ef)

    if len(err_vals) > 0:
        err_all = np.concatenate(err_vals)
        err_vmax = float(np.nanpercentile(err_all, 99.0))
        if (not np.isfinite(err_vmax)) or (err_vmax <= 0.0):
            err_vmax = float(np.nanmax(err_all)) if err_all.size > 0 else 1.0
        if (not np.isfinite(err_vmax)) or (err_vmax <= 0.0):
            err_vmax = 1.0
    else:
        err_vmax = 1.0

    normErr = mpl.colors.Normalize(vmin=0.0, vmax=err_vmax)

    plot_scatter3d_color(
        P_feas[plot_pos], err_scatter, LAB_ABSERR,
        f"Umb_AL_step{step_name}_abserr_scatter",
        title=rf"Absolute error scatter, step {step_name}, {GRAFT}",
        cmap=ABSERR_CMAP, norm=normErr, overlay=overlay,
        scatter_swap_xy=False
    )

    normS = mpl.colors.Normalize(vmin=float(np.nanmin(std_target)), vmax=float(np.nanmax(std_target)))
    plot_scatter3d_color(
        P_feas[plot_pos], std_target[plot_pos], r"$\sigma_{\mathrm{pred}}$",
        f"Umb_AL_step{step_name}_sigma_scatter",
        title=rf"Student $\sigma$ (AL step {step_name}), {GRAFT}",
        cmap="plasma", norm=normS, overlay=overlay,
        scatter_swap_xy=False
    )

    plot_slice_2d_heat(D12s, D13s, mu2, mask2,
        f"Umb_AL_step{step_name}_mu_slice2D",
        title=rf"Student $\mu$ diag slice ($d_{{23}}=d_{{13}}$), step {step_name}, {GRAFT}",
        clabel=LAB_C, cmap="viridis", norm=normU
    )
    plot_slice_3d_surface(D12s, D13s, mu2, mask2,
        f"Umb_AL_step{step_name}_mu_slice3D",
        title=rf"Student $\mu$ diag surface ($d_{{23}}=d_{{13}}$), step {step_name}, {GRAFT}",
        clabel=LAB_C, cmap="viridis", norm=normU
    )

    plot_slice_2d_heat(D12s, D13s, err_diag, mask2,
        f"Umb_AL_step{step_name}_abserr_slice2D",
        title=rf"Absolute error diag slice ($d_{{23}}=d_{{13}}$), step {step_name}, {GRAFT}",
        clabel=LAB_ABSERR, cmap=ABSERR_CMAP, norm=normErr
    )
    plot_slice_3d_surface(D12s, D13s, err_diag, mask2,
        f"Umb_AL_step{step_name}_abserr_slice3D",
        title=rf"Absolute error diag surface ($d_{{23}}=d_{{13}}$), step {step_name}, {GRAFT}",
        clabel=LAB_ABSERR, cmap=ABSERR_CMAP, norm=normErr
    )

    normSig = mpl.colors.Normalize(vmin=float(np.nanmin(sig2[mask2])), vmax=float(np.nanmax(sig2[mask2])))
    plot_slice_2d_heat(D12s, D13s, sig2, mask2,
        f"Umb_AL_step{step_name}_sigma_slice2D",
        title=rf"Student $\sigma$ diag slice ($d_{{23}}=d_{{13}}$), step {step_name}, {GRAFT}",
        clabel=r"$\sigma_{\mathrm{pred}}$", cmap="plasma", norm=normSig
    )
    plot_slice_3d_surface(D12s, D13s, sig2, mask2,
        f"Umb_AL_step{step_name}_sigma_slice3D",
        title=rf"Student $\sigma$ diag surface ($d_{{23}}=d_{{13}}$), step {step_name}, {GRAFT}",
        clabel=r"$\sigma_{\mathrm{pred}}$", cmap="plasma", norm=normSig
    )

    plot_face_2d_heat(d13f, d23f, mu_face, maskf,
        f"Umb_AL_step{step_name}_mu_face_d12min_slice2D",
        title=rf"Student $\mu$ face ($d_{{12}}={D12_MIN:.2f}$), step {step_name}, {GRAFT}",
        clabel=LAB_C, cmap="viridis", norm=normU
    )
    plot_face_3d_surface(D13f, D23f, mu_face, maskf,
        f"Umb_AL_step{step_name}_mu_face_d12min_slice3D",
        title=rf"Student $\mu$ face surface ($d_{{12}}={D12_MIN:.2f}$), step {step_name}, {GRAFT}",
        clabel=LAB_C, cmap="viridis", norm=normU
    )

    plot_face_2d_heat(d13f, d23f, err_face, maskf,
        f"Umb_AL_step{step_name}_abserr_face_d12min_slice2D",
        title=rf"Absolute error face ($d_{{12}}={D12_MIN:.2f}$), step {step_name}, {GRAFT}",
        clabel=LAB_ABSERR, cmap=ABSERR_CMAP, norm=normErr
    )
    plot_face_3d_surface(D13f, D23f, err_face, maskf,
        f"Umb_AL_step{step_name}_abserr_face_d12min_slice3D",
        title=rf"Absolute error face surface ($d_{{12}}={D12_MIN:.2f}$), step {step_name}, {GRAFT}",
        clabel=LAB_ABSERR, cmap=ABSERR_CMAP, norm=normErr
    )

    W2sum_diag = W2sum_pairs(D12s, D13s, D23s, p2)
    dW3_diag_ex = mu2 - (W2sum_diag - W2_anchor)
    normDW3_diag = _norm_from_masked(dW3_diag_ex, mask2)
    plot_slice_2d_heat(D12s, D13s, dW3_diag_ex, mask2,
        f"Umb_AL_step{step_name}_extracted_deltaW3_diag_slice2D",
        title=rf"Student extracted $\Delta W_3$ diag ($d_{{23}}=d_{{13}}$), step {step_name}, {GRAFT}",
        clabel=LAB_DW3, cmap="viridis", norm=normDW3_diag
    )
    plot_slice_3d_surface(D12s, D13s, dW3_diag_ex, mask2,
        f"Umb_AL_step{step_name}_extracted_deltaW3_diag_slice3D",
        title=rf"Student extracted $\Delta W_3$ diag surface, step {step_name}, {GRAFT}",
        clabel=LAB_DW3, cmap="viridis", norm=normDW3_diag
    )

    W2sum_face = W2sum_pairs(D12f, D13f, D23f, p2)
    dW3_face_ex = mu_face - (W2sum_face - W2_anchor)
    normDW3_face = _norm_from_masked(dW3_face_ex, maskf)
    plot_face_2d_heat(d13f, d23f, dW3_face_ex, maskf,
        f"Umb_AL_step{step_name}_extracted_deltaW3_face_d12min_slice2D",
        title=rf"Student extracted $\Delta W_3$ face ($d_{{12}}={D12_MIN:.2f}$), step {step_name}, {GRAFT}",
        clabel=LAB_DW3, cmap="viridis", norm=normDW3_face
    )
    plot_face_3d_surface(D13f, D23f, dW3_face_ex, maskf,
        f"Umb_AL_step{step_name}_extracted_deltaW3_face_d12min_slice3D",
        title=rf"Student extracted $\Delta W_3$ face surface, step {step_name}, {GRAFT}",
        clabel=LAB_DW3, cmap="viridis", norm=normDW3_face
    )

    if (X_tr is not None) and (y_tr is not None) and (X_val is not None) and (y_val is not None):
        plot_parity_each_step(model, X_tr, y_tr, X_val, y_val, ymu, ysd, step_name, target_label, train_set_label=train_set_label)
    if (rmse_tr_hist is not None) and (rmse_val_hist is not None):
        plot_rmse_history_each_step(rmse_tr_hist, rmse_val_hist, step_name, target_label, train_set_label=train_set_label)
    if mst_hist is not None:
        plot_mst_history_each_step(mst_hist, step_name)

# =============================================================================
# Small redundancy removal helpers (core fix lives here)
# =============================================================================
def make_train_xy_from_nodes(train_nodes, y_phys, Xs_feas, feas_pos):
    train_nodes = np.asarray(train_nodes, dtype=int)
    pos = feas_pos[train_nodes]
    m = pos >= 0
    if not np.any(m):
        return np.zeros((0,3)), np.zeros((0,)), np.zeros((0,)), 0.0, 1.0
    X_tr = Xs_feas[pos[m]]
    y_tr = np.asarray(y_phys, float)[m]

    X_tr, y_tr = _sanitize_training_data(X_tr, y_tr)
    if X_tr.shape[0] == 0:
        return X_tr, y_tr, y_tr, 0.0, 1.0

    ymu, ysd = y_fit(y_tr)
    z_tr = y2z(y_tr, ymu, ysd)

    assert X_tr.shape[0] == y_tr.shape[0] == z_tr.shape[0]
    return X_tr, y_tr, z_tr, ymu, ysd

def eval_model_rmse(model, X, y_true_phys, ymu_model, ysd_model):
    mu_z, _ = predict_mu_std_z(model, X)
    yhat = z2y(mu_z, ymu_model, ysd_model)
    return rmse(y_true_phys, yhat)

# =============================================================================
# MAIN
# =============================================================================
def main():
    global FIG_DIR

    args = parse_cli()
    START_MODE   = args.start_mode
    TARGET_MODE  = args.target_mode
    SYMM_MODE    = args.symmetry_mode
    KMODE        = args.kernel_mode
    LABEL_SOURCE = args.label_source

    USE_LSQR = (LABEL_SOURCE == "lsqr")
    TRAIN_SET_LABEL = "queried" if LABEL_SOURCE == "oracle_queried" else "paid"

    TARGET_LABEL = "Umb" if TARGET_MODE == "umb" else "Residual"
    print(f"[CONFIG] start_mode={START_MODE}  target_mode={TARGET_MODE}  sym={SYMM_MODE}  "
          f"kernel={KMODE}  label_source={LABEL_SOURCE}  graft={GRAFT}  N3={N3}")

    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    np.set_printoptions(precision=4, suppress=True)
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    gpflow.config.set_default_float(np.float64)
    gpflow.config.set_default_jitter(JITTER_DEF)

    FIG_DIR = Path(
        f"./figs_{RUN_TAG}_stage2_AL_md_poly_oracle_3D_forceint_"
        f"{GRAFT}_"
        f"edges{INIT_EDGE_COUNT}_face{INIT_FACE_DIAG_COUNT}_space{INIT_SPACE_DIAG_COUNT}_"
        f"{START_MODE}_{TARGET_MODE}_sym{SYMM_MODE}_k{KMODE}_lab{LABEL_SOURCE}"
    )
    FIG_DIR.mkdir(exist_ok=True)
    print(f"[OUT] FIG_DIR = {FIG_DIR}")

    rng = np.random.default_rng(SEED)
    p2 = PARAMS[GRAFT]["dimer"]
    p3 = PARAMS[GRAFT]["trimer"]

    d12 = np.linspace(D12_MIN, D12_MAX, N3)
    d13 = np.linspace(D13_MIN, D13_MAX, N3)
    d23 = np.linspace(D23_MIN, D23_MAX, N3)
    n12, n13, n23 = len(d12), len(d13), len(d23)

    D12v, D13v, D23v = np.meshgrid(d12, d13, d23, indexing="ij")
    feas = triangle_mask(D12v, D13v, D23v)

    Umb3_raw, W3sw3, W3raw3 = Umb(D12v, D13v, D23v, p2, p3)
    Umb3   = np.where(feas, Umb3_raw, np.nan)
    W3sw3  = np.where(feas, W3sw3,    np.nan)

    feas_flat = np.isfinite(Umb3.ravel(order="C"))
    feasible_g = np.where(feas_flat)[0]
    if feasible_g.size == 0:
        raise RuntimeError("No feasible triangles on this grid. Adjust ranges or mask.")

    P_full = np.c_[D13v.ravel(order="C"), D23v.ravel(order="C"), D12v.ravel(order="C")]

    edge_all = cube_edge_nodes(feasible_g, n12, n13, n23)
    anchor_target = np.array([D13_MAX, D23_MAX, D12_MAX], float)
    if edge_all.size > 0:
        kdt_edge = cKDTree(P_full[edge_all])
        anchor_g = int(edge_all[int(kdt_edge.query(anchor_target, k=1)[1])])
    else:
        kdt_feas = cKDTree(P_full[feasible_g])
        anchor_g = int(feasible_g[int(kdt_feas.query(anchor_target, k=1)[1])])

    step0_nodes, step0_extra_edges = _build_step0_geometry_from_counts(
        anchor_g=anchor_g, n12=n12, n13=n13, n23=n23,
        feas_mask_flat=feas_flat,
        edge_count=INIT_EDGE_COUNT,
        face_diag_count=INIT_FACE_DIAG_COUNT,
        space_diag_count=INIT_SPACE_DIAG_COUNT
    )
    step0_nodes_connected = enforce_step0_connected(step0_nodes, anchor_g, P_full, n12, n13, n23, feas_flat)

    print(f"[STEP0 geometry] requested edges/face/space = {INIT_EDGE_COUNT}/{INIT_FACE_DIAG_COUNT}/{INIT_SPACE_DIAG_COUNT}")
    print(f"[STEP0 geometry] nodes(before connect)={len(step0_nodes)} nodes(after connect)={len(step0_nodes_connected)} "
          f"extra_edges={len(step0_extra_edges)} anchor_g={anchor_g}")

    step0_overlay_edges = step0_extra_edges if (step0_extra_edges is not None and len(step0_extra_edges) > 0) else None

    U_raw_flat = Umb3.ravel(order="C")
    U_anchor_raw = float(U_raw_flat[anchor_g])
    U_ref = U_raw_flat - U_anchor_raw

    d13_a, d23_a, d12_a = P_full[anchor_g]
    W2_anchor = float(W2sum_pairs(d12_a, d13_a, d23_a, p2))

    d13_flat = P_full[:, 0]
    d23_flat = P_full[:, 1]
    d12_flat = P_full[:, 2]
    W2sum_flat = W2sum_pairs(d12_flat, d13_flat, d23_flat, p2)
    W2anch_flat = W2sum_flat - W2_anchor
    RES_ref = U_ref - W2anch_flat

    plot_g = feasible_g
    if plot_g.size > PLOT_MAX:
        plot_g = rng.choice(plot_g, size=PLOT_MAX, replace=False)

    normU = _norm_from_clim()
    d12s, d13s, D12s, D13s, D23s, mask2 = build_diag_slice_grid()
    d13f, d23f, D13f, D23f, D12f, maskf = build_face_slice_grid()

    plot_scatter3d_color(
        P_full[plot_g], U_ref[plot_g], LAB_C,
        "Umb_01_oracle_Umb_scatter",
        title=rf"MD-poly oracle $U_{{\mathrm{{mb}}}}$ (anchored), {GRAFT}",
        cmap="viridis", norm=normU,
        cbar_ticks=np.arange(CLIM_MIN, CLIM_MAX + 1e-9, 10) if USE_FIXED_CLIM else None,
        scatter_swap_xy=False
    )

    U2, _, _ = oracle_on_slice(D12s, D13s, p2, p3)
    U2a = U2 - U_anchor_raw
    plot_slice_2d_heat(D12s, D13s, U2a, mask2, "Umb_01_oracle_Umb_slice2D",
        title=rf"MD-poly oracle $U_{{\mathrm{{mb}}}}$ diag slice (anchored), {GRAFT}",
        clabel=LAB_C, cmap="viridis", norm=normU
    )
    plot_slice_3d_surface(D12s, D13s, U2a, mask2, "Umb_01_oracle_Umb_slice3D",
        title=rf"MD-poly oracle $U_{{\mathrm{{mb}}}}$ diag surface (anchored), {GRAFT}",
        clabel=LAB_C, cmap="viridis", norm=normU
    )

    Uface, _, _ = oracle_on_face(D13f, D23f, D12f, p2, p3)
    Uface_a = Uface - U_anchor_raw
    plot_face_2d_heat(d13f, d23f, Uface_a, maskf, "Umb_01_oracle_Umb_face_d12min_slice2D",
        title=rf"MD-poly oracle $U_{{\mathrm{{mb}}}}$ face ($d_{{12}}={D12_MIN:.2f}$) (anchored), {GRAFT}",
        clabel=LAB_C, cmap="viridis", norm=normU
    )
    plot_face_3d_surface(D13f, D23f, Uface_a, maskf, "Umb_01_oracle_Umb_face_d12min_slice3D",
        title=rf"MD-poly oracle $U_{{\mathrm{{mb}}}}$ face surface ($d_{{12}}={D12_MIN:.2f}$) (anchored), {GRAFT}",
        clabel=LAB_C, cmap="viridis", norm=normU
    )

    dW3_flat = W3sw3.ravel(order="C")
    plot_scatter3d_color(P_full[plot_g], dW3_flat[plot_g], LAB_DW3,
        "Umb_02_oracle_deltaW3_scatter",
        title=rf"MD-poly oracle $\Delta W_3$, {GRAFT}",
        cmap="viridis", norm=None, scatter_swap_xy=False
    )

    U_full_map = None
    Fx = Fy = Fz = None

    def teacher_query_force_at_nodes(_nodes):
        return

    if USE_LSQR:
        Ugrid = U_ref.reshape(n12, n13, n23, order="C")
        Ugrid_nan = np.where(feas, Ugrid, np.nan)

        dU_dd12, dU_dd13, dU_dd23 = compute_teacher_forces(Ugrid_nan, feas, d12, d13, d23)
        Fd12_flat = dU_dd12.ravel(order="C")
        Fd13_flat = dU_dd13.ravel(order="C")
        Fd23_flat = dU_dd23.ravel(order="C")

        Fx = np.full(P_full.shape[0], np.nan, float)
        Fy = np.full(P_full.shape[0], np.nan, float)
        Fz = np.full(P_full.shape[0], np.nan, float)

        def teacher_query_force_at_nodes(nodes):
            nodes = np.array(sorted(set(map(int, nodes))), dtype=int)
            if nodes.size == 0:
                return
            nodes = nodes[feas_flat[nodes]]
            if nodes.size == 0:
                return
            Fx[nodes] = Fd13_flat[nodes]
            Fy[nodes] = Fd23_flat[nodes]
            Fz[nodes] = Fd12_flat[nodes]

        if DO_FULLGRID_LSQR_DIAGNOSTIC:
            teacher_query_force_at_nodes(feasible_g)
            U_full, nodes_full = lsqr_U_on_known_nodes_3d(
                known_nodes=feasible_g, anchor_g=anchor_g, P_full=P_full,
                Fx=Fx, Fy=Fy, Fz=Fz,
                n12=n12, n13=n13, n23=n23, feas_mask_flat=feas_flat,
                iter_lim=20000,
                graph_mode=LSQR_GRAPH_MODE,
                knn_k=LSQR_KNN_K,
                knn_max_dist=LSQR_KNN_MAX_DIST,
                extra_edges=None
            )
            U_full_map = np.full(P_full.shape[0], np.nan, float)
            U_full_map[nodes_full] = U_full
            keep = np.isfinite(U_full_map[feasible_g]) & np.isfinite(U_ref[feasible_g])
            print(f"[LSQR full-grid diagnostic] RMSE vs oracle (feasible) = "
                  f"{rmse(U_ref[feasible_g][keep], U_full_map[feasible_g][keep]):.6f}")
            if PLOT_FULLGRID_LSQR_DIAGNOSTIC:
                plot_scatter3d_color(
                    P_full[plot_g], U_full_map[plot_g], LAB_C,
                    "Umb_03_diag_LSQR_fullgrid_scatter",
                    title=rf"LSQR PMF from oracle forces (full-grid diagnostic), {GRAFT}",
                    cmap="viridis", norm=normU, scatter_swap_xy=False
                )
            Fx[:] = np.nan; Fy[:] = np.nan; Fz[:] = np.nan

    P_feas = P_full[feasible_g]
    P_feat_feas = _features_from_P(P_feas, symmetry_mode=SYMM_MODE)
    xmu, xsd = standardize_fit(P_feat_feas)
    Xs_feas = standardize_apply(P_feat_feas, xmu, xsd)

    W2sum_feas = W2sum_pairs(P_feas[:, 2], P_feas[:, 0], P_feas[:, 1], p2)
    W2anch_feas = W2sum_feas - W2_anchor

    feas_pos = -np.ones(P_full.shape[0], dtype=int)
    feas_pos[feasible_g] = np.arange(feasible_g.size, dtype=int)

    exclude0 = np.union1d(step0_nodes_connected, np.array([anchor_g], dtype=int))
    val_pool = np.setdiff1d(feasible_g, exclude0)
    n_val = int(np.clip(VAL_FRAC * feasible_g.size, VAL_MIN, VAL_MAX))
    if val_pool.size == 0:
        val_pool = feasible_g[feasible_g != anchor_g]
    val_g = rng.choice(val_pool, size=min(n_val, val_pool.size), replace=False)

    is_val = np.zeros(P_full.shape[0], dtype=bool)
    is_val[val_g] = True
    is_val[int(anchor_g)] = False

    allowed_train_mask = feas_flat & (~is_val)
    allowed_train_mask[int(anchor_g)] = True

    y_val_true = U_ref[val_g] if TARGET_MODE == "umb" else RES_ref[val_g]
    X_val = Xs_feas[feas_pos[val_g]]

    def _oracle_target_on_nodes(nodes):
        nodes = np.asarray(nodes, dtype=int)
        if TARGET_MODE == "umb":
            return np.asarray(U_ref[nodes], float)
        else:
            return np.asarray(RES_ref[nodes], float)

    def _build_training_nodes_and_labels(label_source, train_all, train_q,
                                         anchor_g, P_full, n12, n13, n23, feas_mask_flat,
                                         Fx, Fy, Fz, graph_mode, extra_edges):
        ls = str(label_source).lower()

        if ls == "lsqr":
            U_paid, paid_nodes = lsqr_U_on_known_nodes_3d(
                known_nodes=train_all, anchor_g=anchor_g, P_full=P_full,
                Fx=Fx, Fy=Fy, Fz=Fz,
                n12=n12, n13=n13, n23=n23, feas_mask_flat=feas_mask_flat,
                iter_lim=12000,
                graph_mode=graph_mode,
                knn_k=LSQR_KNN_K,
                knn_max_dist=LSQR_KNN_MAX_DIST,
                extra_edges=extra_edges
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

        elif ls == "oracle_paid":
            nodes = np.array(sorted(set(map(int, train_all))), dtype=int)
            nodes = nodes[feas_mask_flat[nodes]]
            y = _oracle_target_on_nodes(nodes)
            return nodes, y, None, None

        elif ls == "oracle_queried":
            nodes = np.array(sorted(set(map(int, train_q))), dtype=int)
            nodes = nodes[feas_mask_flat[nodes]]
            y = _oracle_target_on_nodes(nodes)
            return nodes, y, None, None

        else:
            raise ValueError(f"Unknown label_source={label_source}")

    train_q   = set(map(int, step0_nodes_connected.tolist()))
    train_all = set(train_q)

    extra0 = step0_extra_edges if (step0_extra_edges is not None and len(step0_extra_edges) > 0) else None
    if USE_LSQR:
        teacher_query_force_at_nodes(step0_nodes_connected)

    train_nodes, y_phys_raw, paid_nodes_lsqr, U_paid_lsqr = _build_training_nodes_and_labels(
        LABEL_SOURCE, train_all, train_q,
        anchor_g, P_full, n12, n13, n23, feas_flat,
        Fx, Fy, Fz,
        graph_mode=STEP0_LSQR_GRAPH_MODE,
        extra_edges=extra0
    )
    if train_nodes.size == 0:
        raise RuntimeError("Initial labeling produced no training labels (check LSQR/forces or node sets).")

    X_tr, y_tr, z_tr, ymu, ysd = make_train_xy_from_nodes(train_nodes, y_phys_raw, Xs_feas, feas_pos)
    if X_tr.shape[0] < 2:
        raise RuntimeError("Not enough training points after sanitization.")

    prev_m = build_gpr(
        X_tr, z_tr, KMODE, KERNEL_SPEC, ARD, INIT_ITERS,
        m_prev=None
    )
    prev_ymu, prev_ysd = ymu, ysd

    rtr0 = eval_model_rmse(prev_m, X_tr, y_tr, prev_ymu, prev_ysd)
    rv0  = eval_model_rmse(prev_m, X_val, y_val_true, prev_ymu, prev_ysd)

    rmse_tr_hist  = [rtr0]
    rmse_val_hist = [rv0]
    mst_hist      = [mst_total_length(P_full, np.array(sorted(train_all), int))]

    print(f"[AL init] target={TARGET_MODE}  paid={len(train_all)} queried={len(train_q)}  "
          f"RMSE(train-{TRAIN_SET_LABEL},{TARGET_LABEL})={rtr0:.4f}  RMSE(val-oracle,{TARGET_LABEL})={rv0:.4f}")

    assert len(set(val_g.tolist()) & train_all) == 0
    assert len(set(val_g.tolist()) & train_q) == 0

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
        "step0_extra_edges": extra0,
        "paid_nodes": (paid_nodes_lsqr.copy() if paid_nodes_lsqr is not None else None),
        "U_paid": (U_paid_lsqr.copy() if U_paid_lsqr is not None else None),
    }

    subN = min(6000, Xs_feas.shape[0])
    if subN >= 2:
        sub_idx = rng.choice(np.arange(Xs_feas.shape[0]), size=subN, replace=False) if subN < Xs_feas.shape[0] else np.arange(subN)
        tree_sub = cKDTree(Xs_feas[sub_idx])
        dnn, _ = tree_sub.query(Xs_feas[sub_idx], k=2)
        base_h = float(np.median(dnn[:, 1])) if dnn.ndim == 2 and dnn.shape[1] >= 2 else 0.15
    else:
        base_h = 0.15
    base_h = max(base_h, 0.10)
    min_sep = MIN_SEP_MULT * base_h

    for it in range(AL_ITERS):
        mask_unl = np.ones(P_full.shape[0], bool)
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
        cand_g   = cand_g_all[gate]
        cand_pos = cand_pos_all[gate]
        d_to_train = d_to_train_all[gate]

        if cand_g.size < max(1, int(POOL_MIN_FRACTION * feasible_g.size)):
            old = min_sep
            min_sep *= ADAPT_MINSEP_DECAY
            print(f"[AL {it+1:02d}] relax min_sep: {old:.4g} -> {min_sep:.4g} (pool {cand_g.size})")
            gate = d_to_train_all >= min_sep
            cand_g   = cand_g_all[gate]
            cand_pos = cand_pos_all[gate]
            d_to_train = d_to_train_all[gate]

        if cand_g.size == 0:
            cand_g   = cand_g_all
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
                take_g.append(g); taken.append(p)
            else:
                dmin_now = np.min(np.linalg.norm(p - np.vstack(taken), axis=1))
                if dmin_now >= r_min:
                    take_g.append(g); taken.append(p)
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
            pending_plot["step_name"], pending_plot["model"],
            plot_g, P_full, P_feas, feas_pos, Xs_feas,
            xmu, xsd, pending_plot["ymu"], pending_plot["ysd"], normU,
            U_ref, U2a, Uface_a,
            pending_plot["train_all"], pending_plot["train_q"], val_g, anchor_g,
            d12s, d13s, D12s, D13s, D23s, mask2,
            d13f, d23f, D13f, D23f, D12f, maskf,
            p2, W2_anchor,
            score_map_for_this_step=score_map,
            cand_overlay_idx_for_this_step=cand_overlay_idx,
            X_tr=pending_plot["X_tr"], y_tr=pending_plot["y_tr"],
            X_val=pending_plot["X_val"], y_val=pending_plot["y_val"],
            rmse_tr_hist=rmse_tr_hist, rmse_val_hist=rmse_val_hist, mst_hist=mst_hist,
            n12=n12, n13=n13, n23=n23, feas_flat=feas_flat,
            step0_extra_edges=pending_plot.get("step0_extra_edges", None),
            step0_overlay_edges=step0_overlay_edges,
            target_mode=TARGET_MODE,
            W2anch_feas=W2anch_feas,
            target_label=TARGET_LABEL,
            symmetry_mode=SYMM_MODE,
            train_set_label=TRAIN_SET_LABEL,
            U_full_map=U_full_map,
            paid_nodes_step=pending_plot.get("paid_nodes", None),
            U_paid_step=pending_plot.get("U_paid", None)
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

            kd_cur = cKDTree(P_full[cur_nodes])
            dists, locs = kd_cur.query(P_full[pending], k=1)
            best_pos = int(np.argmin(np.atleast_1d(dists)))
            g_new = int(pending[best_pos])
            g_tgt = int(cur_nodes[int(np.atleast_1d(locs)[best_pos])])

            added_path = shortest_path_add_3d(
                train_set_all, g_new, g_tgt,
                P_full, n12, n13, n23, allowed_train_mask
            )
            newly_paid.extend(added_path)
            pending.pop(best_pos)

        newly_paid = [int(g) for g in newly_paid if allowed_train_mask[int(g)]]
        if USE_LSQR:
            teacher_query_force_at_nodes(newly_paid)

        train_all = set(train_set_all)

        assert len(set(val_g.tolist()) & train_all) == 0
        assert len(set(val_g.tolist()) & train_q) == 0

        train_nodes, y_phys_raw, paid_nodes_lsqr, U_paid_lsqr = _build_training_nodes_and_labels(
            LABEL_SOURCE, train_all, train_q,
            anchor_g, P_full, n12, n13, n23, feas_flat,
            Fx, Fy, Fz,
            graph_mode=LSQR_GRAPH_MODE,
            extra_edges=None
        )
        if train_nodes.size == 0:
            print(f"[AL {it+1:02d}] WARNING: no training labels produced; skipping update.")
            continue

        X_tr, y_tr, z_tr, ymu, ysd = make_train_xy_from_nodes(train_nodes, y_phys_raw, Xs_feas, feas_pos)
        if X_tr.shape[0] < 2:
            print(f"[AL {it+1:02d}] WARNING: too few points after sanitization; skipping update.")
            continue

        rtr_prev = eval_model_rmse(prev_m, X_tr, y_tr, prev_ymu, prev_ysd)
        rv_prev  = eval_model_rmse(prev_m, X_val, y_val_true, prev_ymu, prev_ysd)

        mprev_for_build = prev_m if (START_MODE == "warm") else None

        try:
            m_new = build_gpr(X_tr, z_tr, KMODE, KERNEL_SPEC, ARD, 800, m_prev=mprev_for_build)
        except Exception as e:
            print(f"[AL {it+1:02d}] WARN: GP build failed ({e}); retrying with safer settings.")
            gpflow.config.set_default_jitter(1e-2)
            m_new = build_gpr(X_tr, z_tr, KMODE, KERNEL_SPEC, ARD, 400, m_prev=mprev_for_build)

        rtr_new = eval_model_rmse(m_new, X_tr, y_tr, ymu, ysd)
        rv_new  = eval_model_rmse(m_new, X_val, y_val_true, ymu, ysd)

        mu_new_tr_z, std_new_tr_z = predict_mu_std_z(m_new, X_tr)
        ok_new = (np.all(np.isfinite(mu_new_tr_z)) and
                  np.all(np.isfinite(std_new_tr_z)) and
                  np.all(std_new_tr_z > 0) and
                  np.isfinite(rtr_new))

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
        mst_hist.append(mst_total_length(P_full, np.array(sorted(train_all), int)))

        step_next = f"{it+1:02d}"
        print(f"[AL {step_next}] {acc_flag}  target={TARGET_MODE}  paid={len(train_all)} "
              f"queried={len(train_q)}  RMSE(train-{TRAIN_SET_LABEL},{TARGET_LABEL})={rtr_acc:.4f}  RMSE(val-oracle,{TARGET_LABEL})={rv_acc:.4f}")

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
            "step0_extra_edges": None,
            "paid_nodes": (paid_nodes_lsqr.copy() if paid_nodes_lsqr is not None else None),
            "U_paid": (U_paid_lsqr.copy() if U_paid_lsqr is not None else None),
        }

    plot_all_figs_for_step(
        pending_plot["step_name"], pending_plot["model"],
        plot_g, P_full, P_feas, feas_pos, Xs_feas,
        xmu, xsd, pending_plot["ymu"], pending_plot["ysd"], normU,
        U_ref, U2a, Uface_a,
        pending_plot["train_all"], pending_plot["train_q"], val_g, anchor_g,
        d12s, d13s, D12s, D13s, D23s, mask2,
        d13f, d23f, D13f, D23f, D12f, maskf,
        p2, W2_anchor,
        score_map_for_this_step=None,
        cand_overlay_idx_for_this_step=np.array([], dtype=int),
        X_tr=pending_plot["X_tr"], y_tr=pending_plot["y_tr"],
        X_val=pending_plot["X_val"], y_val=pending_plot["y_val"],
        rmse_tr_hist=rmse_tr_hist, rmse_val_hist=rmse_val_hist, mst_hist=mst_hist,
        n12=n12, n13=n13, n23=n23, feas_flat=feas_flat,
        step0_extra_edges=None,
        step0_overlay_edges=step0_overlay_edges,
        target_mode=TARGET_MODE,
        W2anch_feas=W2anch_feas,
        target_label=TARGET_LABEL,
        symmetry_mode=SYMM_MODE,
        train_set_label=TRAIN_SET_LABEL,
        U_full_map=U_full_map,
        paid_nodes_step=pending_plot.get("paid_nodes", None),
        U_paid_step=pending_plot.get("U_paid", None)
    )

    fig, ax = plt.subplots(figsize=FIGSIZE_RMSE)
    its = np.arange(1, len(rmse_tr_hist) + 1)
    ax.plot(its, rmse_tr_hist, "-", lw=5, ms=80, label=f"Train ({TRAIN_SET_LABEL},{TARGET_LABEL})")
    ax.plot(its, rmse_val_hist, "-", lw=5, ms=80, label=f"Val (oracle,{TARGET_LABEL})")
    ax.set_xlabel("Iteration", fontsize=AXIS_LABEL_FZ)
    ax.set_ylabel(f"RMSE ({TARGET_LABEL})", fontsize=AXIS_LABEL_FZ)
    ax.set_xlim(0.5, len(its) + 0.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis="x", labelsize=TICK_LABEL_FZ, pad=6)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FZ, pad=6)
    ax.legend(frameon=False, fontsize=LEGEND_FZ, loc="upper right",
          markerscale=2.0, handlelength=2.0, handletextpad=0.6,
          labelspacing=0.5, borderpad=0.2, borderaxespad=0.2)
    savefig(fig, f"{TARGET_LABEL}_90_rmse_vs_iter")

    fig, ax = plt.subplots(figsize=FIGSIZE_RMSE)
    steps = np.arange(len(mst_hist))
    ax.plot(steps, mst_hist, "-o", lw=5, ms=16, mfc="none")
    ax.set_xlabel("AL step", fontsize=AXIS_LABEL_FZ)
    ax.set_ylabel("MST length", fontsize=AXIS_LABEL_FZ)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis="x", labelsize=TICK_LABEL_FZ, pad=6)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FZ, pad=6)
    savefig(fig, "Umb_91_mst_vs_iter")

    steps_out = np.arange(len(rmse_tr_hist), dtype=int)

    metrics_out = np.column_stack([
        steps_out,
        np.asarray(rmse_tr_hist, float),
        np.asarray(rmse_val_hist, float),
        np.asarray(mst_hist, float),
    ])

    np.savetxt(
        FIG_DIR / "al_metrics.csv",
        metrics_out,
        header="step rmse_train rmse_val mst_length",
        fmt=["%d", "%.10f", "%.10f", "%.10f"],
    )

    plt.close("all")

if __name__ == "__main__":
    main()
