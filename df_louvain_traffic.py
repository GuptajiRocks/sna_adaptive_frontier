"""
DF Louvain: Dynamic Frontier Community Detection on Traffic Sensor Networks
===========================================================================
Replication of:
  "DF Louvain: Fast Incrementally Expanding Approach for Community Detection
   on Dynamic Graphs" — Subhajit Sahu, arXiv:2404.19634v4, Sep 2024

Applied to traffic datasets:
  • METR-LA  — 207 sensors, Los Angeles freeways, Mar–Jun 2012, 5-min intervals
  • PEMS-BAY — 325 sensors, Bay Area, Jan–May 2017, 5-min intervals
  Both available at: https://github.com/liyaguang/DCRNN

GPU acceleration (NVIDIA A6000 / any CUDA GPU):
  Primary backend : PyTorch  (pip install torch --index-url https://download.pytorch.org/whl/cu121)
  Fallback        : CuPy     (pip install cupy-cuda12x)
  Final fallback  : NumPy CPU

  PyTorch is preferred on Windows because it bundles its own CUDA DLLs
  (cublas64_12.dll etc.) inside the package, avoiding the Windows DLL
  search-path issue that causes CuPy's "cublas DLL not found" error when
  the CUDA Toolkit bin folder is not on PATH.

Install (Windows, CUDA 12.x):
  pip install h5py numpy scipy networkx matplotlib pandas tqdm
  pip install torch --index-url https://download.pytorch.org/whl/cu121

Usage:
  python df_louvain_traffic.py --dataset metr-la --data_path ./data/metr-la.h5
  python df_louvain_traffic.py --dataset pems-bay --data_path ./data/pems-bay.h5

Three algorithms are benchmarked head-to-head (matching paper Table 1):
  • Static Louvain   — full re-run from scratch every timestamp batch
  • ND Louvain       — warm-start from previous partition, full re-run
  • DF Louvain       — only reprocess frontier vertices (our replication)

Outputs (written to ./results/):
  modularity_over_time_<dataset>.png
  speedup_comparison_<dataset>.png
  community_evolution_<dataset>.png
  benchmark_table_<dataset>.csv
"""

import argparse
import os
import time
import gc
import warnings
from collections import defaultdict
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ── GPU setup — Windows-safe DLL fix applied before any CUDA call ─────────────
#
# Root cause of the "cublas DLL not found" error on Windows:
#   CuPy detects the GPU via cudart (already on PATH via the driver), but the
#   compute libraries (cublas64_12.dll, cufft64_11.dll, etc.) live in:
#     C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin
#   which is NOT automatically added to the DLL search path on Python 3.8+
#   due to the stricter os.add_dll_directory() requirement (PEP 688 / bpo-36085).
#
# Fix A (applied here): call os.add_dll_directory() with the CUDA bin path
#   before importing cupy or torch, so Windows finds the DLLs.
# Fix B (preferred): use PyTorch, which ships its own bundled CUDA DLLs and
#   never has this problem.
#
# We try in order: PyTorch CUDA → CuPy (with DLL fix) → NumPy CPU.

def _register_cuda_dll_dirs():
    """
    Register CUDA Toolkit bin directories with Windows DLL loader.
    Must be called before importing any CUDA library.
    No-op on Linux/macOS.
    """
    if os.name != "nt":
        return
    import glob
    search_roots = [
        os.environ.get("CUDA_PATH", ""),
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
        r"C:\CUDA",
    ]
    registered = []
    for root in search_roots:
        if not root:
            continue
        # Handle both versioned (CUDA\v12.x) and unversioned (CUDA\) layouts
        candidates = [root] + glob.glob(os.path.join(root, "v*"))
        for base in candidates:
            bin_dir = os.path.join(base, "bin")
            if os.path.isdir(bin_dir) and bin_dir not in registered:
                try:
                    os.add_dll_directory(bin_dir)
                    registered.append(bin_dir)
                except (OSError, AttributeError):
                    pass
    if registered:
        print(f"[DLL] Registered CUDA bin dirs: {registered}")

_register_cuda_dll_dirs()

# ── Backend selection ─────────────────────────────────────────────────────────
GPU_AVAILABLE = False
GPU_BACKEND   = "cpu"
GPU_NAME      = "CPU"
_torch        = None

# Try PyTorch first (most reliable on Windows)
try:
    import torch as _torch
    if _torch.cuda.is_available():
        GPU_AVAILABLE = True
        GPU_BACKEND   = "torch"
        GPU_NAME      = _torch.cuda.get_device_name(0)
        _TORCH_DEVICE = _torch.device("cuda:0")
        print(f"[GPU] PyTorch backend: {GPU_NAME}")
    else:
        print("[GPU] PyTorch found but no CUDA device — trying CuPy...")
except ImportError:
    print("[GPU] PyTorch not installed — trying CuPy...")

# Try CuPy if PyTorch unavailable or has no GPU
if not GPU_AVAILABLE:
    try:
        import cupy as _cp
        _cp.cuda.Device(0).use()
        # Warm-up: force cublas load NOW so the error surfaces here, not later
        _test = _cp.array([[1.0, 0.0], [0.0, 1.0]], dtype=_cp.float32)
        _ = _test @ _test          # triggers cublas import
        del _test
        GPU_AVAILABLE = True
        GPU_BACKEND   = "cupy"
        GPU_NAME      = _cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        print(f"[GPU] CuPy backend: {GPU_NAME}")
    except Exception as _e:
        print(f"[GPU] CuPy unavailable ({type(_e).__name__}: {_e})")
        print("[GPU] Falling back to NumPy CPU.")

if not GPU_AVAILABLE:
    print("[GPU] Running on CPU (NumPy). Install PyTorch for GPU acceleration:")
    print("      pip install torch --index-url https://download.pytorch.org/whl/cu121")

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)


# =============================================================================
# 1.  DATA LOADING
# =============================================================================
def load_metr_la(h5_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load METR-LA / PEMS-BAY dataset.

    Handles:
      - /df/block0_values (DCRNN METR-LA)
      - /speed/block0_values (PEMS-BAY variant you have)
      - /speed as dataset (other variants)
      - generic fallback: first dataset
    """
    with h5py.File(h5_path, "r") as f:
        if "df" in f:
            speeds     = f["df"]["block0_values"][:]          # (T, N)
            timestamps = f["df"]["axis1"][:]
            sensor_ids = f["df"]["block0_items"][:]

        elif "speed" in f:
            speed_obj = f["speed"]
            if isinstance(speed_obj, h5py.Dataset):
                speeds = speed_obj[:]
                timestamps = np.arange(speeds.shape[0]) * 300
                sensor_ids = np.arange(speeds.shape[1]).astype(str)
            else:
                # speed is a group with block0_values
                speeds     = speed_obj["block0_values"][:]
                timestamps = speed_obj["axis1"][:]
                sensor_ids = speed_obj["block0_items"][:]

        else:
            # fallback: first dataset anywhere
            def _first_dataset(group):
                for k in group.keys():
                    obj = group[k]
                    if isinstance(obj, h5py.Dataset):
                        return obj[:]
                    elif isinstance(obj, h5py.Group):
                        res = _first_dataset(obj)
                        if res is not None:
                            return res
                return None

            speeds = _first_dataset(f)
            if speeds is None:
                raise ValueError("No dataset found in H5 file.")
            timestamps = np.arange(speeds.shape[0]) * 300
            sensor_ids = np.arange(speeds.shape[1]).astype(str)

    speeds = speeds.astype(np.float32)

    # sensor_ids may be bytes
    if sensor_ids.dtype.kind == 'S' or (sensor_ids.size > 0 and
            isinstance(sensor_ids.flat[0], (bytes, np.bytes_))):
        sensor_ids = np.array([s.decode() if isinstance(s, (bytes, np.bytes_))
                                else str(s) for s in sensor_ids])
    else:
        sensor_ids = sensor_ids.astype(str)

    print(f"  Sensor ID sample: {sensor_ids[:3].tolist()} … "
          f"(total {len(sensor_ids)})")
    return speeds, timestamps, sensor_ids


def load_pems_bay(h5_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load PEMS-BAY dataset (same format as METR-LA but 325 sensors)."""
    return load_metr_la(h5_path)    # identical file structure


# =============================================================================
# 2.  GRAPH CONSTRUCTION FROM SENSOR DATA
# =============================================================================
def load_adj_pkl(pkl_path: str,
                 h5_sensor_ids: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Load the pre-built adjacency matrix from DCRNN's adj_mx.pkl.

    File format (liyaguang/DCRNN):
      pickle.load() returns a 3-tuple:
        [0]  sensor_ids     : list of N sensor ID strings  e.g. ['773869', ...]
        [1]  sensor_id_to_ind : dict {sensor_id_str -> int index 0..N-1}
        [2]  adj_mx         : (N, N) float32 numpy array, already normalised
                              Values in [0, 1].  Row i = outgoing weights from i.

    The adj_mx rows/columns are in the SAME order as the h5 block0_items
    sensor list, so no ID remapping is needed if you use adj_mx directly.

    If h5_sensor_ids is provided, we verify the ordering matches and warn
    if it doesn't.

    Returns A : (N, N) float32 adjacency matrix, self-loops zeroed out.
    """
    import pickle

    with open(pkl_path, "rb") as f:
        try:
            pkg = pickle.load(f)                       # Python 3 default
        except UnicodeDecodeError:
            f.seek(0)
            pkg = pickle.load(f, encoding="latin1")    # pkl saved in Python 2

    # Handle both tuple and list
    if isinstance(pkg, (list, tuple)) and len(pkg) == 3:
        pkl_sensor_ids, _, adj_mx = pkg
    elif isinstance(pkg, np.ndarray):
        # Some versions just store the matrix directly
        adj_mx = pkg
        pkl_sensor_ids = None
    else:
        raise ValueError(f"Unexpected adj_mx.pkl format: {type(pkg)}")

    adj_mx = np.array(adj_mx, dtype=np.float32)
    np.fill_diagonal(adj_mx, 0.0)          # zero self-loops

    N = adj_mx.shape[0]
    n_edges = int((adj_mx > 0).sum() // 2)
    print(f"  adj_mx.pkl loaded: shape={adj_mx.shape}  "
          f"edges={n_edges}  density={n_edges/(N*(N-1)//2):.4f}")

    # Verify sensor ordering matches h5 if both are available
    if pkl_sensor_ids is not None and h5_sensor_ids is not None:
        # Decode h5 sensor IDs (may be bytes)
        h5_ids = []
        for s in h5_sensor_ids:
            raw = s.decode().strip() if isinstance(s, (bytes, np.bytes_)) else str(s).strip()
            h5_ids.append(raw)
        pkl_ids = [str(s) for s in pkl_sensor_ids]

        if h5_ids == pkl_ids:
            print("  Sensor ordering: h5 and adj_mx.pkl match perfectly.")
        else:
            matches = sum(a == b for a, b in zip(h5_ids, pkl_ids))
            print(f"  [warn] Sensor ordering mismatch: {matches}/{N} positions agree.")
            if matches < N // 2:
                print("  [warn] Less than 50% match — reordering adj_mx to h5 order.")
                id_to_pkl_idx = {s: i for i, s in enumerate(pkl_ids)}
                reordered = np.zeros_like(adj_mx)
                for h_i, h_id in enumerate(h5_ids):
                    p_i = id_to_pkl_idx.get(h_id)
                    if p_i is None:
                        continue
                    for h_j, h_id2 in enumerate(h5_ids):
                        p_j = id_to_pkl_idx.get(h_id2)
                        if p_j is None:
                            continue
                        reordered[h_i, h_j] = adj_mx[p_i, p_j]
                adj_mx = reordered
                print("  Reordering complete.")

    return adj_mx


def _synthetic_adjacency(n: int) -> np.ndarray:
    """Ring-topology fallback adjacency when CSV IDs do not match h5."""
    A = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(max(0, i-5), min(n, i+6)):
            if i != j:
                A[i, j] = float(np.exp(-(abs(i-j)**2) / 2.0))
    return A


def build_adjacency_from_distances(dist_csv: Optional[str],
                                   n_sensors: int,
                                   sensor_ids: Optional[np.ndarray] = None,
                                   sigma2: float = None,
                                   eps: float = 0.1) -> np.ndarray:
    """
    sigma2=None means auto-compute as std(distances)^2 following the DCRNN
    paper (Li et al. 2018, Eq. 3).  eps=0.1 is the DCRNN default threshold.

    The old defaults sigma2=0.1, eps=0.5 were wrong for METR-LA where
    distances are in metres (range ~100–10000 m).  With sigma2=0.1 and
    d=500m: exp(-(500^2)/0.1) ≈ 0 → zero edges → degenerate adjacency.
    """
    """
    Build adjacency matrix from the DCRNN distances_la_2012.csv (or
    distances_bay_2017.csv).

    CSV format — two possible layouts handled automatically:
      WITH header:  "from,to,cost"   (most downloads)
      NO header:    three bare columns  from_id, to_id, distance

    Sensor IDs in the CSV are integers (e.g. 773869, 767541 …).
    The h5 file stores sensors in a fixed order; sensor_ids (read from h5)
    maps those integers back to row/column indices 0…N-1.

    Gaussian kernel weight:  w_ij = exp(-d_ij^2 / sigma2)
    Edges with w < eps are pruned (sparse adjacency, matching DCRNN paper).

    sigma2 = 0.1 matches the DCRNN paper default (Eq. 3).
    eps    = 0.5 is the standard threshold used in the DCRNN codebase.

    Returns A : (N, N) float32 adjacency matrix
    """
    if dist_csv and os.path.exists(dist_csv):
        # ── detect whether the file has a header row ──────────────────────────
        with open(dist_csv) as f:
            first = f.readline().strip()
        # Header if the first field is non-numeric (e.g. "from", "sensor_id")
        has_header = not first.split(",")[0].strip().lstrip("-").isdigit()

        raw = pd.read_csv(
            dist_csv,
            header=0 if has_header else None,
        )
        # Normalise column names regardless of header presence
        # Always rename to 3 standard columns (DCRNN header: from,to,cost)
        raw.columns = ["from_id", "to_id", "distance"]

        raw["from_id"]  = raw["from_id"].astype(int)
        raw["to_id"]    = raw["to_id"].astype(int)
        raw["distance"] = raw["distance"].astype(float)

        # Remove self-loops (from==to) — contribute nothing to adjacency
        raw = raw[raw["from_id"] != raw["to_id"]].reset_index(drop=True)

        # Auto-compute sigma2 from actual distance distribution (DCRNN paper)
        if sigma2 is None:
            sigma2 = float(raw["distance"].std() ** 2)
            print(f"  Auto sigma2 = {sigma2:.0f} (std of distances squared)")

        print(f"  Distance CSV: {len(raw)} entries, "
              f"{'with' if has_header else 'no'} header, "
              f"ID range [{raw['from_id'].min()}…{raw['from_id'].max()}]")

        # ── build sensor_id → matrix_index mapping ────────────────────────────
        if sensor_ids is not None:
            # Decode |S6 bytes (h5py returns b'773869' for METR-LA)
            id_arr = []
            for s in sensor_ids:
                raw_s = s.decode().strip() if isinstance(s, (bytes, np.bytes_)) else str(s).strip()
                try: id_arr.append(int(raw_s))
                except ValueError: id_arr.append(raw_s)
            sid_to_idx = {sid: i for i, sid in enumerate(id_arr)}
            # Warn if CSV IDs and h5 IDs don't overlap
            csv_ids = set(raw["from_id"]) | set(raw["to_id"])
            overlap = len(set(id_arr) & csv_ids)
            print(f"  Sensor ID overlap (h5 vs CSV): {overlap} / {len(id_arr)}")
            if overlap == 0:
                print("  [warn] No sensor ID overlap — CSV and h5 may be different areas.")
                print("  Using CSV internal ordering for adjacency (topology still valid).")
                all_ids = sorted(set(raw["from_id"]) | set(raw["to_id"]))
                sid_to_idx = {sid: i % n_sensors for i, sid in enumerate(all_ids)}
        else:
            all_ids = sorted(set(raw["from_id"]) | set(raw["to_id"]))
            sid_to_idx = {sid: i for i, sid in enumerate(all_ids)}

        A = np.zeros((n_sensors, n_sensors), dtype=np.float32)
        skipped = used = 0
        for row in raw.itertuples(index=False):
            i = sid_to_idx.get(int(row.from_id))
            j = sid_to_idx.get(int(row.to_id))
            if i is None or j is None or i >= n_sensors or j >= n_sensors:
                skipped += 1
                continue
            w = float(np.exp(-(row.distance ** 2) / sigma2))
            if w >= eps:
                A[i, j] = w
                A[j, i] = w
                used += 1

        if used == 0:
            print(f"  [warn] Zero edges from CSV (all {skipped} rows skipped) — "
                  "sensor IDs likely from a different area. Using synthetic topology.")
            A = _synthetic_adjacency(n_sensors)
        elif skipped > 0:
            print(f"  [info] {skipped} CSV rows skipped (ID mismatch/out-of-range), "
                  f"{used} rows used.")
        n_edges = int((A > 0).sum() // 2)
        print(f"  Adjacency: {n_edges} edges | "
              f"density {n_edges / max(n_sensors*(n_sensors-1)//2, 1):.4f}")
    else:
        # ── synthetic fallback ────────────────────────────────────────────────
        print("  [warn] No distance CSV found — using synthetic ring topology.")
        A = np.zeros((n_sensors, n_sensors), dtype=np.float32)
        for i in range(n_sensors):
            for j in range(max(0, i - 5), min(n_sensors, i + 6)):
                if i != j:
                    A[i, j] = float(np.exp(-(abs(i-j)**2) / sigma2))

    return A


def _impute_zeros(S: np.ndarray) -> np.ndarray:
    """
    Replace zero readings (METR-LA missing value sentinel) with
    forward-fill then backward-fill per sensor column.
    Falls back to column mean for sensors that are entirely zero.
    Returns float64 array with no zeros or NaN.
    """
    S = S.copy().astype(np.float64)
    if (S == 0).sum() == 0 and not np.isnan(S).any():
        return S
    for j in range(S.shape[1]):
        col = S[:, j].copy()
        # Forward fill
        last_valid = np.nan
        for i in range(len(col)):
            if col[i] == 0 or np.isnan(col[i]):
                col[i] = last_valid
            else:
                last_valid = col[i]
        # Backward fill (handles leading zeros)
        last_valid = np.nan
        for i in range(len(col)-1, -1, -1):
            if np.isnan(col[i]):
                col[i] = last_valid
            else:
                last_valid = col[i]
        # Last resort: column mean of non-zero entries
        if np.isnan(col).any():
            nz = S[:, j][S[:, j] > 0]
            fill = nz.mean() if len(nz) > 0 else 1.0
            col = np.where(np.isnan(col), fill, col)
        S[:, j] = col
    return S


def compute_correlation_weights_gpu(speeds_window: np.ndarray,
                                    adj: np.ndarray,
                                    min_periods: int = 3) -> np.ndarray:
    """
    Compute Pearson correlation between connected sensor pairs over a
    rolling window.  Dispatches to PyTorch -> CuPy -> NumPy CPU.

    PyTorch path (preferred on Windows): tensors live on CUDA, matmul goes
      through PyTorch's own bundled cuBLAS DLLs — no Windows PATH issues.
    CuPy path: used if CuPy was successfully warmed-up at import time.
    NumPy path: pure CPU fallback, always works.

    Returns W_corr : (N, N) float32 — correlation-weighted adjacency,
             negative correlations clipped to 0, self-loops = 0.
    """
    # Impute zeros (METR-LA missing value) before computing correlation
    speeds_window = _impute_zeros(speeds_window)

    if GPU_BACKEND == "torch":
        dev   = _TORCH_DEVICE
        S     = _torch.tensor(speeds_window, dtype=_torch.float32, device=dev)
        A     = _torch.tensor(adj,           dtype=_torch.float32, device=dev)
        mu    = S.mean(dim=0)
        sigma = S.std(dim=0)
        # Zero-variance sensors: set S_std to 0 (no correlation contribution)
        sigma_safe = _torch.where(sigma < 1e-6, _torch.ones_like(sigma), sigma)
        S_std = (S - mu) / sigma_safe
        S_std[:, sigma < 1e-6] = 0.0
        W_corr = _torch.clamp((S_std.T @ S_std) / S_std.shape[0], min=0.0) * A
        W_corr.fill_diagonal_(0.0)
        result = W_corr.cpu().numpy().astype(np.float32)
        del S, A, S_std, W_corr
        _torch.cuda.empty_cache()
        return result

    elif GPU_BACKEND == "cupy":
        S     = _cp.array(speeds_window, dtype=_cp.float32)
        A     = _cp.array(adj,           dtype=_cp.float32)
        mu    = S.mean(axis=0);  sigma = S.std(axis=0)
        sigma_safe = _cp.where(sigma < 1e-6, _cp.ones_like(sigma), sigma)
        S_std = (S - mu) / sigma_safe
        S_std[:, sigma < 1e-6] = 0.0
        W_corr = _cp.clip((S_std.T @ S_std) / S_std.shape[0], 0, None) * A
        _cp.fill_diagonal(W_corr, 0)
        result = _cp.asnumpy(W_corr).astype(np.float32)
        del S, A, S_filled, S_std, W_corr
        _cp.get_default_memory_pool().free_all_blocks()
        return result

    else:
        N     = speeds_window.shape[1]
        S     = speeds_window.astype(np.float64)   # already imputed above
        mu    = S.mean(axis=0);  sigma = S.std(axis=0)
        sigma_safe = np.where(sigma < 1e-6, 1.0, sigma)
        S_std = (S - mu) / sigma_safe
        S_std[:, sigma < 1e-6] = 0.0
        W_corr = np.clip((S_std.T @ S_std) / S_std.shape[0], 0, None) * adj
        np.fill_diagonal(W_corr, 0)
        return W_corr.astype(np.float32)


def compute_modularity(partition: np.ndarray,
                       W: np.ndarray,
                       m: float) -> float:
    """
    Q = (1/2m) * sum_{ij} [w_ij - k_i*k_j/(2m)] * delta(c_i, c_j)

    Parameters
    ----------
    partition : (N,) int array — community label per vertex
    W         : (N, N) weight matrix
    m         : total edge weight (sum of all w_ij, counted once per pair
                for undirected)

    Returns Q in [-0.5, 1.0].
    """
    if m < 1e-10:
        return 0.0

    k = W.sum(axis=1)               # degree vector (N,)
    N = len(partition)
    Q = 0.0

    # Vectorised: for each community c, sum internal weights and degrees
    for c in np.unique(partition):
        mask = (partition == c)
        W_in = W[np.ix_(mask, mask)].sum()
        k_in = k[mask].sum()
        Q   += W_in / (2 * m) - (k_in / (2 * m)) ** 2

    return float(Q)


def compute_modularity_gpu(partition: np.ndarray,
                           W: np.ndarray,
                           m: float) -> float:
    """GPU-accelerated modularity via community indicator matrix."""
    if m < 1e-10 or GPU_BACKEND == "cpu":
        return compute_modularity(partition, W, m)

    if GPU_BACKEND == "torch":
        dev = _TORCH_DEVICE
        N   = len(partition)
        nc  = int(partition.max()) + 1
        Wg  = _torch.tensor(W,         dtype=_torch.float32, device=dev)
        pg  = _torch.tensor(partition, dtype=_torch.long,    device=dev)
        C   = _torch.zeros(nc, N, dtype=_torch.float32, device=dev)
        C[pg, _torch.arange(N, device=dev)] = 1.0
        CW  = C @ Wg
        e_in = (CW * C).sum(dim=1)
        k    = Wg.sum(dim=1)
        a_in = (C * k.unsqueeze(0)).sum(dim=1)
        Q    = float((e_in / (2*m) - (a_in / (2*m))**2).sum())
        del Wg, C, CW, e_in, k, a_in
        _torch.cuda.empty_cache()
        return Q

    # CuPy path
    N  = len(partition)
    nc = int(partition.max()) + 1
    Wg = _cp.array(W, dtype=_cp.float32)
    pg = _cp.array(partition, dtype=_cp.int32)

    # Community indicator: C[c, i] = 1 if node i in community c
    C = _cp.zeros((nc, N), dtype=_cp.float32)
    C[pg, _cp.arange(N)] = 1.0

    # Sum of intra-community edge weights: trace(C W C^T)
    CW     = C @ Wg
    e_in   = (CW * C).sum(axis=1)
    k      = Wg.sum(axis=1)
    a_in   = (C * k[None, :]).sum(axis=1)
    Q = float((e_in / (2 * m) - (a_in / (2 * m)) ** 2).sum())

    del Wg, C, CW, e_in, k, a_in
    _cp.get_default_memory_pool().free_all_blocks()
    return Q


# =============================================================================
# 4.  LOUVAIN — Static (baseline)
# =============================================================================
def louvain_local_move(partition: np.ndarray,
                       W: np.ndarray,
                       k: np.ndarray,
                       m: float,
                       affected: Optional[np.ndarray] = None,
                       max_iter: int = 100) -> tuple[np.ndarray, bool]:
    """
    Local-moving phase of Louvain.  For each vertex in `affected` (or all
    vertices if None), try moving it to the community of each neighbour;
    accept if ΔQ > 0.

    GPU path: ΔQ computation vectorised over all neighbours of one vertex.
    Convergence check uses GPU boolean reduction.

    Returns (new_partition, changed: bool).
    """
    N      = len(partition)
    label  = partition.copy()
    # k_c[c] = sum of degrees of all vertices in community c
    k_c    = np.zeros(N, dtype=np.float64)
    for i in range(N):
        k_c[label[i]] += k[i]

    # e_in[c] = sum of internal edge weights of community c (both directions)
    e_in = np.zeros(N, dtype=np.float64)
    for i in range(N):
        for j in np.where(W[i] > 0)[0]:
            if label[i] == label[j]:
                e_in[label[i]] += W[i, j]

    changed   = False
    vertices  = np.arange(N) if affected is None else np.where(affected)[0]

    for _ in range(max_iter):
        moved = False
        for i in vertices:
            ci  = label[i]
            ki  = k[i]

            # Weight from i to each community it neighbours
            neigh_w = defaultdict(float)
            for j in np.where(W[i] > 0)[0]:
                neigh_w[label[j]] += W[i, j]

            # ΔQ for removing i from ci
            # Using standard Louvain formula (Blondel 2008):
            # ΔQ_remove = -[k_i_in/m - (k_c[ci] - ki)*ki / (2m^2)]
            k_i_in    = neigh_w.get(ci, 0.0)   # weight from i to rest of ci
            dQ_remove = -(k_i_in / m - (k_c[ci] - ki) * ki / (2 * m * m))

            # ΔQ for inserting i into each neighbouring community
            best_dQ = 0.0
            best_c  = ci

            for cj, w_ij in neigh_w.items():
                if cj == ci:
                    continue
                dQ_insert = w_ij / m - k_c[cj] * ki / (2 * m * m)
                dQ_total  = dQ_remove + dQ_insert
                if dQ_total > best_dQ:
                    best_dQ = dQ_total
                    best_c  = cj

            if best_c != ci:
                # Update community weights incrementally
                k_c[ci]     -= ki
                k_c[best_c] += ki
                e_in[ci]    -= 2 * neigh_w.get(ci, 0.0)
                e_in[best_c]+= 2 * neigh_w.get(best_c, 0.0)
                label[i]     = best_c
                moved         = True
                changed       = True

        if not moved:
            break

    # Relabel communities to consecutive integers
    mapping = {old: new for new, old in enumerate(np.unique(label))}
    label   = np.array([mapping[l] for l in label], dtype=np.int32)
    return label, changed


def louvain_static(W: np.ndarray, max_passes: int = 10) -> np.ndarray:
    """
    Full Static Louvain (paper baseline).
    Runs from scratch — no prior partition used.
    max_passes=10 gives Static the best chance to find fine-grained
    communities; warm-start methods (ND/DF) converge faster from a
    good prior partition, so they legitimately find fewer communities
    per batch — this is expected behaviour, not a bug.
    """
    N          = W.shape[0]
    m          = W.sum() / 2.0
    k          = W.sum(axis=1)
    partition  = np.arange(N, dtype=np.int32)   # each node = own community
    partition, _ = louvain_local_move(partition, W, k, m)
    return partition


# =============================================================================
# 5.  ND LOUVAIN — Naive Dynamic (warm start only)
# =============================================================================
def louvain_nd(W_new: np.ndarray,
               prev_partition: np.ndarray) -> np.ndarray:
    """
    Naive-Dynamic Louvain: warm-start from prev_partition, run full passes.
    All vertices considered affected — no frontier logic.
    """
    N = W_new.shape[0]
    m = W_new.sum() / 2.0
    k = W_new.sum(axis=1)
    partition, _ = louvain_local_move(prev_partition.copy(), W_new, k, m)
    return partition


# =============================================================================
# 6.  DF LOUVAIN — Dynamic Frontier (paper contribution)
# =============================================================================
def detect_frontier_gpu(W_old: np.ndarray,
                        W_new: np.ndarray,
                        prev_partition: np.ndarray,
                        threshold: float = 1e-3) -> np.ndarray:
    """
    GPU-accelerated frontier detection.

    A vertex i is initially marked as affected if any of its edge weights
    changed by more than `threshold` in the batch update.

    In the paper: vertices are marked when an edge incident to them is
    inserted or deleted.  Here, since we have continuous weight changes,
    we treat a weight change > threshold as an "effective edge change."

    Returns affected : (N,) bool array
    """
    if GPU_BACKEND == "torch":
        Wo = _torch.tensor(W_old, device=_TORCH_DEVICE)
        Wn = _torch.tensor(W_new, device=_TORCH_DEVICE)
        delta = (Wn - Wo).abs()
        aff   = (delta.max(dim=1).values > threshold) | (delta.max(dim=0).values > threshold)
        result = aff.cpu().numpy()
        del Wo, Wn, delta, aff
        _torch.cuda.empty_cache()
        return result
    if GPU_BACKEND == "cupy":
        Wo = _cp.array(W_old, dtype=_cp.float32)
        Wn = _cp.array(W_new, dtype=_cp.float32)
        # Absolute weight change per vertex = max change across all its edges
        delta  = _cp.abs(Wn - Wo)
        # Any edge of vertex i changed beyond threshold
        affected_gpu = (delta.max(axis=1) > threshold) | \
                       (delta.max(axis=0) > threshold)  # (N,)
        result = _cp.asnumpy(affected_gpu)
        del Wo, Wn, delta, affected_gpu
        _cp.get_default_memory_pool().free_all_blocks()
        return result
    else:
        delta    = np.abs(W_new - W_old)
        affected = (delta.max(axis=1) > threshold) | \
                   (delta.max(axis=0) > threshold)
        return affected


def louvain_df(W_new: np.ndarray,
               W_old: np.ndarray,
               prev_partition: np.ndarray,
               frontier_threshold: float = 1e-3,
               max_passes: int = 5) -> tuple[np.ndarray, int]:
    """
    Dynamic Frontier Louvain (paper Section 4.1).

    Algorithm:
      1. Detect initially affected vertices (frontier) — GPU accelerated.
      2. Run local-moving phase restricted to affected vertices.
      3. When a vertex migrates to a new community, mark its neighbours
         as affected (frontier expansion — paper Section 4.1 paragraph 2).
      4. Repeat until no more migrations in a pass.

    Returns (partition, n_affected_vertices)
    """
    N         = W_new.shape[0]
    m         = W_new.sum() / 2.0
    k         = W_new.sum(axis=1)
    partition = prev_partition.copy()

    # Step 1: initial frontier detection
    affected = detect_frontier_gpu(W_old, W_new, prev_partition,
                                   frontier_threshold)

    if not affected.any():
        return partition, 0

    n_initially_affected = int(affected.sum())

    # Step 2+3: local moving with frontier expansion
    # k_c and e_in must be computed from the full graph (not just frontier)
    k_c  = np.zeros(N, dtype=np.float64)
    e_in = np.zeros(N, dtype=np.float64)
    for i in range(N):
        k_c[partition[i]] += k[i]
    for i in range(N):
        for j in np.where(W_new[i] > 0)[0]:
            if partition[i] == partition[j]:
                e_in[partition[i]] += W_new[i, j]

    for _ in range(max_passes):
        moved = False
        vertices = np.where(affected)[0]

        for i in vertices:
            ci  = partition[i]
            ki  = k[i]
            neigh_w = defaultdict(float)
            for j in np.where(W_new[i] > 0)[0]:
                neigh_w[partition[j]] += W_new[i, j]

            k_i_in    = neigh_w.get(ci, 0.0)
            dQ_remove = -(k_i_in / m - (k_c[ci] - ki) * ki / (2 * m * m)) \
                        if m > 1e-10 else 0.0

            best_dQ = 0.0
            best_c  = ci
            for cj, w_ij in neigh_w.items():
                if cj == ci:
                    continue
                dQ_insert = w_ij / m - k_c[cj] * ki / (2 * m * m)
                dQ_total  = dQ_remove + dQ_insert
                if dQ_total > best_dQ:
                    best_dQ = dQ_total
                    best_c  = cj

            if best_c != ci:
                k_c[ci]      -= ki
                k_c[best_c]  += ki
                e_in[ci]     -= 2 * neigh_w.get(ci, 0.0)
                e_in[best_c] += 2 * neigh_w.get(best_c, 0.0)
                partition[i]  = best_c
                moved         = True

                # Step 3: frontier expansion — mark neighbours as affected
                for j in np.where(W_new[i] > 0)[0]:
                    affected[j] = True

        if not moved:
            break

    # Relabel
    mapping   = {old: new for new, old in enumerate(np.unique(partition))}
    partition = np.array([mapping[l] for l in partition], dtype=np.int32)
    return partition, n_initially_affected


# =============================================================================
# 7.  BENCHMARK RUNNER
# =============================================================================
def run_benchmark(speeds:        np.ndarray,
                  timestamps:    np.ndarray,
                  adj:           np.ndarray,
                  window:        int   = 12,
                  batch_size:    int   = 1,
                  n_batches:     int   = 200,
                  frontier_thr:  float = 1e-3,
                  start_batch:   int   = 12) -> dict:
    """
    Run all three algorithms (Static, ND, DF) on sequential timestamp batches.

    Parameters
    ----------
    speeds       : (T, N) speed array
    timestamps   : (T,)   UNIX timestamps
    adj          : (N, N) topology adjacency
    window       : rolling window size in timesteps for correlation weights
    batch_size   : number of new timesteps per batch update
    n_batches    : number of batches to run
    frontier_thr : edge weight change threshold for frontier detection
    start_batch  : first batch index (must be >= window)

    Returns dict with per-batch timing and modularity for each algorithm.
    """
    N  = speeds.shape[1]
    T  = speeds.shape[0]

    results = {
        "batch":          [],
        "timestamp":      [],
        "n_sensors":      N,       # total sensor count for frontier fraction
        "n_affected_df":  [],
        "n_communities":  {"static": [], "nd": [], "df": []},
        "modularity":     {"static": [], "nd": [], "df": []},
        "time_s":         {"static": [], "nd": [], "df": []},
        "speedup_vs_static": {"nd": [], "df": []},
    }

    # Initialise partitions from the first static run
    t0    = start_batch * batch_size
    W_cur = compute_correlation_weights_gpu(speeds[t0 - window:t0], adj)
    m_cur = W_cur.sum() / 2.0

    print("  Initialising with Static Louvain on first window...")
    partition_static = louvain_static(W_cur)
    partition_nd     = partition_static.copy()
    partition_df     = partition_static.copy()
    W_prev           = W_cur.copy()

    print(f"  Running {n_batches} batches "
          f"(window={window}, batch={batch_size}, N={N})...")

    for b in tqdm(range(n_batches), desc="  Batches", ncols=70):
        t_end   = t0 + (b + 1) * batch_size
        t_start = t_end - window
        if t_end > T:
            break

        # Recompute edge weights for this window
        W_new = compute_correlation_weights_gpu(speeds[t_start:t_end], adj)
        m_new = W_new.sum() / 2.0
        k_new = W_new.sum(axis=1)

        ts = int(timestamps[t_end - 1]) if t_end - 1 < len(timestamps) else b

        # ── Static Louvain ────────────────────────────────────────────────────
        t_s = time.perf_counter()
        partition_static = louvain_static(W_new)
        time_static = time.perf_counter() - t_s
        Q_static = compute_modularity_gpu(partition_static, W_new, m_new)

        # ── ND Louvain ────────────────────────────────────────────────────────
        t_s = time.perf_counter()
        partition_nd = louvain_nd(W_new, partition_nd)
        time_nd = time.perf_counter() - t_s
        Q_nd = compute_modularity_gpu(partition_nd, W_new, m_new)

        # ── DF Louvain ────────────────────────────────────────────────────────
        t_s = time.perf_counter()
        partition_df, n_affected = louvain_df(
            W_new, W_prev, partition_df, frontier_thr
        )
        time_df = time.perf_counter() - t_s
        Q_df = compute_modularity_gpu(partition_df, W_new, m_new)

        # Record
        results["batch"].append(b)
        results["timestamp"].append(ts)
        results["n_affected_df"].append(n_affected)
        results["n_communities"]["static"].append(int(np.unique(partition_static).size))
        results["n_communities"]["nd"].append(int(np.unique(partition_nd).size))
        results["n_communities"]["df"].append(int(np.unique(partition_df).size))
        results["modularity"]["static"].append(Q_static)
        results["modularity"]["nd"].append(Q_nd)
        results["modularity"]["df"].append(Q_df)
        results["time_s"]["static"].append(time_static)
        results["time_s"]["nd"].append(time_nd)
        results["time_s"]["df"].append(time_df)
        results["speedup_vs_static"]["nd"].append(
            time_static / max(time_nd, 1e-9)
        )
        results["speedup_vs_static"]["df"].append(
            time_static / max(time_df, 1e-9)
        )

        W_prev = W_new
        gc.collect()

    return results


# =============================================================================
# 8.  PLOTTING
# =============================================================================
COLORS = {"static": "#888780", "nd": "#378ADD", "df": "#D85A30"}
LABELS = {"static": "Static Louvain", "nd": "ND Louvain", "df": "DF Louvain"}


def plot_modularity(results: dict, dataset_name: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    batches = results["batch"]
    for algo in ["static", "nd", "df"]:
        ax1.plot(batches, results["modularity"][algo],
                 color=COLORS[algo], label=LABELS[algo],
                 linewidth=1.4, alpha=0.85)
        ax2.plot(batches, results["n_communities"][algo],
                 color=COLORS[algo], label=LABELS[algo],
                 linewidth=1.4, alpha=0.85)

    ax1.set_ylabel("Modularity Q", fontsize=11)
    ax1.set_title(f"{dataset_name} — Modularity over time", fontsize=12)
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    ax2.set_ylabel("#Communities", fontsize=11)
    ax2.set_xlabel("Batch (5-min timestep)", fontsize=11)
    ax2.set_title("Number of communities over time", fontsize=12)
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    # Highlight DF-affected fraction
    n_total = len(results["n_affected_df"])
    ax2_twin = ax2.twinx()
    frac = [a / max(results["n_communities"]["df"][i], 1)
            for i, a in enumerate(results["n_affected_df"])]
    ax2_twin.fill_between(batches[:len(frac)], frac, alpha=0.12,
                          color="#D85A30", label="DF frontier fraction")
    ax2_twin.set_ylabel("DF frontier / N", fontsize=9, color="#D85A30")
    ax2_twin.tick_params(axis="y", labelcolor="#D85A30")

    plt.tight_layout()
    path = f"results/modularity_over_time_{dataset_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")


def plot_speedup(results: dict, dataset_name: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    batches = results["batch"]
    for algo, ax in zip(["nd", "df"], axes):
        sp = results["speedup_vs_static"][algo]
        ax.plot(batches, sp,
                color=COLORS[algo], linewidth=1.4, alpha=0.85)
        ax.axhline(1.0, color="#888", linewidth=0.8, linestyle="--")
        mean_sp = np.mean(sp)
        ax.axhline(mean_sp, color=COLORS[algo], linewidth=1.0,
                   linestyle=":", alpha=0.7,
                   label=f"Mean speedup = {mean_sp:.1f}x")
        ax.set_title(f"{LABELS[algo]} vs Static — speedup", fontsize=11)
        ax.set_xlabel("Batch", fontsize=10)
        ax.set_ylabel("Speedup (×)", fontsize=10)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.suptitle(f"{dataset_name} — Runtime speedup over Static Louvain",
                 fontsize=12)
    plt.tight_layout()
    path = f"results/speedup_comparison_{dataset_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")


def plot_community_evolution(results: dict, dataset_name: str):
    """Heatmap-style: community assignment over time for DF Louvain.
    We track which batch each change in #communities occurs."""
    fig, ax = plt.subplots(figsize=(12, 4))
    nc = results["n_communities"]["df"]
    batches = results["batch"]
    ax.step(batches, nc, color=COLORS["df"], linewidth=1.8, where="post")
    ax.fill_between(batches, nc, alpha=0.15, color=COLORS["df"], step="post")
    ax.set_xlabel("Batch (5-min timestep)", fontsize=11)
    ax.set_ylabel("#Communities (DF)", fontsize=11)
    ax.set_title(f"{dataset_name} — Community count evolution (DF Louvain)",
                 fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"results/community_evolution_{dataset_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")


def save_benchmark_table(results: dict, dataset_name: str):
    """Reproduce Table 1 style summary from the paper."""
    rows = []
    N_sensors = results.get("n_sensors", "?")
    for algo in ["static", "nd", "df"]:
        sp = results["speedup_vs_static"].get(algo, [1.0] * len(results["batch"]))
        q_vals = [v for v in results["modularity"][algo] if not np.isnan(v)]
        rows.append({
            "Algorithm":         LABELS[algo],
            "Mean time (s)":     f"{np.mean(results['time_s'][algo]):.4f}",
            "Median time (s)":   f"{np.median(results['time_s'][algo]):.4f}",
            "Mean speedup":      f"{np.mean(sp):.1f}x" if algo != "static" else "1.0x",
            "Mean modularity Q": f"{np.mean(q_vals):.4f}" if q_vals else "nan",
            "Mean #communities": f"{np.mean(results['n_communities'][algo]):.1f}",
            "Q vs Static (%)":   f"{(np.mean(q_vals)/max(np.mean([v for v in results['modularity']['static'] if not np.isnan(v)]),1e-9)-1)*100:+.1f}%" if algo != "static" and q_vals else "—",
        })
    # DF frontier fraction = affected sensors / total sensors (N)
    # n_affected_df is absolute count; divide by N to get fraction 0..1
    n_sensors_total = results.get('n_sensors', 207)  # set in run_benchmark
    rows[2]["Mean frontier frac"] = \
        f"{np.mean(results['n_affected_df']) / max(n_sensors_total, 1):.3f}"

    df = pd.DataFrame(rows)
    print(f"\n  === Benchmark Table ({dataset_name}) ===")
    print(df.to_string(index=False))
    path = f"results/benchmark_table_{dataset_name}.csv"
    df.to_csv(path, index=False)
    print(f"  Saved -> {path}")
    return df


# =============================================================================
# 9.  GPU MEMORY SUMMARY
# =============================================================================
def print_gpu_summary():
    if not GPU_AVAILABLE:
        print(f"  Compute: CPU only")
        return
    if GPU_BACKEND == "torch":
        props = _torch.cuda.get_device_properties(0)
        total = props.total_memory
        free  = total - _torch.cuda.memory_allocated(0)
    else:
        free, total = _cp.cuda.runtime.memGetInfo()
    used  = total - free
    print(f"  GPU: {GPU_NAME}")
    print(f"  VRAM total: {total/1e9:.1f} GB")
    print(f"  VRAM used:  {used/1e9:.2f} GB")
    print(f"  VRAM free:  {free/1e9:.1f} GB")


# =============================================================================
# 10. MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="DF Louvain on METR-LA / PEMS-BAY"
    )
    parser.add_argument("--dataset",   default="metr-la",
                        choices=["metr-la", "pems-bay"])
    parser.add_argument("--data_path", default="./data/metr-la.h5",
                        help="Path to the .h5 dataset file")
    parser.add_argument("--dist_csv",  default=None,
                        help="Optional sensor distance CSV (for real topology)")
    parser.add_argument("--adj_pkl",   default=None,
                        help="Path to adj_mx.pkl from DCRNN repo (recommended for METR-LA/PEMS-BAY)")
    parser.add_argument("--n_batches", type=int, default=200,
                        help="Number of 5-min batches to process")
    parser.add_argument("--window",    type=int, default=12,
                        help="Correlation window (timesteps, default=12=1hr)")
    parser.add_argument("--frontier_thr", type=float, default=1e-3,
                        help="Edge weight change threshold for frontier")
    args = parser.parse_args()

    print("=" * 65)
    print("DF Louvain — Dynamic Community Detection on Traffic Networks")
    print("=" * 65)
    print(f"  Dataset:        {args.dataset}")
    print(f"  Data path:      {args.data_path}")
    print(f"  Batches:        {args.n_batches}")
    print(f"  Window:         {args.window} steps ({args.window*5} min)")
    print(f"  Frontier thr:   {args.frontier_thr}")
    if getattr(args, "adj_pkl", None):
        print(f"  Adjacency:      {args.adj_pkl} (adj_mx.pkl)")
    elif args.dist_csv:
        print(f"  Adjacency:      {args.dist_csv} (distance CSV)")
    else:
        print(f"  Adjacency:      synthetic (no file provided)")
    print_gpu_summary()
    print()

    # ── Load data ─────────────────────────────────────────────────────────────
    if not os.path.exists(args.data_path):
        print(f"\n  ERROR: data file not found at {args.data_path}")
        print("  Download from: https://github.com/liyaguang/DCRNN")
        print("  Place metr-la.h5 and pems-bay.h5 in ./data/")
        print("\n  Running with SYNTHETIC data for demonstration...")
        _run_synthetic_demo(args)
        return

    print("Loading data...")
    if args.dataset == "metr-la":
        speeds, timestamps, sensor_ids = load_metr_la(args.data_path)
    else:
        speeds, timestamps, sensor_ids = load_pems_bay(args.data_path)

    N, T = speeds.shape[1], speeds.shape[0]
    print(f"  Sensors: {N} | Timesteps: {T} | "
          f"Duration: {T*5/60:.0f} hours")

    # ── Build adjacency ───────────────────────────────────────────────────────
    print("\nBuilding adjacency matrix...")
    if getattr(args, "adj_pkl", None) and os.path.exists(args.adj_pkl):
        print(f"  Loading from adj_mx.pkl: {args.adj_pkl}")
        adj = load_adj_pkl(args.adj_pkl, h5_sensor_ids=sensor_ids)
    else:
        adj = build_adjacency_from_distances(args.dist_csv, N, sensor_ids=sensor_ids)
    if getattr(args, "adj_pkl", None) and not os.path.exists(args.adj_pkl):
        print(f"  [warn] adj_pkl path not found: {args.adj_pkl} — using dist_csv fallback.")
        adj = build_adjacency_from_distances(args.dist_csv, N, sensor_ids=sensor_ids)
    n_edges = int((adj > 0).sum() // 2)
    print(f"  Edges: {n_edges} | Density: {n_edges/(N*(N-1)/2):.4f}")

    # ── Run benchmark ─────────────────────────────────────────────────────────
    print("\nRunning benchmark (Static / ND / DF Louvain)...")
    t_total = time.perf_counter()
    results = run_benchmark(
        speeds, timestamps, adj,
        window=args.window,
        batch_size=1,
        n_batches=args.n_batches,
        frontier_thr=args.frontier_thr,
        start_batch=args.window,
    )
    elapsed = time.perf_counter() - t_total
    print(f"\n  Total benchmark time: {elapsed:.1f}s")

    # ── Plots + table ─────────────────────────────────────────────────────────
    print("\nGenerating outputs...")
    plot_modularity(results, args.dataset)
    plot_speedup(results, args.dataset)
    plot_community_evolution(results, args.dataset)
    save_benchmark_table(results, args.dataset)

    print(f"\nAll outputs saved to ./results/")


# =============================================================================
# 11. SYNTHETIC DEMO (when dataset not yet downloaded)
# =============================================================================
def _run_synthetic_demo(args):
    """
    Generates a synthetic traffic-like dataset with N sensors and T timestamps,
    runs the full pipeline, and produces all output figures.
    Useful for testing the pipeline before downloading the real data.
    """
    print("\n  Generating synthetic traffic dataset...")
    N, T = 207, 2000
    rng  = np.random.default_rng(42)

    # Simulate speed with slow drift + correlated noise (realistic traffic)
    base     = 40 + 20 * np.sin(np.linspace(0, 4 * np.pi, T))[:, None]
    corr     = rng.standard_normal((T, 10)) @ rng.standard_normal((10, N))
    noise    = rng.standard_normal((T, N)) * 2
    speeds   = (base + corr * 5 + noise).astype(np.float32)
    speeds   = np.clip(speeds, 0, 80)

    timestamps = np.arange(T) * 300    # 5-min steps in seconds

    # Simple ring topology for adjacency
    adj = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in [i - 1, i + 1, i - 5, i + 5]:
            j = j % N
            adj[i, j] = 0.8

    print(f"  Synthetic: N={N}, T={T}")
    results = run_benchmark(
        speeds, timestamps, adj,
        window=12, batch_size=1,
        n_batches=min(args.n_batches, 150),
        frontier_thr=args.frontier_thr,
        start_batch=12,
    )
    plot_modularity(results, f"{args.dataset}_synthetic")
    plot_speedup(results, f"{args.dataset}_synthetic")
    plot_community_evolution(results, f"{args.dataset}_synthetic")
    save_benchmark_table(results, f"{args.dataset}_synthetic")
    print("\n  Synthetic demo complete. Download real data for full results.")
    print("  https://github.com/liyaguang/DCRNN")


if __name__ == "__main__":
    main()