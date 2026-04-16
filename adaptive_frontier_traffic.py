"""
Adaptive Frontier Louvain on Road Networks (Traffic + SNAP)
===========================================================
Novel approach: Adaptive Frontier detection for dynamic community detection.
- Adjusts frontier threshold per batch using change-score quantiles.
- Caps frontier size for low-spec machines.
- Expands frontier locally to preserve quality.

Supports:
  • METR-LA / PEMS-BAY (traffic sensor networks, DCRNN format)
  • SNAP road networks (edge list) with synthetic traffic dynamics

Outputs:
  - CLI summaries per run
  - CSV tables (summary + per-batch time series)
  - Plots (modularity, speedup, community evolution, frontier adaptation)

Usage examples:
  python adaptive_frontier_traffic.py --dataset metr-la --data_path ./data/metr-la.h5 --adj_pkl ./data/adj_mx_metr.pkl
  python adaptive_frontier_traffic.py --dataset pems-bay --data_path ./data/pems-bay.h5 --adj_pkl ./data/adj_mx_bay.pkl
  python adaptive_frontier_traffic.py --dataset snap --snap_path ./data/roadNet-CA.txt --snap_nodes 1500
"""

import argparse
import os
import time
import gc
from collections import defaultdict
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# Reuse the DF Louvain implementation and GPU setup from existing script
from df_louvain_traffic import (
    load_metr_la,
    load_pems_bay,
    load_adj_pkl,
    build_adjacency_from_distances,
    compute_correlation_weights_gpu,
    compute_modularity_gpu,
    louvain_static,
    louvain_nd,
    louvain_df,
    GPU_AVAILABLE,
    GPU_BACKEND,
    GPU_NAME,
    print_gpu_summary,
)

os.makedirs("results", exist_ok=True)

COLORS = {
    "static": "#888780",
    "nd": "#378ADD",
    "df": "#D85A30",
    "af": "#2FAF6D",
}
LABELS = {
    "static": "Static Louvain",
    "nd": "ND Louvain",
    "df": "DF Louvain",
    "af": "Adaptive Frontier (AF)",
}


def load_snap_roadnet(edge_path: str, max_nodes: int = 2000) -> np.ndarray:
    """
    Load SNAP road network edge list and build an undirected adjacency.
    Large SNAP graphs are huge; max_nodes restricts to first N unique nodes
    encountered to keep memory small for low-spec machines.
    """
    if not os.path.exists(edge_path):
        raise FileNotFoundError(edge_path)

    edges = []
    node_map = {}
    next_id = 0

    with open(edge_path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            u_str, v_str = line.strip().split()[:2]
            u, v = int(u_str), int(v_str)
            if u not in node_map:
                if next_id >= max_nodes:
                    continue
                node_map[u] = next_id
                next_id += 1
            if v not in node_map:
                if next_id >= max_nodes:
                    continue
                node_map[v] = next_id
                next_id += 1
            ui, vi = node_map[u], node_map[v]
            if ui != vi:
                edges.append((ui, vi))

    n = len(node_map)
    A = np.zeros((n, n), dtype=np.float32)
    for ui, vi in edges:
        A[ui, vi] = 1.0
        A[vi, ui] = 1.0
    np.fill_diagonal(A, 0.0)

    n_edges = int((A > 0).sum() // 2)
    print(f"  SNAP roadnet: nodes={n}, edges={n_edges}, density={n_edges / max(n*(n-1)/2,1):.6f}")
    return A


def synthetic_traffic(T: int, N: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic traffic-like speeds for SNAP road networks."""
    rng = np.random.default_rng(seed)
    base = 35 + 15 * np.sin(np.linspace(0, 6 * np.pi, T))[:, None]
    latent = rng.standard_normal((T, 12)) @ rng.standard_normal((12, N))
    noise = rng.standard_normal((T, N)) * 2.0
    speeds = (base + latent * 4 + noise).astype(np.float32)
    speeds = np.clip(speeds, 0, 80)
    timestamps = np.arange(T) * 300
    return speeds, timestamps


def _adaptive_frontier_mask(
    W_old: np.ndarray,
    W_new: np.ndarray,
    k_old: np.ndarray,
    k_new: np.ndarray,
    percentile: float = 0.90,
    min_thr: float = 1e-4,
    max_frac: float = 0.20,
    degree_alpha: float = 0.30,
    neighbor_scale: float = 0.50,
) -> Tuple[np.ndarray, float]:
    """
    Adaptive frontier detection.

    Score_i = max(|Δw|) + degree_alpha * |Δk| / (k_old + eps)
    Threshold = max(min_thr, quantile(score, percentile)).
    Frontier capped to max_frac via top-k if needed.
    """
    delta = np.abs(W_new - W_old)
    max_edge_change = delta.max(axis=1)
    deg_change = np.abs(k_new - k_old) / np.maximum(k_old, 1e-6)
    score = max_edge_change + degree_alpha * deg_change

    nonzero = score[score > 0]
    if nonzero.size == 0:
        return np.zeros_like(score, dtype=bool), min_thr

    thr = max(min_thr, float(np.quantile(nonzero, percentile)))
    affected = score >= thr

    # Expand to neighbors with a softer threshold
    if neighbor_scale < 1.0 and affected.any():
        neighbor_mask = np.zeros_like(affected)
        idx = np.where(affected)[0]
        for i in idx:
            neighbors = np.where(W_new[i] > 0)[0]
            if neighbors.size:
                neighbor_mask[neighbors] = True
        affected = affected | (neighbor_mask & (score >= thr * neighbor_scale))

    # Cap frontier size for low-spec machines
    max_nodes = int(max_frac * len(score))
    if max_nodes > 0 and affected.sum() > max_nodes:
        top_idx = np.argsort(score)[-max_nodes:]
        mask = np.zeros_like(affected)
        mask[top_idx] = True
        affected = mask

    return affected, thr


def louvain_af(
    W_new: np.ndarray,
    W_old: np.ndarray,
    prev_partition: np.ndarray,
    percentile: float = 0.90,
    min_thr: float = 1e-4,
    max_frac: float = 0.20,
    degree_alpha: float = 0.30,
    neighbor_scale: float = 0.50,
    max_passes: int = 5,
) -> Tuple[np.ndarray, int, float]:
    """
    Adaptive Frontier Louvain.

    1) Detect adaptive frontier (quantile-based threshold).
    2) Run local move only on frontier vertices.
    3) Expand frontier to neighbors when a vertex moves (like DF Louvain).
    """
    N = W_new.shape[0]
    m = W_new.sum() / 2.0
    k_new = W_new.sum(axis=1)
    k_old = W_old.sum(axis=1)
    partition = prev_partition.copy()

    affected, thr = _adaptive_frontier_mask(
        W_old, W_new, k_old, k_new,
        percentile=percentile,
        min_thr=min_thr,
        max_frac=max_frac,
        degree_alpha=degree_alpha,
        neighbor_scale=neighbor_scale,
    )

    if not affected.any():
        return partition, 0, thr

    # Precompute community degrees
    k_c = np.zeros(N, dtype=np.float64)
    for i in range(N):
        k_c[partition[i]] += k_new[i]

    for _ in range(max_passes):
        moved = False
        vertices = np.where(affected)[0]
        for i in vertices:
            ci = partition[i]
            ki = k_new[i]

            neigh_w = defaultdict(float)
            for j in np.where(W_new[i] > 0)[0]:
                neigh_w[partition[j]] += W_new[i, j]

            k_i_in = neigh_w.get(ci, 0.0)
            dQ_remove = -(k_i_in / m - (k_c[ci] - ki) * ki / (2 * m * m)) if m > 1e-10 else 0.0

            best_dQ = 0.0
            best_c = ci
            for cj, w_ij in neigh_w.items():
                if cj == ci:
                    continue
                dQ_insert = w_ij / m - k_c[cj] * ki / (2 * m * m)
                dQ_total = dQ_remove + dQ_insert
                if dQ_total > best_dQ:
                    best_dQ = dQ_total
                    best_c = cj

            if best_c != ci:
                k_c[ci] -= ki
                k_c[best_c] += ki
                partition[i] = best_c
                moved = True

                # Frontier expansion to neighbors
                for j in np.where(W_new[i] > 0)[0]:
                    affected[j] = True

        if not moved:
            break

    # Relabel communities to consecutive integers
    mapping = {old: new for new, old in enumerate(np.unique(partition))}
    partition = np.array([mapping[l] for l in partition], dtype=np.int32)
    return partition, int(affected.sum()), thr


def run_benchmark(
    speeds: np.ndarray,
    timestamps: np.ndarray,
    adj: np.ndarray,
    window: int = 12,
    batch_size: int = 1,
    n_batches: int = 200,
    start_batch: int = 12,
    frontier_thr: float = 1e-3,
    af_percentile: float = 0.90,
    af_min_thr: float = 1e-4,
    af_max_frac: float = 0.20,
    af_degree_alpha: float = 0.30,
    af_neighbor_scale: float = 0.50,
) -> dict:
    N = speeds.shape[1]
    T = speeds.shape[0]

    results = {
        "batch": [],
        "timestamp": [],
        "n_sensors": N,
        "n_affected_df": [],
        "n_affected_af": [],
        "adaptive_threshold": [],
        "frontier_frac_af": [],
        "n_communities": {"static": [], "nd": [], "df": [], "af": []},
        "modularity": {"static": [], "nd": [], "df": [], "af": []},
        "time_s": {"static": [], "nd": [], "df": [], "af": []},
        "speedup_vs_static": {"nd": [], "df": [], "af": []},
    }

    t0 = start_batch * batch_size
    W_cur = compute_correlation_weights_gpu(speeds[t0 - window:t0], adj)
    m_cur = W_cur.sum() / 2.0

    print("  Initialising with Static Louvain on first window...")
    partition_static = louvain_static(W_cur)
    partition_nd = partition_static.copy()
    partition_df = partition_static.copy()
    partition_af = partition_static.copy()
    W_prev = W_cur.copy()

    print(f"  Running {n_batches} batches (window={{window}}, batch={{batch_size}}, N={{N}})...")
    for b in tqdm(range(n_batches), desc="  Batches", ncols=70):
        t_end = t0 + (b + 1) * batch_size
        t_start = t_end - window
        if t_end > T:
            break

        W_new = compute_correlation_weights_gpu(speeds[t_start:t_end], adj)
        m_new = W_new.sum() / 2.0

        ts = int(timestamps[t_end - 1]) if t_end - 1 < len(timestamps) else b

        # Static
        t_s = time.perf_counter()
        partition_static = louvain_static(W_new)
        time_static = time.perf_counter() - t_s
        Q_static = compute_modularity_gpu(partition_static, W_new, m_new)

        # ND
        t_s = time.perf_counter()
        partition_nd = louvain_nd(W_new, partition_nd)
        time_nd = time.perf_counter() - t_s
        Q_nd = compute_modularity_gpu(partition_nd, W_new, m_new)

        # DF
        t_s = time.perf_counter()
        partition_df, n_affected_df = louvain_df(W_new, W_prev, partition_df, frontier_thr)
        time_df = time.perf_counter() - t_s
        Q_df = compute_modularity_gpu(partition_df, W_new, m_new)

        # AF
        t_s = time.perf_counter()
        partition_af, n_affected_af, thr = louvain_af(
            W_new, W_prev, partition_af,
            percentile=af_percentile,
            min_thr=af_min_thr,
            max_frac=af_max_frac,
            degree_alpha=af_degree_alpha,
            neighbor_scale=af_neighbor_scale,
        )
        time_af = time.perf_counter() - t_s
        Q_af = compute_modularity_gpu(partition_af, W_new, m_new)

        results["batch"].append(b)
        results["timestamp"].append(ts)
        results["n_affected_df"].append(n_affected_df)
        results["n_affected_af"].append(n_affected_af)
        results["adaptive_threshold"].append(thr)
        results["frontier_frac_af"].append(n_affected_af / max(N, 1))

        for algo, part, Q, tsec in [
            ("static", partition_static, Q_static, time_static),
            ("nd", partition_nd, Q_nd, time_nd),
            ("df", partition_df, Q_df, time_df),
            ("af", partition_af, Q_af, time_af),
        ]:
            results["n_communities"][algo].append(int(np.unique(part).size))
            results["modularity"][algo].append(Q)
            results["time_s"][algo].append(tsec)

        results["speedup_vs_static"]["nd"].append(time_static / max(time_nd, 1e-9))
        results["speedup_vs_static"]["df"].append(time_static / max(time_df, 1e-9))
        results["speedup_vs_static"]["af"].append(time_static / max(time_af, 1e-9))

        W_prev = W_new
        gc.collect()

    return results


def plot_modularity(results: dict, dataset_name: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    batches = results["batch"]
    for algo in ["static", "nd", "df", "af"]:
        ax1.plot(batches, results["modularity"][algo],
                 color=COLORS[algo], label=LABELS[algo], linewidth=1.4, alpha=0.85)
        ax2.plot(batches, results["n_communities"][algo],
                 color=COLORS[algo], label=LABELS[algo], linewidth=1.4, alpha=0.85)

    ax1.set_ylabel("Modularity Q", fontsize=11)
    ax1.set_title(f"{{dataset_name}} — Modularity over time", fontsize=12)
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    ax2.set_ylabel("#Communities", fontsize=11)
    ax2.set_xlabel("Batch (5-min timestep)", fontsize=11)
    ax2.set_title("Number of communities over time", fontsize=12)
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = f"results/modularity_over_time_{{dataset_name}}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {{path}}")


def plot_speedup(results: dict, dataset_name: str):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    batches = results["batch"]

    for algo, ax in zip(["nd", "df", "af"], axes):
        sp = results["speedup_vs_static"][algo]
        ax.plot(batches, sp, color=COLORS[algo], linewidth=1.4, alpha=0.85)
        ax.axhline(1.0, color="#888", linewidth=0.8, linestyle="--")
        mean_sp = np.mean(sp)
        ax.axhline(mean_sp, color=COLORS[algo], linewidth=1.0, linestyle=":",
                   alpha=0.7, label=f"Mean speedup = {{mean_sp:.1f}}x")
        ax.set_title(f"{{LABELS[algo]}} vs Static — speedup", fontsize=10)
        ax.set_xlabel("Batch", fontsize=10)
        ax.set_ylabel("Speedup (×)", fontsize=10)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle(f"{{dataset_name}} — Runtime speedup over Static Louvain", fontsize=12)
    plt.tight_layout()
    path = f"results/speedup_comparison_{{dataset_name}}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {{path}}")


def plot_community_evolution(results: dict, dataset_name: str):
    fig, ax = plt.subplots(figsize=(12, 4))
    nc = results["n_communities"]["af"]
    batches = results["batch"]
    ax.step(batches, nc, color=COLORS["af"], linewidth=1.8, where="post")
    ax.fill_between(batches, nc, alpha=0.15, color=COLORS["af"], step="post")
    ax.set_xlabel("Batch (5-min timestep)", fontsize=11)
    ax.set_ylabel("#Communities (AF)", fontsize=11)
    ax.set_title(f"{{dataset_name}} — Community count evolution (Adaptive Frontier)", fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"results/community_evolution_{{dataset_name}}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {{path}}")


def plot_frontier_adaptation(results: dict, dataset_name: str):
    fig, ax1 = plt.subplots(figsize=(12, 4))
    batches = results["batch"]

    df_frac = [a / max(results["n_sensors"], 1) for a in results["n_affected_df"]]
    af_frac = results["frontier_frac_af"]
    ax1.plot(batches, df_frac, color=COLORS["df"], label="DF frontier fraction")
    ax1.plot(batches, af_frac, color=COLORS["af"], label="AF frontier fraction")
    ax1.set_xlabel("Batch", fontsize=10)
    ax1.set_ylabel("Frontier fraction", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(batches, results["adaptive_threshold"], color="#444", linestyle=":",
             label="Adaptive threshold")
    ax2.set_ylabel("Adaptive threshold", fontsize=10, color="#444")
    ax2.tick_params(axis="y", labelcolor="#444")

    plt.title(f"{{dataset_name}} — Frontier adaptation over time", fontsize=12)
    plt.tight_layout()
    path = f"results/frontier_adaptation_{{dataset_name}}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {{path}}")


def save_benchmark_table(results: dict, dataset_name: str):
    rows = []
    n_sensors_total = results.get("n_sensors", 1)
    for algo in ["static", "nd", "df", "af"]:
        sp = results["speedup_vs_static"].get(algo, [1.0] * len(results["batch"]))
        q_vals = [v for v in results["modularity"][algo] if not np.isnan(v)]
        rows.append({
            "Algorithm": LABELS[algo],
            "Mean time (s)": f"{{np.mean(results['time_s'][algo]):.4f}}",
            "Median time (s)": f"{{np.median(results['time_s'][algo]):.4f}}",
            "Mean speedup": f"{{np.mean(sp):.1f}}x" if algo != "static" else "1.0x",
            "Mean modularity Q": f"{{np.mean(q_vals):.4f}}" if q_vals else "nan",
            "Mean #communities": f"{{np.mean(results['n_communities'][algo]):.1f}}",
            "Q vs Static (%)": (
                f"{{(np.mean(q_vals)/max(np.mean([v for v in results['modularity']['static'] if not np.isnan(v)]),1e-9)-1)*100:+.1f}}%"
                if algo != "static" and q_vals else "—"
            ),
        })

    # Add AF frontier fraction
    rows[-1]["Mean frontier frac"] = f"{{np.mean(results['n_affected_af']) / max(n_sensors_total, 1):.3f}}"

    df = pd.DataFrame(rows)
    print(f"\n  === Benchmark Table ({{dataset_name}}) ===")
    print(df.to_string(index=False))
    path = f"results/benchmark_table_{{dataset_name}}.csv"
    df.to_csv(path, index=False)
    print(f"  Saved -> {{path}}")
    return df


def save_timeseries_csv(results: dict, dataset_name: str):
    df = pd.DataFrame({
        "batch": results["batch"],
        "timestamp": results["timestamp"],
        "time_static": results["time_s"]["static"],
        "time_nd": results["time_s"]["nd"],
        "time_df": results["time_s"]["df"],
        "time_af": results["time_s"]["af"],
        "modularity_static": results["modularity"]["static"],
        "modularity_nd": results["modularity"]["nd"],
        "modularity_df": results["modularity"]["df"],
        "modularity_af": results["modularity"]["af"],
        "communities_static": results["n_communities"]["static"],
        "communities_nd": results["n_communities"]["nd"],
        "communities_df": results["n_communities"]["df"],
        "communities_af": results["n_communities"]["af"],
        "speedup_nd": results["speedup_vs_static"]["nd"],
        "speedup_df": results["speedup_vs_static"]["df"],
        "speedup_af": results["speedup_vs_static"]["af"],
        "n_affected_df": results["n_affected_df"],
        "n_affected_af": results["n_affected_af"],
        "adaptive_threshold": results["adaptive_threshold"],
        "frontier_frac_af": results["frontier_frac_af"],
    })

    path = f"results/benchmark_timeseries_{{dataset_name}}.csv"
    df.to_csv(path, index=False)
    print(f"  Saved -> {{path}}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive Frontier Louvain on Traffic + SNAP Road Networks"
    )
    parser.add_argument("--dataset", default="metr-la",
                        choices=["metr-la", "pems-bay", "snap"])
    parser.add_argument("--data_path", default="./data/metr-la.h5",
                        help="Path to the .h5 dataset file (METR/PEMS)")
    parser.add_argument("--dist_csv", default=None,
                        help="Optional sensor distance CSV (METR/PEMS)")
    parser.add_argument("--adj_pkl", default=None,
                        help="Path to adj_mx.pkl from DCRNN (recommended)")

    # SNAP options
    parser.add_argument("--snap_path", default=None,
                        help="Path to SNAP road network edge list (e.g., roadNet-CA.txt)")
    parser.add_argument("--snap_nodes", type=int, default=2000,
                        help="Max SNAP nodes to load (keeps memory low)")
    parser.add_argument("--snap_timesteps", type=int, default=2000,
                        help="Synthetic timesteps for SNAP traffic series")

    # Benchmark config
    parser.add_argument("--n_batches", type=int, default=200)
    parser.add_argument("--window", type=int, default=12,
                        help="Correlation window (timesteps, default=12=1hr)")
    parser.add_argument("--frontier_thr", type=float, default=1e-3,
                        help="DF frontier threshold")

    # Adaptive Frontier knobs
    parser.add_argument("--af_percentile", type=float, default=0.90,
                        help="Adaptive frontier percentile (0-1)")
    parser.add_argument("--af_min_thr", type=float, default=1e-4,
                        help="Minimum adaptive threshold")
    parser.add_argument("--af_max_frac", type=float, default=0.20,
                        help="Max frontier fraction for AF")
    parser.add_argument("--af_degree_alpha", type=float, default=0.30,
                        help="Weight for degree-change in AF score")
    parser.add_argument("--af_neighbor_scale", type=float, default=0.50,
                        help="Neighbor expansion threshold scale (AF)")

    args = parser.parse_args()

    print("=" * 72)
    print("Adaptive Frontier Louvain — Dynamic Community Detection")
    print("=" * 72)
    print(f"  Dataset:        {{args.dataset}}")
    print(f"  Batches:        {{args.n_batches}}")
    print(f"  Window:         {{args.window}} steps ({{args.window*5}} min)")
    print(f"  DF frontier:    {{args.frontier_thr}}")
    print(f"  AF percentile:  {{args.af_percentile}}")
    print(f"  AF min thr:     {{args.af_min_thr}}")
    print(f"  AF max frac:    {{args.af_max_frac}}")
    print(f"  AF degree α:    {{args.af_degree_alpha}}")
    print(f"  AF neigh scale: {{args.af_neighbor_scale}}")
    print_gpu_summary()
    print()  

    # Load data
    if args.dataset in ["metr-la", "pems-bay"]:
        if not os.path.exists(args.data_path):
            raise FileNotFoundError(args.data_path)

        print("Loading traffic data...")
        if args.dataset == "metr-la":
            speeds, timestamps, sensor_ids = load_metr_la(args.data_path)
        else:
            speeds, timestamps, sensor_ids = load_pems_bay(args.data_path)

        N, T = speeds.shape[1], speeds.shape[0]
        print(f"  Sensors: {{N}} | Timesteps: {{T}}")

        print("Building adjacency matrix...")
        if args.adj_pkl and os.path.exists(args.adj_pkl):
            adj = load_adj_pkl(args.adj_pkl, h5_sensor_ids=sensor_ids)
        else:
            adj = build_adjacency_from_distances(args.dist_csv, N, sensor_ids=sensor_ids)

        dataset_name = args.dataset
    else:
        if not args.snap_path:
            raise ValueError("--snap_path is required for dataset=snap")
        print("Loading SNAP road network...")
        adj = load_snap_roadnet(args.snap_path, max_nodes=args.snap_nodes)
        N = adj.shape[0]
        speeds, timestamps = synthetic_traffic(args.snap_timesteps, N)
        dataset_name = f"snap_{{os.path.splitext(os.path.basename(args.snap_path))[0]}}" 

    # Run benchmark
    print("\nRunning benchmark (Static / ND / DF / AF Louvain)...")
    t_total = time.perf_counter()
    results = run_benchmark(
        speeds, timestamps, adj,
        window=args.window,
        batch_size=1,
        n_batches=args.n_batches,
        frontier_thr=args.frontier_thr,
        start_batch=args.window,
        af_percentile=args.af_percentile,
        af_min_thr=args.af_min_thr,
        af_max_frac=args.af_max_frac,
        af_degree_alpha=args.af_degree_alpha,
        af_neighbor_scale=args.af_neighbor_scale,
    )
    elapsed = time.perf_counter() - t_total
    print(f"\n  Total benchmark time: {{elapsed:.1f}}s")

    # Outputs
    print("\nGenerating outputs...")
    plot_modularity(results, dataset_name)
    plot_speedup(results, dataset_name)
    plot_community_evolution(results, dataset_name)
    plot_frontier_adaptation(results, dataset_name)
    save_benchmark_table(results, dataset_name)
    save_timeseries_csv(results, dataset_name)

    print("\nAll outputs saved to ./results/")


if __name__ == "__main__":
    main()