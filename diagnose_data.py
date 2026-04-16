# diagnose_data.py  — run this first, paste the output here
import h5py, numpy as np, pandas as pd

h5_path  = "./data/metr-la.h5"
csv_path = "./data/distances_la_2012.csv"

print("=== H5 file structure ===")
with h5py.File(h5_path, "r") as f:
    def _show(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"  {name}: shape={obj.shape} dtype={obj.dtype}")
    f.visititems(_show)

print("\n=== Speed data sample ===")
with h5py.File(h5_path, "r") as f:
    speeds = f["df"]["block0_values"][:]
print(f"Shape: {speeds.shape}  dtype: {speeds.dtype}")
print(f"Min: {np.nanmin(speeds):.4f}  Max: {np.nanmax(speeds):.4f}")
print(f"Zeros: {(speeds == 0).mean()*100:.1f}%")
print(f"NaN:   {np.isnan(speeds).mean()*100:.1f}%")
print(f"First window (rows 0-12) col 0 values: {speeds[:12, 0]}")

print("\n=== Distance CSV sample ===")
df = pd.read_csv(csv_path)
print(df.head())
print(f"Rows: {len(df)}  Columns: {df.columns.tolist()}")
print(f"Min distance: {df.iloc[:,2].min():.4f}  Max: {df.iloc[:,2].max():.4f}")

print("\n=== Correlation test (window=12, no NaN replacement) ===")
S = speeds[12:24].astype(np.float64)
print(f"Any NaN in window: {np.isnan(S).any()}")
print(f"Any zero in window: {(S==0).any()}")
mu = S.mean(axis=0); sigma = S.std(axis=0)
print(f"Sensors with std=0: {(sigma < 1e-6).sum()} / {len(sigma)}")
S_std = (S - mu) / (sigma + 1e-8)
W = (S_std.T @ S_std) / S_std.shape[0]
print(f"Corr matrix: min={W.min():.4f}  max={W.max():.4f}  nan_count={np.isnan(W).sum()}")
print(f"Positive entries: {(W > 0.3).sum()}")