import h5py
# Open the file in read-only mode
with h5py.File('data/pems-bay.h5', 'r') as f:
    # List all root groups/datasets
    print(list(f.keys()))
    # Access a specific dataset
    data = f['speed/block0_values'][:]
    print(data)

