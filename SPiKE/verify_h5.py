import h5py
import numpy as np

# Path to your test_labels.h5 file
h5_path = "C:/Users/llar/Documents/GitHub/SPiKE/datasets/processed/test_labels.h5"

# Open and print contents
with h5py.File(h5_path, "r") as f:
    print("🔍 Keys in HDF5:", list(f.keys()))  # Should contain ['id', 'real_world_coordinates', 'is_valid']

    print("🔍 Frame IDs:", f["id"][:])  # Should contain b'00_00000'

    print("🔍 Joint Positions Shape:", f["real_world_coordinates"].shape)  # Should be (1, 15, 3)

    print("🔍 is_valid Flags:", f["is_valid"][:])  # Should be [1]
