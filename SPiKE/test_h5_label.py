import h5py
import numpy as np
import os

# Define joint positions (same as before, for a test case)
joint_positions = np.array([
    [0, 1.75, 0],   # Head
    [0, 1.5, 0],    # Neck
    [-0.3, 1.4, 0], # Left Shoulder
    [-0.6, 1.1, 0], # Left Elbow
    [-0.9, 0.8, 0], # Left Hand
    [0.3, 1.4, 0],  # Right Shoulder
    [0.6, 1.1, 0],  # Right Elbow
    [0.9, 0.8, 0],  # Right Hand
    [0, 1.0, 0],    # Spine
    [-0.3, 0.9, 0], # Left Hip
    [-0.3, 0.5, 0], # Left Knee
    [-0.3, 0.1, 0], # Left Foot
    [0.3, 0.9, 0],  # Right Hip
    [0.3, 0.5, 0],  # Right Knee
    [0.3, 0.1, 0],  # Right Foot
])

# Create fake frame identifiers
frame_ids = np.array([b"00_00000"])  # Needs to be byte strings

# Validity flag (1 means valid, 0 means invalid)
is_valid = np.array([1], dtype=np.uint8)  # Mark as valid (must be uint8)


# Define save path
output_folder = "C:/Users/llar/Documents/GitHub/SPiKE/datasets/processed/"
os.makedirs(output_folder, exist_ok=True)
h5_path = os.path.join(output_folder, "test_labels.h5")

# Save as HDF5
with h5py.File(h5_path, "w") as f:
    f.create_dataset("id", data=frame_ids)
    f.create_dataset("real_world_coordinates", data=np.expand_dims(joint_positions, axis=0))  # Add batch dimension
    f.create_dataset("is_valid", data=is_valid)

print(f"✅ HDF5 label file saved at {h5_path}")
