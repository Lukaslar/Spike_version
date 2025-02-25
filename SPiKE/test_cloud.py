import numpy as np
import os
import h5py

# Define 15 joint positions for a basic human model (x, y, z coordinates)
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

# Paths
output_folder = "C:/Users/llar/Documents/GitHub/SPiKE/datasets/processed/test/"
os.makedirs(output_folder, exist_ok=True)
h5_path = "C:/Users/llar/Documents/GitHub/SPiKE/datasets/processed/test_labels.h5"

# Create HDF5 file for labels
with h5py.File(h5_path, "w") as f:
    id_list = []
    real_world_coords = []
    is_valid_list = []

    # Generate 12 similar frames
    for i in range(12):
        frame_id = f"00_{str(i).zfill(5)}"
        id_list.append(frame_id.encode())  # Encode to bytes
        
        # Add some minor variations to joint positions to simulate movement
        joints = joint_positions + np.random.normal(scale=0.02, size=joint_positions.shape)
        real_world_coords.append(joints)

        # Mark as valid
        is_valid_list.append(1)

        # Generate 2048 points around the skeleton
        num_points = 2048
        point_cloud = []
        for joint in joints:
            num_samples = num_points // len(joint_positions)
            noise = np.random.normal(scale=0.1, size=(num_samples, 3))
            points = joint + noise  # Spread points around the joint
            point_cloud.append(points)
        point_cloud = np.vstack(point_cloud)

        # Save point cloud to .npz
        np.savez(os.path.join(output_folder, f"{i:05d}.npz"), point_cloud)

    # Save to HDF5
    f.create_dataset("id", data=np.array(id_list))
    f.create_dataset("real_world_coordinates", data=np.array(real_world_coords))
    f.create_dataset("is_valid", data=np.array(is_valid_list, dtype=np.uint8))

print("✅ 12 Human-like point clouds and labels saved successfully!")
