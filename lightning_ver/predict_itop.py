"""
Module for evaluating the SPiKE model on the ITOP dataset.
"""

from __future__ import print_function
import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from model import model_builder
from trainer_itop import load_data, create_criterion
from utils.config_utils import load_config, set_random_seed
from utils.metrics import joint_accuracy


import open3d as o3d
import numpy as np

def visualize_with_pointcloud(predictions, targets, point_cloud):
    """Visualizes predictions, ground truth, and point cloud."""

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 🔹 Convert to NumPy float64
    point_cloud = np.asarray(point_cloud, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    # 🔹 Fix point cloud shape if needed (from (3, 4096, 3) to (N, 3))
    if point_cloud.ndim == 3:
        point_cloud = point_cloud.reshape(-1, 3)

    # 🔹 Fix predictions shape if needed (from (1, 15, 3) to (15, 3))
    if predictions.shape == (1, 15, 3):
        predictions = predictions.squeeze(0)  # Remove batch dimension

    if targets.shape == (1, 15, 3):
        targets = targets.squeeze(0)  # Remove batch dimension

    # 🔹 Ensure correct shape
    if point_cloud.ndim != 2 or point_cloud.shape[1] != 3:
        raise ValueError(f"Point cloud shape is invalid after reshaping: {point_cloud.shape}, expected (N, 3)")

    if predictions.shape != (15, 3):
        raise ValueError(f"Predictions shape is invalid after reshaping: {predictions.shape}, expected (15, 3)")

    if targets.shape != (15, 3):
        raise ValueError(f"Ground truth shape is invalid after reshaping: {targets.shape}, expected (15, 3)")

    # Convert to Open3D Point Cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.paint_uniform_color([0.6, 0.6, 0.6])  # Grey points

    # Ground truth joints (Green)
    gt_joints = o3d.geometry.PointCloud()
    gt_joints.points = o3d.utility.Vector3dVector(targets)
    gt_joints.paint_uniform_color([0, 1, 0])  # Green for GT

    # Predicted joints (Red)
    pred_joints = o3d.geometry.PointCloud()
    pred_joints.points = o3d.utility.Vector3dVector(predictions)
    pred_joints.paint_uniform_color([1, 0, 0])  # Red for predictions

    vis.add_geometry(pcd)
    vis.add_geometry(gt_joints)
    vis.add_geometry(pred_joints)
    vis.run()
    vis.destroy_window()


def evaluate(model, criterion, data_loader, device, threshold):

    model.eval()
    total_loss = 0.0
    total_pck = np.zeros(15)
    total_map = 0.0
    clip_losses = []

    with torch.no_grad():
        for batch_clips, batch_targets, batch_video_ids in tqdm(
            data_loader, desc="Validation" if data_loader.dataset.train else "Test"
        ):
            for clip, target, video_id in zip(
                batch_clips, batch_targets, batch_video_ids
            ):
                clip = clip.unsqueeze(0).to(device, non_blocking=True)
                target = target.unsqueeze(0).to(device, non_blocking=True)

                output = model(clip).reshape(target.shape)
                loss = criterion(output, target)

                pck, mean_ap = joint_accuracy(output, target, threshold)
                total_pck += pck.detach().cpu().numpy()
                total_map += mean_ap.detach().cpu().item()

                total_loss += loss.item()
                clip_losses.append(
                    (
                        video_id.cpu().detach().numpy(),
                        loss.item(),
                        clip.cpu().detach().numpy(),
                        target.cpu().detach().numpy(),
                        output.cpu().detach().numpy(),
                    )
                )

        total_loss /= len(data_loader.dataset)
        total_map /= len(data_loader.dataset)
        total_pck /= len(data_loader.dataset)

    if len(clip_losses) > 0:
        first_prediction = clip_losses[0][4][0]  # Model's first output (15, 3)
        first_target = clip_losses[0][3][0]      # Corresponding ground truth (15, 3)
        first_point_cloud = clip_losses[0][2][0] # First input point cloud (2048, 3)

    # Optional: Show full point cloud with joints
    # visualize_with_pointcloud(first_prediction, first_target, first_point_cloud)


    visualize_with_pointcloud(first_prediction, first_target, first_point_cloud)

    return clip_losses, total_loss, total_map, total_pck


def main(arguments):

    config = load_config(arguments.config)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["device_args"])
    print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    device = torch.device(0)

    set_random_seed(config["seed"])

    print(f"Loading data from {config['dataset_path']}")
    data_loader_test, num_coord_joints = load_data(config, mode="test")

    model = model_builder.create_model(config, num_coord_joints)
    model.to(device)

    criterion = create_criterion(config)

    # print(f"Loading model from {arguments.model}")
    # checkpoint = torch.load(arguments.model, map_location="cpu")
    # model.load_state_dict(checkpoint["model"], strict=True)

    print(f"Loading model from {arguments.model}")
    checkpoint = torch.load(arguments.model, map_location="cpu")

    # 🔍 Debug: Print available keys in the checkpoint file
    print(f"🔍 Checkpoint Keys: {checkpoint.keys()}")

    # Ensure correct key for loading model weights
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
    elif "state_dict" in checkpoint:  # Some checkpoints store weights under "state_dict"
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        raise KeyError("Model weights not found in checkpoint file!")    

    # ----

    losses, val_clip_loss, val_map, val_pck = evaluate(
        model, criterion, data_loader_test, device=device, threshold=config["threshold"]
    )
    losses.sort(key=lambda x: x[1], reverse=True)

    print(f"Validation Loss: {val_clip_loss:.4f}")
    print(f"Validation mAP: {val_map:.4f}")
    print(f"Validation PCK: {val_pck}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPiKE Testing on ITOP dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="ITOP-SIDE/1",
        help="Path to the YAML config file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="experiments/ITOP-SIDE/1/log/best_model.pth",
        help="Path to the model checkpoint",
    )

    args = parser.parse_args()
    main(args)
