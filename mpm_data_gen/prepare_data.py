import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
import os
from tqdm import tqdm
import torch
from torch_geometric.data import Data


def create_graph(positions, time_step, prev_positions, clip_radius=0.1, connectivity_radius=0.1):
    n_particles = positions.shape[0]

    # Create KDTree for efficient neighbor search
    tree = cKDTree(positions)

    # Find edges
    edges = tree.query_pairs(r=connectivity_radius, output_type='ndarray')

    # Node features
    features = []
    for i in range(n_particles):
        pos = positions[i]

        # Previous velocities
        velocities = []
        for j in range(1, 6):
            if time_step - j >= 0:
                velocities.append(prev_positions[j][i] - prev_positions[j + 1][i])
            else:
                velocities.append(np.zeros(2))

        # Distance to walls
        distance_to_walls = np.clip(np.array([pos, 1 - pos]), 0, clip_radius)

        # Combine features
        node_features = np.concatenate([
            pos,
            np.array(velocities).flatten(),
            distance_to_walls.flatten(),
            np.array([-9.8])  # gravity
        ])

        features.append(node_features)

    # Convert to PyTorch tensors
    x = torch.tensor(features, dtype=torch.float)
    edge_index = torch.tensor(edges.T, dtype=torch.long)

    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index)

    return data


def process_simulation(file_path):
    data = np.load(file_path)
    positions = data['data']

    graphs = []
    for t in range(positions.shape[0] - 1):  # -1 because we need the next position for the target
        prev_positions = [positions[max(0, t - i)] for i in range(7)]  # Current + 6 previous
        graph = create_graph(positions[t], t, prev_positions)
        graph.y = torch.tensor(positions[t + 1], dtype=torch.float)  # Next position as target
        graphs.append(graph)

    return graphs


def prepare_dataset(data_dir):
    all_graphs = []
    for file_name in tqdm(os.listdir(data_dir)):
        if file_name.endswith('.npz'):
            file_path = os.path.join(data_dir, file_name)
            all_graphs.extend(process_simulation(file_path))
    return all_graphs


# Prepare datasets
train_dir = '/home/groups/deissero/mrohatgi/MPM/MPM_TRAIN'
val_dir = '/home/groups/deissero/mrohatgi/MPM/MPM_VAL'

print("Preparing training dataset...")
train_dataset = prepare_dataset(train_dir)
print("Preparing validation dataset...")
val_dataset = prepare_dataset(val_dir)

# Save datasets
torch.save(train_dataset, '/home/groups/deissero/mrohatgi/MPM/train_dataset.pt')
torch.save(val_dataset, '/home/groups/deissero/mrohatgi/MPM/val_dataset.pt')

print(f"Saved {len(train_dataset)} training graphs and {len(val_dataset)} validation graphs.")