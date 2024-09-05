import torch
import h5py
import numpy as np
from tqdm import tqdm
from scipy.spatial import cKDTree
from model import GraphAttentionNetwork

def load_initial_state(file_path, start_graph_index=6):
    with h5py.File(file_path, 'r') as f:
        graph_data_prev = f[f'graph_{start_graph_index-1}']
        graph_data_current = f[f'graph_{start_graph_index}']
        
        x = np.array(graph_data_current['x'][:])
        edge_index = np.array(graph_data_current['edge_index'][:])
        edge_attr = np.array(graph_data_current['edge_attr'][:])
        
        positions_prev = np.array(graph_data_prev['x'][:, :2])
        positions_current = np.array(graph_data_current['x'][:, :2])
        
    initial_velocity = positions_current - positions_prev
    
    return x, edge_index, edge_attr, initial_velocity, positions_prev

def create_graph(positions, prev_positions, time_step, clip_radius, connectivity_radius):
    n_particles = positions.shape[0]

    tree = cKDTree(positions)
    edges = tree.query_pairs(r=connectivity_radius, output_type='ndarray')

    features_list = []
    for i in range(n_particles):
        pos = positions[i]
        velocities = [prev_positions[j][i] - prev_positions[j+1][i] if time_step - j >= 0 else np.zeros(2) for j in range(0, 5)]
        distance_to_walls = np.clip(np.array([pos, 1-pos]), 0, clip_radius)
        node_features = np.concatenate([pos, np.array(velocities).flatten(), distance_to_walls.flatten(), np.array([-9.8])])
        features_list.append(node_features)

    x = np.array(features_list, dtype=np.float32)
    edge_index = edges.astype(np.int32)
    edge_attr = np.linalg.norm(positions[edges[:, 0]] - positions[edges[:, 1]], axis=1).astype(np.float32)

    return x, edge_index, edge_attr

def run_rollout(model, initial_state, num_steps, device, clip_radius, connectivity_radius):
    model.eval()
    x, edge_index, edge_attr, initial_velocity, initial_prev_position = initial_state

    current_positions = x[:, :2]
    prev_positions = [current_positions, 
                      initial_prev_position] + [initial_prev_position for _ in range(4)]
    
    trajectory = [current_positions]
    #print(trajectory, flush = True)
    for time_step in tqdm(range(num_steps)):
        x_tensor = torch.FloatTensor(x).to(device)
        edge_index_tensor = torch.LongTensor(edge_index.T).to(device)
        edge_attr_tensor = torch.FloatTensor(edge_attr).unsqueeze(1).to(device)

        with torch.no_grad():
            predicted_positions = model(x_tensor, edge_index_tensor, edge_attr_tensor)
            predicted_positions = predicted_positions.cpu().numpy()

        del x_tensor, edge_index_tensor, edge_attr_tensor
        torch.cuda.empty_cache()

        prev_positions = [predicted_positions] + prev_positions[:-1]

        x, edge_index, edge_attr = create_graph(
            predicted_positions, 
            prev_positions, 
            time_step + 1, 
            clip_radius, 
            connectivity_radius
        )

        trajectory.append(predicted_positions)

        if (time_step + 1) % 100 == 0:
            print(trajectory[0], flush=True)
            np.save(f'rollout_trajectory_part_{time_step + 1}.npy', np.array(trajectory))
            trajectory = [trajectory[-1]]

    np.save(f'rollout_trajectory_part_final.npy', np.array(trajectory))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = '/home/groups/deissero/mrohatgi/MPM/model_checkpoints_mini/full_model_epoch_3.pth'
    model = torch.load(model_path).to(device)

    initial_state_path = '/scratch/users/mrohatgi/MPM_processed/test/dataset_chunk_0.h5'
    initial_state = load_initial_state(initial_state_path, start_graph_index=1)

    clip_radius = 0.007
    connectivity_radius = 0.007
    num_steps = 3000

    trajectory = run_rollout(model, initial_state, num_steps, device, clip_radius, connectivity_radius)

    np.save('rollout_trajectory.npy', trajectory)
