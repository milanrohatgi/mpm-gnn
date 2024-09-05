import numpy as np
import os
from scipy.spatial import cKDTree
import h5py
from tqdm import tqdm
import multiprocessing as mp
import time

def create_graph(args):
    positions, time_step, prev_positions, clip_radius, connectivity_radius = args
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

def process_simulation(file_path, pool, clip_radius=0.007, connectivity_radius=0.007):
    
    data = np.load(file_path)
    positions = data['data']
    
    graph_args = []
    for t in range(positions.shape[0] - 1):
        prev_positions = [positions[max(0, t-i)] for i in range(7)]
        graph_args.append((positions[t], t, prev_positions, clip_radius, connectivity_radius))


    graphs = []
    for i, (x, edge_index, edge_attr) in enumerate(tqdm(pool.imap(create_graph, graph_args), total=len(graph_args), desc="Creating graphs")):
        y = positions[i+1]
        graphs.append((x, y, edge_index, edge_attr))
        if i % 100 == 0:
            print(f"Processed {i}/{len(graph_args)} graphs", flush=True)

    return graphs

def save_chunk(chunk, output_file):
    start_time = time.time()
    
    with h5py.File(output_file, 'w') as f:
        for i, (x, y, edge_index, edge_attr) in enumerate(chunk):
            g = f.create_group(f'graph_{i}')
            g.create_dataset('x', data=x, compression='lzf')
            g.create_dataset('y', data=y, compression='lzf')
            g.create_dataset('edge_index', data=edge_index, compression='lzf')
            g.create_dataset('edge_attr', data=edge_attr, compression='lzf')
            if i % 200 == 0:
                print(f"Saved {i}/{len(chunk)} graphs", flush=True)
    
def prepare_dataset(data_dir, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')]
    
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for file_index, file_path in enumerate(tqdm(file_paths, desc="processing files")):
            output_file = os.path.join(output_dir, f'dataset_chunk_{file_index}.h5')
            save_chunk(graphs, output_file)


# Prepare datasets
train_dir = '/home/groups/deissero/mrohatgi/MPM/MPM_TRAIN'
val_dir = '/home/groups/deissero/mrohatgi/MPM/MPM_VAL'
test_dir = '/home/groups/deissero/mrohatgi/MPM/MPM_TEST'

output_dir = '/scratch/users/mrohatgi/MPM_processed'

#print("Starting data preparation process", flush=True)
#print("Preparing training dataset...", flush=True)
#prepare_dataset(train_dir, os.path.join(output_dir, 'train'))

#print("Preparing validation dataset...", flush=True)
#prepare_dataset(val_dir, os.path.join(output_dir, 'val'))


prepare_dataset(test_dir, os.path.join(output_dir, 'test'))

