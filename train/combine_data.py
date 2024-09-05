import os
import h5py
import numpy as np
from tqdm import tqdm

def combine_hdf5_files(input_dir, output_file):
    h5_files = [f for f in os.listdir(input_dir) if f.endswith('.h5')]
    
    with h5py.File(output_file, 'w') as out_file:
        graph_counter = 0
        for h5_file in tqdm(h5_files, desc="Processing files"):
            file_path = os.path.join(input_dir, h5_file)
            with h5py.File(file_path, 'r') as in_file:
                for graph_name in in_file.keys():
                    new_graph_name = f'graph_{graph_counter}'
                    graph = in_file[graph_name]
                    
                    for dataset_name, dataset in graph.items():
                        out_file.create_dataset(f'{new_graph_name}/{dataset_name}', data=dataset[:])
                    
                    graph_counter += 1

    print(f"Combined file saved to {output_file}")
    print(f"Total number of graphs: {graph_counter}")

# edit paths
train_input_dir = '/scratch/users/mrohatgi/MPM_processed/train'
val_input_dir = '/scratch/users/mrohatgi/MPM_processed/val'
train_output_file = '/scratch/users/mrohatgi/MPM_processed/combined_train_flat.h5'
val_output_file = '/scratch/users/mrohatgi/MPM_processed/combined_val_flat.h5'

combine_hdf5_files(train_input_dir, train_output_file)

combine_hdf5_files(val_input_dir, val_output_file)

print("All data combined successfully.")
