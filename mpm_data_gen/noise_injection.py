import os
import numpy as np

sigma_v = 0.00015
source_dir = '/Users/mrohatgi/MPM_TRAINING'
target_dir = '/Users/mrohatgi/MPM_TRAIN_NOISE'
n_files = 210

def add_noise_and_adjust(data):
    n_timesteps, n_particles, _ = data.shape
    noise_accumulation = np.zeros((n_particles, 2))

    for time_step in range(1, n_timesteps):
        noise = np.random.normal(0, sigma_v, (n_particles, 2))

        noise_accumulation += noise

        for i in range(n_particles):
            velocity = data[time_step][i] - data[time_step - 1][i]
            perturbed_velocity = velocity + noise_accumulation[i]
            new_position = data[time_step - 1][i] + perturbed_velocity
            data[time_step][i] = new_position


for run_number in range(1, n_files + 1):
    file_name = f'run_{run_number}.npz'
    source_path = os.path.join(source_dir, file_name)

    if os.path.exists(source_path):
        data = np.load(source_path)['data']
        add_noise_and_adjust(data)
        target_path = os.path.join(target_dir, file_name)
        np.savez_compressed(target_path, data=data)
        print(f"Processed and saved: {file_name}")
    else:
        print(f"File not found: {source_path}")