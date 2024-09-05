import taichi as ti
import numpy as np
import time

# Initialize Taichi GUI
gui = ti.GUI("Taichi MLS-MPM-128", res=512, background_color=0x112F41)

# Hardcoded file path to the .npz file
file_path = '/Users/mrohatgi/MPM_TRAINING/run_270.npz'  # Replace with your actual file path

# Load the .npz file
data = np.load(file_path)['data']

file_path = '/Users/mrohatgi/Downloads/rollout_trajectory_part_100.npy'
data = np.load(file_path)

print(data[20][400:450])
# Number of timesteps
num_timesteps = data.shape[0]

# Play through all timesteps
for t in range(0, num_timesteps, 16):
    gui.circles(
        data[t],
        radius=1.5
    )


    gui.show()
# Keep the window open after the animation ends
while gui.running:
    gui.show()
