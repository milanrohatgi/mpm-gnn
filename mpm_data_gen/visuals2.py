import taichi as ti
import numpy as np
import time

# Initialize Taichi GUI
gui = ti.GUI("Taichi MLS-MPM-128", res=512, background_color=0x112F41)

# Hardcoded file path to the .npy file
file_path = '/Users/mrohatgi/Downloads/rollout_trajectory_part_100.npy'
data = np.load(file_path)

print(data[19][0:20])  # Print first 20 particles of the first frame

# Number of timesteps
num_timesteps = data.shape[0]

# Current frame index
current_frame = 0

# Function to display the current frame
def display_frame(frame):
    gui.clear()
    gui.circles(
        data[frame],
        radius=1.5
    )
    gui.text(content=f'Frame: {frame}/{num_timesteps-1}', pos=(0.05, 0.95), color=0xFFFFFF)
    gui.show()

# Main loop
while gui.running:
    # Display the current frame
    display_frame(current_frame)

    # Check for key presses
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key == ti.GUI.ESCAPE:
            gui.running = False
        elif e.key == ti.GUI.RIGHT:
            # Move to next frame
            current_frame = (current_frame + 1) % num_timesteps
        elif e.key == ti.GUI.LEFT:
            # Move to previous frame
            current_frame = (current_frame - 1) % num_timesteps
        elif e.key == ti.GUI.SPACE:
            # Play/pause automatic playback
            playing = True
            while playing and gui.running:
                display_frame(current_frame)
                current_frame = (current_frame + 1) % num_timesteps
                time.sleep(1)  # 1 second delay between frames
                for event in gui.get_events(ti.GUI.PRESS):
                    if event.key == ti.GUI.SPACE:
                        playing = False
                        break

    # Add a small delay to prevent the loop from running too fast
    time.sleep(0.01)