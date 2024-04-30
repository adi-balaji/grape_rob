import pandas as pd
import matplotlib.pyplot as plt

# Load the data from gripper_frame_err.csv
data = pd.read_csv('gripper_frame_err.csv')

# Calculate average error for each direction
avg_x_err = data['x_err'].mean()
avg_y_err = data['y_err'].mean()
avg_z_err = data['z_err'].mean()

# Create scatter plot
plt.figure(figsize=(10, 6))

# Plot x_err
plt.scatter(['x']*len(data), data['x_err'], label='x_err', color='r')

# Plot y_err
plt.scatter(['y']*len(data), data['y_err'], label='y_err', color='g')

# Plot z_err
plt.scatter(['z']*len(data), data['z_err'], label='z_err', color='b')

# Plot bars for average errors
plt.bar(['x']*len(data), [avg_x_err], color='r', alpha=0.01, zorder=0)
plt.bar(['y']*len(data), [avg_y_err], color='g', alpha=0.01, zorder=0)
plt.bar(['z']*len(data), [avg_z_err], color='b', alpha=0.01, zorder=0)

# Add labels and legend
plt.xlabel('Error Direction')
plt.ylabel('Absolute Error in Centimeters (cm)')
plt.title('Gripper Pose Error')

# Show plot
plt.savefig('gripper_pose_error.png')
plt.show()

print(f'Avergae gripper pose error [{avg_x_err},{avg_y_err},{avg_z_err}]')
