import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
mpl.use('TkAgg')
# Create a 3D plot
fig3d = plt.figure()
ax3d = fig3d.add_subplot(111, projection='3d')

# Create a 2D projection plot
fig2d = plt.figure()
ax2d = fig2d.add_subplot(111)

# Define the cube vertices
cube_vertices = np.array([
    [-1, -1, -1],
    [1, -1, -1],
    [1, 1, -1],
    [-1, 1, -1],
    [-1, -1, 1],
    [1, -1, 1],
    [1, 1, 1],
    [-1, 1, 1]
])

# Define the cube edges
cube_edges = np.array([
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7]
])

# Function to rotate the cube
def update(frame):
    ax3d.clear()
    ax2d.clear()

    # Apply rotation matrix around Y-axis
    angle = frame * np.pi / 180.0
    rotation_matrix = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])
    rotated_vertices = np.dot(cube_vertices, rotation_matrix)

    # Plot the rotated cube in 3D
    for edge in cube_edges:
        ax3d.plot(rotated_vertices[edge, 0], rotated_vertices[edge, 1], rotated_vertices[edge, 2], 'b-')

    # Set 3D plot limits and labels
    ax3d.set_xlim(-2, 2)
    ax3d.set_ylim(-2, 2)
    ax3d.set_zlim(-2, 2)
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')

    # Project the rotated cube to 2D
    projected_vertices = rotated_vertices[:, :2]

    # Plot the projected cube in 2D
    for edge in cube_edges:
        ax2d.plot(projected_vertices[edge, 0], projected_vertices[edge, 1], 'b-')

    # Set 2D plot limits and labels
    ax2d.set_xlim(-2, 2)
    ax2d.set_ylim(-2, 2)
    ax2d.set_aspect('equal')
    ax2d.set_xlabel('X')
    ax2d.set_ylabel('Y')

# Create the 3D animation
animation3d = FuncAnimation(fig3d, update, frames=np.arange(0, 360, 10), interval=100)

# Create the 2D animation
animation2d = FuncAnimation(fig2d, update, frames=np.arange(0, 360, 10), interval=100)

# Display the plots
plt.show()